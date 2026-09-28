import io,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from llada_study import common,finalize
from llada_study.common import atomic,canonical,digest,sha,seed_for,verify_realization
from llada_study.worker import save_realization
from llada_study.measurement import summarize_trace

class Tokenizer:
    eos_token_id=99
    def decode(self,ids,skip_special_tokens=False):
        return "Answer: 1" if 7 in ids else "Answer: 0"

def fixture_run(tmp_path,monkeypatch,n=4):
    from llada_study.dataset import prepare_dataset
    config=json.loads((common.PROJECT/"config.json").read_text())
    config["bootstrap_replicates"]=20
    public=[];gold=[]
    for i in range(n):
        uid=digest(["fixture",i])
        public.append({"uid":uid,"family":"ARITHMETIC","prompt":"synthetic-only "+str(i)})
        gold.append({"uid":uid,"gold":"1"})
    inp=tmp_path/"public.jsonl";answers=tmp_path/"gold.jsonl"
    atomic(inp,b"".join(canonical(r)+b"\n" for r in public))
    atomic(answers,b"".join(canonical(r)+b"\n" for r in gold))
    run=tmp_path/"run"
    seal=prepare_dataset(str(inp),run,config,str(answers))
    code=common.code_manifest()
    binding={"code":code,"config":config,"dataset_sha256":digest(seal),"model_receipt_sha256":"synthetic"}
    atomic(run/"RUN_BINDING.json",binding)
    atomic(run/"MODEL_RECEIPT.json",{"synthetic":True})
    atomic(run/"ENVIRONMENT.json",{"synthetic":True})
    binding_hash=sha(run/"RUN_BINDING.json")
    source=json.loads((common.PROJECT/"model_pin.json").read_text())["native_source_sha256"]["generate.py"]
    for i,row in enumerate(public):
        for rep in range(16):
            ident={"uid":row["uid"],"replicate":rep,"seed":seed_for(row["uid"],rep,config["master_seed"]),
                   "binding_sha256":binding_hash,"public_sha256":digest(row)}
            masks=np.array([[1,1,1,1],[0,1,1,1],[0,0,1,1],[0,0,0,1]],bool)
            ent=np.full((4,4),.7,dtype=np.float32)
            prop=np.ones((4,4),dtype=np.int32)
            if rep%3:
                ent[1:]-=.1;ent[2:]-=.1
                prop[1:,1:]=2
            trace={"entropy":ent,"proposals":prop,"masks":masks,"canvas":np.zeros((4,4),dtype=np.int32)}
            process=summarize_trace(ent,prop,masks)
            native={"source_sha256":source,"forward_calls":128,"final_mask_count":0}
            save_realization(common.realization_dir(run,row["uid"],rep),ident,[7 if (rep+i)%3 else 8,99],trace,process,Tokenizer(),[1],native,1.0)
    # Test config lookup while preserving real source/build identity.
    original=finalize.load
    monkeypatch.setattr(finalize,"load",lambda p:config if Path(p)==common.PROJECT/"config.json" else original(p))
    return run,public,config

def test_seed_independent_of_shard_and_order():
    assert seed_for("a"*64,2,1)==seed_for("a"*64,2,1)
    assert len({seed_for("a"*64,r,1) for r in range(16)})==16

def test_bounded_atomic_and_parts(tmp_path):
    with pytest.raises(ValueError):atomic(tmp_path/"oversize",b"x"*common.LIMIT)
    parts=common.write_jsonl_parts(tmp_path,"rows",[{"x":"a"*30}]*20,limit=200)
    assert len(parts)>1
    assert sum(len(common.load_jsonl(tmp_path/p)) for p in parts)==20

def test_completion_corruption_detected(tmp_path,monkeypatch):
    run,public,config=fixture_run(tmp_path,monkeypatch,n=2)
    folder=common.realization_dir(run,public[0]["uid"],0)
    seal=common.load(folder/"COMPLETE.json")
    expected={k:seal[k] for k in ("uid","replicate","seed","binding_sha256","public_sha256")}
    verify_realization(folder,expected)
    atomic(folder/"response.json",{"changed":True})
    with pytest.raises(RuntimeError,match="integrity"):verify_realization(folder,expected)

def test_process_seal_before_gold_and_complete_exports(tmp_path,monkeypatch):
    run,public,_=fixture_run(tmp_path,monkeypatch)
    from llada_study import dataset
    original=dataset.load_gold
    def guarded(path):
        assert (Path(path)/"PROCESS_FEATURES_SEAL.json").exists()
        return original(path)
    monkeypatch.setattr(dataset,"load_gold",guarded)
    result=finalize.finalize(run)
    assert result["status"]=="COMPLETE"
    assert result["n_realizations"]==64
    assert all(p.stat().st_size<10_000_000 for p in run.rglob("*") if p.is_file())
    for part in result["return_parts"]:
        import zipfile
        with zipfile.ZipFile(run/"SEND_TO_CODEX"/part["file"]) as z:assert z.testzip() is None
    assert finalize.finalize(run)==result

def test_missing_population_never_joins_gold(tmp_path,monkeypatch):
    run,public,_=fixture_run(tmp_path,monkeypatch,n=2)
    folder=common.realization_dir(run,public[0]["uid"],0)
    (folder/"COMPLETE.json").unlink()
    with pytest.raises(RuntimeError,match="Incomplete"):finalize.finalize(run)
    assert not (run/"PROCESS_FEATURES_SEAL.json").exists()
    assert not (run/"COMPLETE.json").exists()

def test_completed_run_still_detects_raw_tampering(tmp_path,monkeypatch):
    run,public,_=fixture_run(tmp_path,monkeypatch,n=2)
    finalize.finalize(run)
    folder=common.realization_dir(run,public[0]["uid"],0)
    atomic(folder/"trace.npz",b"corrupt")
    with pytest.raises(RuntimeError,match="integrity"):finalize.finalize(run)

def test_oversized_single_console_line_is_split(tmp_path,monkeypatch):
    from llada_study.run import pump
    class Output:
        buffer=io.BytesIO()
    monkeypatch.setattr("sys.stdout",Output())
    pump(io.BufferedReader(io.BytesIO(b"x"*9_100_000+b"\n")),tmp_path,0)
    logs=list((tmp_path/"logs").glob("*.log"))
    assert len(logs)>=2
    assert sum(p.stat().st_size for p in logs)==9_100_001
    assert all(p.stat().st_size<9_000_000 for p in logs)

def test_environment_checker_detects_compatible_but_wrong_version(monkeypatch):
    from llada_study import environment_check as e
    monkeypatch.setattr(e.sys,"version_info",(3,11,11))
    expected={}
    for line in (common.PROJECT/"requirements.txt").read_text().splitlines():
        if line.strip():
            name,version=line.split("==");expected[name]=version
    expected["torch"]="2.5.1+cu124"
    monkeypatch.setattr(e.metadata,"version",lambda name:expected[name])
    assert e.failures()==[]
    expected["torch"]="2.5.1+cu118"
    assert any("torch" in p for p in e.failures())
