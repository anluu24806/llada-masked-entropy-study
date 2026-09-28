"""Input-only preparation. No scientific forward calls or outcome analysis."""
import os, subprocess, sys, time
from pathlib import Path
from .common import PROJECT, atomic, canonical, code_manifest, digest, load, sha

def hardware():
    import torch
    n=torch.cuda.device_count()
    if n!=4:raise RuntimeError(f"Exactly 4 visible H100 GPUs required; found {n}. Set CUDA_VISIBLE_DEVICES to allocated GPUs.")
    result=[]
    for i in range(n):
        p=torch.cuda.get_device_properties(i)
        if "H100" not in p.name:raise RuntimeError(f"Expected H100, device {i} is {p.name}")
        result.append({"logical_device":i,"name":p.name,"total_bytes":p.total_memory})
    return {"gpus":result,"torch":torch.__version__,"cuda":torch.version.cuda,
            "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","all")}

def token():
    secret=os.environ.get("HF_TOKEN")
    if not secret:
        path=Path(os.environ.get("HF_TOKEN_FILE",str(Path.home()/".config/huggingface/token")))
        if path.is_file():secret=path.read_text().strip()
    if not secret:raise RuntimeError("Set HF_TOKEN or a mode-600 HF_TOKEN_FILE. Credentials are never written to study artifacts.")
    from huggingface_hub import HfApi
    HfApi(token=secret).whoami()
    return secret

def verify_model(model_dir,pin):
    import hashlib
    entries={}
    for row in pin["model_files"]:
        p=Path(model_dir)/row["rfilename"]
        if not p.is_file() or p.stat().st_size!=row["size"]:raise RuntimeError(f"Model size mismatch: {row['rfilename']}")
        actual=sha(p)
        if "lfs" in row:
            if actual!=row["lfs"]["sha256"]:raise RuntimeError(f"Model SHA mismatch: {row['rfilename']}")
        else:
            h=hashlib.sha1()
            h.update(f"blob {p.stat().st_size}\0".encode())
            with p.open("rb") as f:
                for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
            if h.hexdigest()!=row["blobId"]:raise RuntimeError(f"Model git blob mismatch: {row['rfilename']}")
        entries[row["rfilename"]]={"sha256":actual,"bytes":p.stat().st_size}
    return entries

def prepare(run,dataset,gold_path=None):
    from .environment_check import check
    check()
    from .dataset import prepare_dataset
    run=Path(run);config=load(PROJECT/"config.json");pin=load(PROJECT/"model_pin.json")
    if config["replicates"]!=16:raise RuntimeError("This protocol requires exactly 16 realizations per question")
    for field,value in config["native"].items():
        if pin["operating_point"][field]!=value:raise RuntimeError(f"Native configuration drift: {field}")
    if sha(PROJECT/"vendor/generate.py")!=pin["native_source_sha256"]["generate.py"]:
        raise RuntimeError("Pinned native source mismatch")
    os.environ["HF_HUB_DISABLE_XET"]="1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
    credential=token()
    # Data prefetch may use the token via environment; no credential goes into artifacts.
    os.environ["HF_TOKEN"]=credential
    dataset_seal=prepare_dataset(dataset,run,config,gold_path)
    from huggingface_hub import snapshot_download
    print("PREFETCH pinned LLaDA checkpoint; shared once by all four workers",flush=True)
    model_dir=snapshot_download(pin["model_repository"],revision=pin["model_revision"],
            allow_patterns=[r["rfilename"] for r in pin["model_files"]],token=credential,max_workers=4)
    credential=None
    model_files=verify_model(model_dir,pin)
    # Tokenize every input before any scientific model forward; never discover
    # an overlong prompt halfway through a multi-day job or truncate it silently.
    from transformers import AutoTokenizer
    from .common import load_jsonl
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True,local_files_only=True)
    model_config=load(Path(model_dir)/"config.json")
    maximum=model_config.get("max_sequence_length",model_config.get("max_position_embeddings",4096))
    checks=[]
    for relative in dataset_seal["public_files"]:
        for row in load_jsonl(run/relative):
            ids=tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]}],tokenize=True,add_generation_prompt=True)
            if not ids or 126336 in ids:raise RuntimeError(f"Invalid prompt/MASK in input: {row['uid']}")
            if len(ids)+config["native"]["gen_length"]>maximum:raise RuntimeError(f"Prompt too long: {row['uid']}; no truncation permitted")
            checks.append({"uid":row["uid"],"tokens":len(ids),"ids_sha256":digest(ids)})
    atomic(run/"TOKENIZATION_PREFLIGHT.json",{"status":"PASS","max_context":maximum,"questions":checks},immutable=True)
    del tokenizer
    import shutil
    completed=sum(1 for _ in (run/"realizations").glob("*/*/r[0-9][0-9]/COMPLETE.json"))
    needed=max(dataset_seal["uid_count"]*16-completed,0)*1_000_000+5_000_000_000
    free=shutil.disk_usage(run).free
    if free<needed:raise RuntimeError(f"Result disk needs approximately {needed/1e9:.1f} GB free; available {free/1e9:.1f} GB. Set --run-dir to a larger persistent volume.")
    atomic(run/"MODEL_RECEIPT.json",{"repository":pin["model_repository"],"revision":pin["model_revision"],
                                 "files":model_files},immutable=True)
    binding={"config":config,"code":code_manifest(),"dataset_sha256":digest(dataset_seal),
             "model_receipt_sha256":sha(run/"MODEL_RECEIPT.json"),
             "tokenization_sha256":sha(run/"TOKENIZATION_PREFLIGHT.json")}
    atomic(run/"RUN_BINDING.json",binding,immutable=True)
    receipt={"model_dir":model_dir,"binding_sha256":sha(run/"RUN_BINDING.json"),
             "prepared_time_unix":time.time(),"python":sys.version,
             "pip_freeze":subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines()}
    # Exact environment is checked on resume, not silently overwritten.
    atomic(run/"ENVIRONMENT.json",{"python_version":list(sys.version_info[:3]),"pip_freeze":receipt["pip_freeze"]},immutable=True)
    atomic(run/"LOCAL_PREPARATION.json",receipt)
    return model_dir,binding
