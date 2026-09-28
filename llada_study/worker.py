"""Inference worker: imports no gold/parser/analysis interface."""
import argparse, io, os, signal, time
from pathlib import Path
import numpy as np
from .common import PROJECT, atomic, digest, load, load_jsonl, log_event, realization_dir, seed_for, sha, verify_realization
from .measurement import summarize_trace
from .native import load_native_model, equivalence_check

def public_rows(run):
    seal=load(Path(run)/"data/DATASET_SEAL.json")
    rows=[]
    for name in seal["public_files"]:
        p=Path(run)/name
        if sha(p)!=seal["files"][name]["sha256"]:raise RuntimeError("Public input hash mismatch")
        rows.extend(load_jsonl(p))
    return sorted(rows,key=lambda x:x["uid"])

def save_realization(folder,identity,final_ids,trace,process,tokenizer,prompt_ids,metadata,seconds):
    folder=Path(folder)
    stream=io.BytesIO();np.savez_compressed(stream,**trace)
    atomic(folder/"trace.npz",stream.getvalue())
    atomic(folder/"process.json",{**identity,**process,"native":metadata,"seconds":seconds,
           "prompt_token_count":len(prompt_ids),"prompt_ids_sha256":digest(prompt_ids)})
    # Full IDs preserved; scoring uses the prefix strictly before first native EOS.
    eos=tokenizer.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos]) if eos is not None else set()
    stop=next((i for i,v in enumerate(final_ids) if v in eos_ids),len(final_ids))
    atomic(folder/"response.json",{"final_ids":final_ids,"first_eos_position":None if stop==len(final_ids) else stop,
        "eos_ids":sorted(eos_ids),"text_before_eos":tokenizer.decode(final_ids[:stop],skip_special_tokens=False),
        "full_text":tokenizer.decode(final_ids,skip_special_tokens=False)})
    files={name:{"sha256":sha(folder/name),"bytes":(folder/name).stat().st_size}
           for name in ("trace.npz","process.json","response.json")}
    atomic(folder/"COMPLETE.json",{**identity,"files":files},immutable=True)
    verify_realization(folder,identity)

def run_worker(run,model_dir,rank):
    import fcntl
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    lock=(run/f".worker_{rank}.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError(f"Worker {rank} already active; refusing concurrent writes")
    config=load(PROJECT/"config.json");binding=sha(run/"RUN_BINDING.json")
    prompt_receipts={row["uid"]:row for row in load(run/"TOKENIZATION_PREFLIGHT.json")["questions"]}
    # Balanced deterministic allocation; the seed never depends on allocation.
    rows=sorted(public_rows(run),key=lambda x:digest(x["uid"]))[rank::4]
    pending=[];completed=0;total=len(rows)*16
    for row in rows:
        for rep in range(16):
            ident={"uid":row["uid"],"replicate":rep,"seed":seed_for(row["uid"],rep,config["master_seed"]),
                   "binding_sha256":binding,"public_sha256":digest(row)}
            folder=realization_dir(run,row["uid"],rep)
            if (folder/"COMPLETE.json").is_file():
                verify_realization(folder,ident);completed+=1
            else:pending.append((row,ident,folder))
    atomic(run/f"STATUS_WORKER_{rank}.json",{"status":"LOADING" if pending else "COMPLETE","completed":completed,"total":total})
    if not pending:
        print(f"WORKER {rank} all {total} verified; no GPU loading",flush=True);return
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ.pop("HF_TOKEN",None)
    # Bound peak CPU RAM and avoid concurrent cold checkpoint initialization.
    with (run/".model_loading.lock").open("a") as model_lock:
        fcntl.flock(model_lock,fcntl.LOCK_EX)
        runner=load_native_model(model_dir,PROJECT/"vendor/generate.py")
        fcntl.flock(model_lock,fcntl.LOCK_UN)
    gate=equivalence_check(runner,config["native"])
    atomic(run/"gpu_preflight"/f"worker_{rank}_{time.time_ns()}.json",gate)
    started=time.monotonic();durations=[];last_uid=None
    for row,identity,folder in pending:
        if row["uid"]!=last_uid:
            prompt_ids=runner.tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]}],
                          tokenize=True,add_generation_prompt=True)
            if not isinstance(prompt_ids,list) or not prompt_ids:raise RuntimeError("Invalid prompt tokens")
            if digest(prompt_ids)!=prompt_receipts[row["uid"]]["ids_sha256"]:raise RuntimeError("Prompt tokenization differs from preflight")
            context_limit=getattr(runner.model.config,"max_position_embeddings",4096)
            if len(prompt_ids)+config["native"]["gen_length"]>context_limit:
                raise RuntimeError(f"Prompt exceeds model context: {row['uid']}; never truncate silently")
            last_uid=row["uid"]
        if folder.exists() and any(folder.iterdir()):
            archive=folder.with_name(folder.name+f".interrupted.{time.time_ns()}")
            os.replace(folder,archive)
        t=time.monotonic()
        ids,trace=runner.run(prompt_ids,identity["seed"],config["native"])
        process=summarize_trace(trace["entropy"],trace["proposals"],trace["masks"])
        # Native residual masks, if any, are recorded; never replace an unfavorable realization.
        elapsed=time.monotonic()-t
        save_realization(folder,identity,ids,trace,process,runner.tokenizer,prompt_ids,runner.last_run_metadata,elapsed)
        completed+=1;durations.append(elapsed)
        recent=float(np.median(durations[-64:]))
        status={"status":"RUNNING","completed":completed,"total":total,"uid":row["uid"],"replicate":identity["replicate"],
                "seconds_recent_median":recent,"estimated_remaining_seconds":recent*(total-completed),
                "session_elapsed_seconds":time.monotonic()-started,"updated_unix":time.time()}
        atomic(run/f"STATUS_WORKER_{rank}.json",status)
        print(f"WORKER {rank} {completed}/{total} saved; recent {recent:.2f}s/run; ETA {status['estimated_remaining_seconds']/3600:.2f}h",flush=True)
    atomic(run/f"STATUS_WORKER_{rank}.json",{"status":"COMPLETE","completed":completed,"total":total,"updated_unix":time.time()})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--model-dir",required=True);p.add_argument("--rank",type=int,required=True)
    a=p.parse_args()
    try:run_worker(a.run,a.model_dir,a.rank)
    except BaseException as exc:
        import traceback
        log_event(a.run,{"status":"WORKER_FAILED","rank":a.rank,"exception":type(exc).__name__,
                       "message":str(exc),"traceback":traceback.format_exc()})
        raise
