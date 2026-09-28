import argparse, os, signal, subprocess, sys, threading, time
from pathlib import Path
from .common import PROJECT, atomic, code_manifest, load, log_event, sha

def pump(pipe,run,rank):
    number=0;size=0;handle=None
    try:
        for line in iter(lambda:pipe.read1(65536),b""):
            if handle is None or size+len(line)>=8_000_000:
                if handle:handle.close()
                path=Path(run)/"logs"/f"worker_{rank}_{time.time_ns()}_{number:03d}.log"
                path.parent.mkdir(parents=True,exist_ok=True);handle=path.open("wb");number+=1;size=0
            handle.write(line);handle.flush();size+=len(line)
            sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
    finally:
        if handle:handle.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",help="gsm8k, existing_discovery, or public JSONL path; must be chosen explicitly")
    p.add_argument("--gold",help="Protected gold JSONL for a custom dataset")
    p.add_argument("--run-dir",default=str(PROJECT/"runs/main"))
    p.add_argument("--analyze-only",action="store_true")
    p.add_argument("--cpu-only",action="store_true",help="Tests only: no model/data download or inference")
    a=p.parse_args();run=Path(a.run_dir).resolve()
    if a.cpu_only:
        subprocess.run([sys.executable,"-m","pytest","-q",str(PROJECT/"tests")],check=True,cwd=PROJECT);return
    import fcntl
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/".orchestrator.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("This run directory already has an active orchestrator")
    from .finalize import finalize
    if a.analyze_only:
        finalize(run);return
    if not a.dataset:raise ValueError("Select the scientific population: --dataset gsm8k OR --dataset existing_discovery OR --dataset public.jsonl --gold gold.jsonl")
    from .prepare import prepare,hardware
    device_info=hardware()
    model_dir,_=prepare(run,a.dataset,a.gold)
    atomic(run/"hardware"/f"{time.time_ns()}.json",device_info)
    atomic(run/"STATUS.json",{"status":"RUNNING","dataset":a.dataset,"updated_unix":time.time()})
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    physical=visible.split(",") if visible else ["0","1","2","3"]
    if len(physical)!=4:raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly four GPUs")
    children=[];threads=[]
    def interrupted(sig,frame):raise KeyboardInterrupt(f"Signal {sig}")
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for rank in range(4):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=physical[rank],HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1",PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false")
            env["HF_MODULES_CACHE"]=str(PROJECT/".cache/hf_modules"/sha(run/"RUN_BINDING.json")/str(rank))
            env.pop("HF_TOKEN",None)
            command=[sys.executable,"-m","llada_study.worker","--run",str(run),"--model-dir",model_dir,"--rank",str(rank)]
            child=subprocess.Popen(command,env=env,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            children.append(child)
            thread=threading.Thread(target=pump,args=(child.stdout,run,rank),daemon=True);thread.start();threads.append(thread)
        last=0
        while any(c.poll() is None for c in children):
            failed=[i for i,c in enumerate(children) if c.poll() not in (None,0)]
            if failed:raise RuntimeError(f"Worker failure {failed}; stopping only this job workers, preserving every completed checkpoint")
            time.sleep(2)
            if time.monotonic()-last>=60:
                statuses=[]
                for rank in range(4):
                    file=run/f"STATUS_WORKER_{rank}.json"
                    statuses.append(load(file) if file.exists() else {"status":"STARTING"})
                atomic(run/"PROGRESS.json",{"workers":statuses,"updated_unix":time.time()})
                print("HEARTBEAT",[(x.get("completed",0),x.get("total"),x["status"]) for x in statuses],flush=True)
                last=time.monotonic()
        for t in threads:t.join()
        if any(c.returncode for c in children):
            raise RuntimeError("At least one worker failed. Completed realizations are preserved. Inspect events/ and logs/; rerun the same command.")
        finalize(run)
    except BaseException:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        raise

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        import traceback
        # The wrapper also preserves a bounded full console log.
        print(traceback.format_exc(),flush=True)
        sys.exit(130 if isinstance(exc,KeyboardInterrupt) else 1)
