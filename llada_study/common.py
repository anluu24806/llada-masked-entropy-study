from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
LIMIT = 9_000_000
PROJECT = Path(__file__).resolve().parents[1]
def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")
def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b""): h.update(chunk)
    return h.hexdigest()
def atomic(path, data, immutable=False):
    path=Path(path)
    if not isinstance(data,bytes): data=canonical(data)+b"\n"
    if len(data)>=LIMIT: raise ValueError(f"Oversized result: {path.name}: {len(data)}")
    path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes()!=data: raise RuntimeError(f"Immutable artifact differs: {path}")
        return
    temp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temp.open("wb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    if os.name!="nt":
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def load(path):return json.loads(Path(path).read_text(encoding="utf8"))
def load_jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]
def write_jsonl_parts(folder,stem,rows,limit=8_000_000):
    folder=Path(folder); chunks=[]; buffer=bytearray()
    for row in rows:
        line=canonical(row)+b"\n"
        if len(line)>=limit:raise ValueError(f"Single {stem} row too large")
        if buffer and len(buffer)+len(line)>=limit:
            name=f"{stem}_{len(chunks):04d}.jsonl";atomic(folder/name,bytes(buffer));chunks.append(name);buffer.clear()
        buffer.extend(line)
    if buffer:
        name=f"{stem}_{len(chunks):04d}.jsonl";atomic(folder/name,bytes(buffer));chunks.append(name)
    return chunks
def code_manifest():
    files=[PROJECT/x for x in ("config.json","model_pin.json","requirements.txt","run_all.sh")]
    files+=sorted((PROJECT/"llada_study").glob("*.py"))
    files+=sorted((PROJECT/"vendor").glob("*.py"))
    return {p.relative_to(PROJECT).as_posix():sha(p) for p in files}
def seed_for(uid,replicate,master_seed):
    return int(digest(["LLADA_MASKED_ENTROPY_16_V1",master_seed,uid,replicate])[:16],16)%(2**63-1)
def realization_dir(run,uid,replicate):
    return Path(run)/"realizations"/uid[:2]/uid/f"r{replicate:02d}"
def verify_realization(folder,expected):
    folder=Path(folder);seal=load(folder/"COMPLETE.json")
    for key,value in expected.items():
        if seal.get(key)!=value:raise RuntimeError(f"Wrong completion identity {key}: {folder}")
    if set(seal["files"])!={"trace.npz","process.json","response.json"}:raise RuntimeError(f"Incomplete file set: {folder}")
    for name,info in seal["files"].items():
        p=folder/name
        if p.stat().st_size!=info["bytes"] or p.stat().st_size>=LIMIT or sha(p)!=info["sha256"]:
            raise RuntimeError(f"Checkpoint integrity failure: {p}")
    return seal
def log_event(run,event):
    atomic(Path(run)/"events"/f"{time.time_ns()}_{os.getpid()}.json",{"time_unix":time.time(),**event})
