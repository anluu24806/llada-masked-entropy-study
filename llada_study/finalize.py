"""Verify and seal all process rows BEFORE importing correctness/analysis."""
import csv, io, re, time, zipfile
from pathlib import Path
import numpy as np
from .common import PROJECT,LIMIT,atomic,canonical,code_manifest,digest,load,load_jsonl,realization_dir,seed_for,sha,verify_realization,write_jsonl_parts

def csv_parts(folder,stem,rows):
    rows=list(rows)
    if not rows:return []
    keys=list(dict.fromkeys(k for row in rows for k in row))
    def encoded(row):
        out=io.StringIO(newline="");w=csv.writer(out,lineterminator="\n")
        w.writerow([canonical(row.get(k)).decode() if isinstance(row.get(k),(list,dict)) else row.get(k) for k in keys])
        return out.getvalue().encode()
    out=io.StringIO(newline="");csv.writer(out,lineterminator="\n").writerow(keys);header=out.getvalue().encode()
    names=[];part=bytearray(header)
    for row in rows:
        line=encoded(row)
        if len(line)+len(header)>=8_000_000:raise RuntimeError("Oversized CSV row")
        if len(part)+len(line)>=8_000_000:
            name=f"{stem}_{len(names):04d}.csv";atomic(Path(folder)/name,bytes(part));names.append(name);part=bytearray(header)
        part.extend(line)
    if len(part)>len(header):
        name=f"{stem}_{len(names):04d}.csv";atomic(Path(folder)/name,bytes(part));names.append(name)
    return names

def _inputs(run):
    seal=load(run/"data/DATASET_SEAL.json")
    for name,info in seal["files"].items():
        if sha(run/name)!=info["sha256"]:raise RuntimeError(f"Dataset hash mismatch: {name}")
    rows=[]
    for name in seal["public_files"]:rows.extend(load_jsonl(run/name))
    return seal,sorted(rows,key=lambda x:x["uid"])

def process_closure(run):
    run=Path(run);binding=load(run/"RUN_BINDING.json")
    if binding["code"]!=code_manifest():raise RuntimeError("Code/config changed since this run; preserve it and use a new run directory")
    if "tokenization_sha256" in binding and sha(run/"TOKENIZATION_PREFLIGHT.json")!=binding["tokenization_sha256"]:raise RuntimeError("Prompt preflight receipt changed")
    config=binding["config"];dataset,public=_inputs(run)
    if digest(dataset)!=binding["dataset_sha256"]:raise RuntimeError("Dataset seal changed")
    binding_hash=sha(run/"RUN_BINDING.json")
    expected={(r["uid"],rep) for r in public for rep in range(16)}
    found=set()
    for p in (run/"realizations").glob("*/*/r[0-9][0-9]/COMPLETE.json"):
        pair=(p.parent.parent.name,int(p.parent.name[1:]))
        if pair in found:raise RuntimeError("Duplicate canonical realization")
        found.add(pair)
    if found!=expected:
        raise RuntimeError(f"Incomplete/unexpected population: expected={len(expected)} found={len(found)}, missing={len(expected-found)}, extra={len(found-expected)}")
    process_rows=[];index=[];depth={}
    for count,row in enumerate(public):
        for rep in range(16):
            identity={"uid":row["uid"],"replicate":rep,"seed":seed_for(row["uid"],rep,config["master_seed"]),
                      "binding_sha256":binding_hash,"public_sha256":digest(row)}
            folder=realization_dir(run,row["uid"],rep)
            seal=verify_realization(folder,identity);pr=load(folder/"process.json")
            for key,value in identity.items():
                if pr[key]!=value:raise RuntimeError("Process identity mismatch")
            if pr["native"]["source_sha256"]!=load(PROJECT/"model_pin.json")["native_source_sha256"]["generate.py"]:
                raise RuntimeError("Process native source changed")
            if pr["native"]["forward_calls"]!=config["native"]["steps"]:raise RuntimeError("Wrong denoising depth")
            process_rows.append({**identity,"family":row["family"],"features":pr["features"],
                    "diagnostics":pr["diagnostics"],"seconds":pr["seconds"],
                    "prompt_token_count":pr["prompt_token_count"],"final_mask_count":pr["native"]["final_mask_count"]})
            index.append({"uid":row["uid"],"replicate":rep,"path":folder.relative_to(run).as_posix(),"seal_sha256":sha(folder/"COMPLETE.json"),"files":seal["files"]})
            for step in pr["steps"]:
                if not step.get("eligible",False):continue
                d=depth.setdefault(step["step"],{"rates_sum":np.zeros(6,dtype=np.float64),"mask_fraction_sum":0.0,"common_count_sum":0.0,"n":0})
                d["rates_sum"]+=step["rates"];d["mask_fraction_sum"]+=step["mask_fraction"];d["common_count_sum"]+=step["common_count"];d["n"]+=1
        if (count+1)%50==0:print(f"CPU INTEGRITY {count+1}/{len(public)} questions",flush=True)
    frozen=run/"process_frozen"
    feature_files=write_jsonl_parts(frozen,"process_features",process_rows)
    index_files=write_jsonl_parts(frozen,"raw_index",index)
    depth_rows=[{"step":t,"n_runs":d["n"],"mean_rates":(d["rates_sum"]/d["n"]).tolist(),
        "mask_fraction":d["mask_fraction_sum"]/d["n"],"common_count":d["common_count_sum"]/d["n"]}
                for t,d in sorted(depth.items())]
    atomic(frozen/"DEPTH_PROCESS.json",depth_rows)
    files=feature_files+index_files+["DEPTH_PROCESS.json"]
    process_seal={"status":"PROCESS_FEATURES_SEALED","run_binding_sha256":binding_hash,
      "n_questions":len(public),"n_realizations":len(expected),"features":[f"process_frozen/{x}" for x in feature_files],
      "files":{x:sha(frozen/x) for x in files},
      "outcome_not_joined":True,"aggregation":"equal-step primary; common-mask-count weighted sensitivity"}
    atomic(run/"PROCESS_FEATURES_SEAL.json",process_seal,immutable=True)
    print("PROCESS_FEATURES_SEALED; correctness join is now permitted",flush=True)
    return process_rows,depth_rows,process_seal

def plots(out,rows,depth,analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=["t1 up+flip","t2 down+flip","t3 up+same","t4 down+same","tie+flip","tie+same"]
    fig,ax=plt.subplots(figsize=(9,4))
    matrix=np.asarray([x["mean_rates"] for x in depth])
    if len(matrix):ax.stackplot([x["step"] for x in depth],matrix.T,labels=labels)
    ax.set(xlabel="Denoising step (pre-unmask prediction)",ylabel="Fraction of shared masked positions",ylim=(0,1))
    ax.legend(loc="upper left",bbox_to_anchor=(1,1));fig.tight_layout()
    fig.savefig(out/"category_rates_by_depth.png",dpi=140);plt.close(fig)
    groups={}
    for r in rows:groups.setdefault(r["uid"],[]).append(r)
    pairs=[]
    for group in groups.values():
        c=[r["features"]["step"][1] for r in group if r["correct"]]
        w=[r["features"]["step"][1] for r in group if not r["correct"]]
        if c and w:pairs.append((np.mean(w),np.mean(c)))
    fig,ax=plt.subplots(figsize=(5,5))
    if pairs:
        x,y=np.asarray(pairs).T;ax.scatter(x,y,s=12,alpha=.5);lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],"k--",lw=1)
    ax.set(xlabel="Mean t2: incorrect runs within question",ylabel="Mean t2: correct runs within question",
           title=f"Mixed-outcome questions only (n={len(pairs)})")
    fig.tight_layout();fig.savefig(out/"within_question_t2.png",dpi=140);plt.close(fig)

def handoff(run,report_dir):
    target=run/"SEND_TO_CODEX";target.mkdir(exist_ok=True)
    # Every part is a standalone ZIP. Include compact tables, not the large raw corpus.
    paths=sorted(p for p in report_dir.rglob("*") if p.is_file())
    paths += [run/x for x in ("RUN_BINDING.json","MODEL_RECEIPT.json","ENVIRONMENT.json","PROCESS_FEATURES_SEAL.json","data/DATASET_SEAL.json")]
    paths += sorted((run/"gpu_preflight").glob("*.json"))
    if (run/"TOKENIZATION_PREFLIGHT.json").exists():paths.append(run/"TOKENIZATION_PREFLIGHT.json")
    parts=[];batch=[];size=0
    def flush():
        if not batch:return
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in batch:z.write(p,p.relative_to(run).as_posix())
        name=f"RESULTS_{len(parts)+1:03d}.zip";atomic(target/name,stream.getvalue())
        parts.append({"file":name,"bytes":(target/name).stat().st_size,"sha256":sha(target/name)})
    for path in paths:
        # Uncompressed sum+ZIP metadata is an upper bound with ample 1 MB headroom.
        if batch and size+path.stat().st_size>=7_500_000:flush();batch=[];size=0
        batch.append(path);size+=path.stat().st_size
    flush()
    atomic(target/"INDEX.json",{"status":"COMPLETE","parts":parts,"raw_data_retained_on_server":True,
            "instructions":"Send every RESULTS_*.zip and INDEX.json together. Each ZIP opens independently."})
    return parts

def finalize(run):
    run=Path(run)
    if (run/"COMPLETE.json").exists():
        complete=load(run/"COMPLETE.json")
        for name,expected in complete["files"].items():
            if sha(run/name)!=expected:raise RuntimeError(f"Final artifact changed: {name}")
        process_closure(run)
        print("COMPLETE: final artifacts AND all raw/provenance hashes verified; no inference/statistics rerun",flush=True);return complete
    process_rows,depth,seal=process_closure(run)
    # Gold first enters this layer strictly after the durable process seal exists.
    from .dataset import load_gold,parse_answer
    from .analysis import analyze
    gold=load_gold(run);rows=[];score_rows=[]
    for r in process_rows:
        response=load(realization_dir(run,r["uid"],r["replicate"])/"response.json")
        parsed=parse_answer(response["text_before_eos"],r["family"])
        answer=gold[r["uid"]]
        if isinstance(answer,dict):answer=answer["gold"]
        scored={**r,"correct":int(parsed["status"]=="PARSED" and parsed["identity"]==answer),"parseable":parsed["status"]=="PARSED"}
        rows.append(scored)
        score_rows.append({**scored,"prediction":parsed,"gold":answer,"response":response["text_before_eos"],"first_eos_position":response["first_eos_position"]})
    config=load(PROJECT/"config.json")
    print("CPU STATISTICS: all 16 realizations/question; bootstrap whole questions",flush=True)
    result=analyze(rows,config["bootstrap_replicates"],config["bootstrap_seed"])
    out=run/"results";out.mkdir(exist_ok=True)
    summary={"status":"COMPLETE","study":config["study"],"n_questions":seal["n_questions"],
       "n_realizations":seal["n_realizations"],"accuracy":float(np.mean([r["correct"] for r in rows])),
       "unparseable_fraction":float(np.mean([not r["parseable"] for r in rows])),
       "runs_with_residual_native_masks":sum(r["final_mask_count"]>0 for r in rows),
       "scientific_summary":result["summary"],"process_seal_sha256":sha(run/"PROCESS_FEATURES_SEAL.json")}
    atomic(out/"SUMMARY.json",summary)
    csv_parts(out,"question_results",result["problem_rows"]);csv_parts(out,"effect_results",result["effect_rows"])
    csv_parts(out,"mask_region_results",result["depth_rows"])
    write_jsonl_parts(out,"scored_realizations",score_rows)
    csv_parts(out,"per_step_category_means",depth)
    plots(out,rows,depth,result)
    primary=result["summary"]["primary"]
    report=f"""# LLaDA same-position masked entropy study

Status: COMPLETE. {seal['n_questions']} questions; {seal['n_realizations']} realizations; exactly 16 per question.
Dataset provenance: see data/DATASET_SEAL.json. Model/source/schedule: RUN_BINDING.json and MODEL_RECEIPT.json.

## Observed results

Final accuracy: {summary['accuracy']:.6f}. Unparseable fraction: {summary['unparseable_fraction']:.6f}.
Unparseable final answers count as task failures and are separately flagged. No semantic or LLM judge.
Runs with residual native MASKs: {summary['runs_with_residual_native_masks']}. They are retained and scored by the same final-answer parser.

Primary within-question T2/correctness result (full precision):

```json
{__import__('json').dumps(primary,indent=2,allow_nan=False)}
```

The coefficient describes association across 16 stochastic realizations of the same question.
The interval resamples questions, retaining all realizations. Significant association does not establish causation.
T1ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œT4 secondary tests, parent-adjusted T2, weighted sensitivity and between-question Spearman are in SUMMARY.json/effect tables.
An unavailable/degenerate estimate is explicitly reported; it is not zero or evidence for no association.

## Measurement and limits

The denominator is positions still masked at both adjacent pre-unmask predictions, on the full fixed 256-token canvas.
T1=entropy up/argmax flip; T2=entropy down/flip; T3=up/same; T4=down/same.
Entropy ties are retained in two additional categories. Entropy uses the raw FP32 full-vocabulary distribution, before sampling noise.
Primary rates average eligible denoising steps equally. Token-step rates are a separate sensitivity.
Raw argmax disagreement is not necessarily an edit to a committed output token.
Surviving masked positions are a selected subset. This estimates their dynamics, not dynamics of all original positions.
Mask count follows a fixed schedule and is confounded with depth; it is not a randomized causal variable.
Process features include the full generated canvas, including any eventual post-EOS positions. Final scoring stops before first EOS.
The task scorer is fixed before running. Parser failures are reported and retained.
All process features were durably sealed before the correctness join.

## Artifacts

question_results_*.csv: question accuracy and mean features.
effect_results_*.csv: primary and secondary estimates/intervals.
mask_region_results_*.csv: predeclared mask-fraction regions.
scored_realizations_*.jsonl: all run-level features, deterministic scores and responses.
Raw per-position entropy/argmax/mask/canvas arrays remain on the server under realizations/.
SEND_TO_CODEX contains independent ZIP parts under 9,000,000 bytes each. Send all parts and INDEX.json.
"""
    atomic(out/"REPORT.md",report.encode())
    # Examples chosen only from immutable UID/family identifiers, never scores.
    _,public=_inputs(run)
    by_family={}
    for item in public:by_family.setdefault(item["family"],[]).append(item)
    chosen=[item for family in sorted(by_family) for item in sorted(by_family[family],key=lambda x:digest(x["uid"]))[:2]]
    atomic(out/"raw_examples/PROMPTS.json",chosen)
    for item in chosen:
        for replicate in (0,1):
            source=realization_dir(run,item["uid"],replicate)
            destination=out/"raw_examples"/item["uid"]/f"r{replicate:02d}"
            for name in ("trace.npz","process.json","response.json","COMPLETE.json"):
                atomic(destination/name,(source/name).read_bytes())
    parts=handoff(run,out)
    # Audit every result artifact, including raw traces, archives and logs.
    oversized=[str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.stat().st_size>=10_000_000]
    if oversized:raise RuntimeError(f"User file-size limit violated: {oversized[:10]}")
    finals=[p for p in out.rglob("*") if p.is_file()]+[p for p in (run/"SEND_TO_CODEX").iterdir() if p.is_file()]
    complete={"status":"COMPLETE","n_questions":seal["n_questions"],"n_realizations":seal["n_realizations"],
              "files":{p.relative_to(run).as_posix():sha(p) for p in finals},
              "return_parts":parts,"completed_unix":time.time()}
    atomic(run/"COMPLETE.json",complete,immutable=True)
    print(f"COMPLETE. Send files in {run/'SEND_TO_CODEX'}. Raw data preserved on server.",flush=True)
    return complete
