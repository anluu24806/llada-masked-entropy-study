"""Prepare immutable public prompts and separately protected canonical gold.

The generation worker intentionally does not import this module. Gold loading
and answer parsing belong only to analysis after all raw/process seals exist.

Before run binding, the existing_discovery adapter groups exact (family, prompt)
pairs using public fields only and retains the lexicographically smallest UID.
All source UIDs remain in provenance aliases. Gold is checked only for agreement
within each group; disagreement fails instead of selecting a representative by
gold. This rule neither normalizes prompt text nor groups semantic similarities.
Other adapters reject repeated exact (family, prompt) questions.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

from .common import LIMIT, PROJECT, atomic, canonical, digest, load, load_jsonl, sha

FAMILIES = frozenset({"ARITHMETIC", "EVIDENCE_TRACKING", "SYMBOLIC_TRANSFORM", "GSM8K"})
GSM8K_REPOSITORY = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILENAME = "main/test-00000-of-00001.parquet"
GSM8K_COUNT = 1319
GSM8K_SUFFIX = "\nSolve the problem. You may show your reasoning. End with exactly one line: Answer: <number>"
_UID = re.compile(r"[0-9a-f]{64}\Z")
_INT = re.compile(r"[+-]?[0-9]+\Z")
_DECIMAL = re.compile(r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)\Z")
_ROOM = re.compile(r"[A-Za-z]+\Z")
_LIST = re.compile(r"[+-]?[0-9]+(?:\s*,\s*[+-]?[0-9]+)*\Z")
_PROTECTED_PUBLIC_KEYS = frozenset({"gold", "answer", "solution", "target", "correct_answer", "reference_answer", "oracle"})


def normalize_numeric(value) -> str:
    """Canonical exact decimal; accept signs and correctly grouped thousands.

    Scientific notation, units, percentages, fractions, nonfinite numbers, and
    binary floating-point inputs are outside the frozen numeric grammar.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("numeric identity must be an integer or decimal string")
    text = str(value).strip()
    if not _DECIMAL.fullmatch(text):
        raise ValueError("invalid signed decimal identity")
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation as error:
        raise ValueError("invalid decimal identity") from error
    if not number.is_finite():
        raise ValueError("numeric identity must be finite")
    if number == 0:
        return "0"
    result = format(number, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def _canonical_answer(value, family):
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError(f"unsupported family: {family}")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("answer identity must be a string or integer")
    answer = str(value).strip()
    if family == "GSM8K":
        return normalize_numeric(answer)
    if family == "ARITHMETIC" and _INT.fullmatch(answer):
        return str(int(answer))
    if family == "EVIDENCE_TRACKING" and _ROOM.fullmatch(answer):
        return answer.lower()
    if family == "SYMBOLIC_TRANSFORM" and _LIST.fullmatch(answer):
        return ",".join(str(int(item.strip())) for item in answer.split(","))
    raise ValueError(f"invalid answer grammar for {family}")


def parse_answer(text, family) -> dict:
    """Parse exactly one whole final nonempty ``Answer: ...`` line.

    Failure is explicit status U and identity None. The parser does not search
    reasoning for a likely answer, use gold to rescue an answer, or infer units.
    """
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError(f"unsupported family: {family}")
    if not isinstance(text, str):
        return {"status": "U", "identity": None, "reason": "response_not_text"}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    answer_lines = [index for index, line in enumerate(lines) if line.startswith("Answer:")]
    if len(answer_lines) != 1:
        return {"status": "U", "identity": None, "reason": "answer_line_count_not_one"}
    index = answer_lines[0]
    if index != len(lines) - 1:
        return {"status": "U", "identity": None, "reason": "answer_line_not_final"}
    candidate = lines[index][len("Answer:"):].strip()
    try:
        identity = _canonical_answer(candidate, family)
    except ValueError:
        return {"status": "U", "identity": None, "reason": "answer_grammar_mismatch"}
    return {"status": "PARSED", "identity": identity, "reason": None}


def _validate_rows(public_rows, gold_rows):
    if not public_rows:
        raise ValueError("public dataset must contain at least one question")
    public, by_uid = [], {}
    for index, row in enumerate(public_rows):
        if not isinstance(row, dict):
            raise ValueError(f"public row {index} must be an object")
        uid, family, prompt = row.get("uid"), row.get("family"), row.get("prompt")
        if not isinstance(uid, str) or not _UID.fullmatch(uid):
            raise ValueError(f"public row {index} requires a lowercase SHA256 uid")
        if uid in by_uid:
            raise ValueError(f"duplicate public uid: {uid}")
        if not isinstance(family, str) or family not in FAMILIES:
            raise ValueError(f"unsupported family: {family}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"public row {index} requires a nonempty prompt")
        if _PROTECTED_PUBLIC_KEYS.intersection(row):
            raise ValueError("public row contains protected answer fields")
        canonical(row)  # Reject NaN and other non-JSON metadata before any write.
        copied = dict(row)
        public.append(copied)
        by_uid[uid] = copied
    answers = {}
    for index, row in enumerate(gold_rows):
        if not isinstance(row, dict) or "uid" not in row or "gold" not in row:
            raise ValueError(f"gold row {index} requires uid and gold")
        uid = row["uid"]
        if not isinstance(uid, str) or not _UID.fullmatch(uid):
            raise ValueError(f"gold row {index} requires a lowercase SHA256 uid")
        if uid in answers:
            raise ValueError(f"duplicate gold uid: {uid}")
        if uid not in by_uid:
            raise ValueError(f"gold uid has no public question: {uid}")
        if "family" in row and row["family"] != by_uid[uid]["family"]:
            raise ValueError(f"gold/public family mismatch for {uid}")
        answers[uid] = _canonical_answer(row["gold"], by_uid[uid]["family"])
    if set(answers) != set(by_uid):
        raise ValueError("public and protected gold UID sets must match exactly")
    gold = [{"uid": row["uid"], "gold": answers[row["uid"]]} for row in public]
    return public, gold


def _deduplicate_existing(public, gold):
    """Apply the pre-run exact public-question rule to validated Discovery rows.

    Returned representatives and complete alias lists are stable under source
    row reordering. The source-byte hashes remain separately bound, so reordering
    source files still changes provenance and cannot silently alter a sealed run.
    """
    groups = {}
    for row in public:
        groups.setdefault((row["family"], row["prompt"]), []).append(row)
    answers = {row["uid"]: row["gold"] for row in gold}
    representatives = []
    aliases = {}
    for members in groups.values():
        # Representative selection is completed from public fields alone.
        ordered = sorted(members, key=lambda row: row["uid"])
        representative = ordered[0]
        uids = [row["uid"] for row in ordered]
        if len({answers[uid] for uid in uids}) != 1:
            raise ValueError("conflicting gold for duplicate exact (family, prompt) question: "
                             + ",".join(uids))
        representatives.append(representative)
        aliases[representative["uid"]] = uids
    representatives.sort(key=lambda row: row["uid"])
    aliases = {uid: aliases[uid] for uid in sorted(aliases)}
    unique_gold = [{"uid": row["uid"], "gold": answers[row["uid"]]}
                   for row in representatives]
    metadata = {
        "rule": "exact_family_prompt_lexicographic_uid_v1",
        "source_row_count": len(public),
        "distinct_question_count": len(representatives),
        "excess_source_rows": len(public) - len(representatives),
        "duplicate_group_count": sum(len(uids) > 1 for uids in aliases.values()),
        "alias_uids": aliases,
    }
    return representatives, unique_gold, metadata


def _reject_duplicate_questions(public):
    seen = {}
    for row in public:
        key = (row["family"], row["prompt"])
        if key in seen:
            raise ValueError("duplicate exact (family, prompt) question is not allowed for "
                             "GSM8K/custom inputs: " + seen[key] + "," + row["uid"])
        seen[key] = row["uid"]


def _gsm8k_rows(config):
    settings = config.get("gsm8k", {})
    pinned = {"repository": GSM8K_REPOSITORY, "revision": GSM8K_REVISION,
              "filename": GSM8K_FILENAME, "expected_questions": GSM8K_COUNT}
    for name, expected in pinned.items():
        if settings.get(name, expected) != expected:
            raise ValueError(f"GSM8K {name} differs from frozen dataset pin")
    suffix = settings.get("prompt_suffix", GSM8K_SUFFIX)
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("GSM8K prompt_suffix must be a nonempty string")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    source = Path(hf_hub_download(
        repo_id=GSM8K_REPOSITORY, repo_type="dataset",
        revision=GSM8K_REVISION, filename=GSM8K_FILENAME,
    ))
    table = parquet.read_table(source, columns=["question", "answer"])
    rows = table.to_pylist()
    if len(rows) != GSM8K_COUNT:
        raise ValueError(f"expected {GSM8K_COUNT} GSM8K questions, found {len(rows)}")
    public, gold = [], []
    for index, row in enumerate(rows):
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
            raise ValueError(f"invalid GSM8K row {index}")
        if answer.count("####") != 1:
            raise ValueError(f"GSM8K row {index} requires one gold delimiter")
        uid = digest({"dataset": GSM8K_REPOSITORY, "revision": GSM8K_REVISION,
                      "split": "main/test", "index": index, "question": question})
        public.append({"uid": uid, "family": "GSM8K", "prompt": question + suffix,
                       "dataset_index": index})
        gold.append({"uid": uid, "gold": normalize_numeric(answer.split("####")[1].strip())})
    provenance = {"kind": "gsm8k", **pinned, "source_sha256": sha(source),
                  "prompt_suffix": suffix, "prompt_suffix_sha256": digest(suffix.encode("utf8")),
                  "uid_definition": "sha256(canonical_json(dataset,revision,split,index,question))"}
    return public, gold, provenance


def _incoming(name_or_public_path, config, gold_path):
    name = str(name_or_public_path)
    if name == "gsm8k":
        if gold_path is not None:
            raise ValueError("gsm8k uses its pinned source gold")
        return _gsm8k_rows(config)
    if name == "existing_discovery":
        if gold_path is not None:
            raise ValueError("existing_discovery uses the bundled protected gold")
        folder = PROJECT / "data" / "existing_discovery"
        public_path, protected_path = folder / "public.jsonl", folder / "gold.jsonl"
        provenance_path = folder / "PROVENANCE.json"
        provenance = {"kind": name, "public_input_sha256": sha(public_path),
                      "gold_input_sha256": sha(protected_path),
                      "bundled_provenance_sha256": sha(provenance_path),
                      "bundled_provenance": load(provenance_path)}
    else:
        if gold_path is None:
            raise ValueError("custom public JSONL requires a separate gold_path")
        public_path, protected_path = Path(name_or_public_path), Path(gold_path)
        if public_path.resolve() == protected_path.resolve():
            raise ValueError("public and protected gold inputs must be separate files")
        provenance = {"kind": "custom_jsonl", "public_input_path": str(public_path.resolve()),
                      "gold_input_path": str(protected_path.resolve()),
                      "public_input_sha256": sha(public_path), "gold_input_sha256": sha(protected_path)}
    return load_jsonl(public_path), load_jsonl(protected_path), provenance


def _parts(stem, rows):
    parts, buffer = {}, bytearray()
    for row in rows:
        line = canonical(row) + b"\n"
        if len(line) >= 8_000_000:
            raise ValueError("single dataset row exceeds the artifact size budget")
        if buffer and len(buffer) + len(line) >= 8_000_000:
            parts[f"data/{stem}_{len(parts):04d}.jsonl"] = bytes(buffer)
            buffer.clear()
        buffer.extend(line)
    if buffer:
        parts[f"data/{stem}_{len(parts):04d}.jsonl"] = bytes(buffer)
    return parts


def _verify_seal(run_path, seal):
    run_path = Path(run_path)
    if seal.get("schema_version") != 1:
        raise RuntimeError("unsupported dataset seal schema")
    names = seal.get("public_files", []) + seal.get("gold_files", [])
    if len(names) != len(set(names)) or set(names) != set(seal.get("files", {})):
        raise RuntimeError("dataset seal file list is inconsistent")
    for name, info in seal["files"].items():
        # Manifest references are constrained to flat dataset files in this run.
        path = Path(name)
        if path.parts[:1] != ("data",) or len(path.parts) != 2 or path.suffix != ".jsonl":
            raise RuntimeError("invalid dataset path in seal")
        file = run_path / path
        if (not file.is_file() or file.stat().st_size != info["bytes"]
                or file.stat().st_size >= LIMIT or sha(file) != info["sha256"]):
            raise RuntimeError(f"sealed dataset integrity failure: {name}")
    return seal


def prepare_dataset(name_or_public_path, run_path, config, gold_path=None) -> dict:
    """Validate and seal inputs, or verify a byte-for-byte compatible resume.

    Sealed scientific datasets are never overwritten. Incoming source bytes,
    canonical rows, provenance, and existing output hashes must all agree.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    run_path = Path(run_path)
    public, gold, provenance = _incoming(name_or_public_path, config, gold_path)
    public, gold = _validate_rows(public, gold)
    source_row_count = len(public)
    if provenance["kind"] == "existing_discovery":
        # Commit all validated source rows as well as their original file hashes,
        # including metadata belonging to aliases that do not become run units.
        source_rows_sha256 = digest({"public": public, "gold": gold})
        public, gold, deduplication = _deduplicate_existing(public, gold)
        provenance = {**provenance, "source_rows_sha256": source_rows_sha256,
                      "deduplication": deduplication}
    else:
        _reject_duplicate_questions(public)
    public_parts, gold_parts = _parts("public", public), _parts("protected_gold", gold)
    parts = {**public_parts, **gold_parts}
    incoming_digest = digest({"public": public, "gold": gold, "provenance": provenance})
    manifest = {
        "schema_version": 1, "uid_count": len(public),
        "source_row_count": source_row_count, "distinct_question_count": len(public),
        "public_files": list(public_parts), "gold_files": list(gold_parts),
        "files": {name: {"bytes": len(content), "sha256": digest(content)}
                  for name, content in parts.items()},
        "provenance": provenance, "incoming_digest": incoming_digest,
        "uid_sha256": digest([row["uid"] for row in public]),
        "families": sorted({row["family"] for row in public}),
    }
    seal_path = run_path / "data" / "DATASET_SEAL.json"
    if seal_path.exists():
        existing = load(seal_path)
        if canonical(existing) != canonical(manifest):
            raise RuntimeError("incoming dataset or provenance differs from the immutable seal")
        return _verify_seal(run_path, existing)
    # Immutable writes also make interrupted, unsealed preparation resumable.
    for name, content in parts.items():
        atomic(run_path / name, content, immutable=True)
    atomic(seal_path, manifest, immutable=True)
    return _verify_seal(run_path, manifest)


def load_public(run_path):
    run_path = Path(run_path)
    seal = _verify_seal(run_path, load(run_path / "data" / "DATASET_SEAL.json"))
    rows = [row for name in seal["public_files"] for row in load_jsonl(run_path / name)]
    if len(rows) != seal["uid_count"] or digest([row["uid"] for row in rows]) != seal["uid_sha256"]:
        raise RuntimeError("sealed public UID count or order differs")
    return rows


def load_gold(run_path):
    run_path = Path(run_path)
    seal = _verify_seal(run_path, load(run_path / "data" / "DATASET_SEAL.json"))
    rows = [row for name in seal["gold_files"] for row in load_jsonl(run_path / name)]
    result = {row["uid"]: row["gold"] for row in rows}
    if len(rows) != seal["uid_count"] or len(result) != len(rows):
        raise RuntimeError("sealed gold UID count or uniqueness differs")
    return result
