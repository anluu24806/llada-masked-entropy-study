"""Verify exact direct pins; pip check alone is insufficient."""
import importlib.metadata as metadata
import sys
from .common import PROJECT

def failures():
    problems=[]
    if sys.version_info[:3]!=(3,11,11):problems.append(f"Python must be 3.11.11, got {sys.version.split()[0]}")
    requirements=[x.strip() for x in (PROJECT/"requirements.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
    requirements.append("torch==2.5.1+cu124")
    for requirement in requirements:
        name,expected=requirement.split("==")
        try:actual=metadata.version(name)
        except metadata.PackageNotFoundError:actual="MISSING"
        if actual!=expected:problems.append(f"{name}: expected {expected}, found {actual}")
    return problems
def check():
    problems=failures()
    if problems:raise RuntimeError("Pinned environment mismatch:\n"+"\n".join(problems))
    print("EXACT_ENVIRONMENT_PINS_PASS",flush=True)
if __name__=="__main__":check()
