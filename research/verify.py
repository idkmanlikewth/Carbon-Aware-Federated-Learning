"""Verify recorded evidence without retraining: python -m research.verify evidence/*"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from .analysis import analyze
from .config import Config


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"{folder}: run is {manifest['status']}")
    for name, expected in manifest["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe artifact filename")
        actual = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    source = folder/"source_snapshot.zip"
    if source.exists():
        digest = hashlib.sha256()
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                digest.update(name.encode()); digest.update(archive.read(name))
        if digest.hexdigest() != manifest["engine_sha256"]:
            raise ValueError("Source snapshot checksum mismatch")
    cfg = dict(manifest["config"])
    cfg["seeds"], cfg["policies"] = tuple(cfg["seeds"]), tuple(cfg["policies"])
    cfg["preferences"] = tuple(tuple(p) for p in cfg["preferences"])
    c = Config(**cfg)
    results = json.loads((folder/"results.json").read_text())
    trace = json.loads((folder/"trajectories.json").read_text())
    count = len(c.seeds)*len(c.policies)*len(c.preferences)
    if len(results) != count or len(trace) != count*c.rounds:
        raise ValueError("Incomplete evaluation records")
    keys = {(r["seed"], r["policy"], r["preference_index"]) for r in results}
    if len(keys) != count:
        raise ValueError("Duplicate result records")
    for row in trace:
        if not row["selected"] or len(set(row["selected"])) != len(row["selected"]) or not set(row["selected"]) <= set(row["available"]):
            raise ValueError("Unavailable or duplicated client selection")
        if any(not 0 <= i < c.clients for i in row["available"]):
            raise ValueError("Invalid client ID")
        if not np.isclose(sum(row["preference"]), 1):
            raise ValueError("Invalid preference simplex")
        if not np.isfinite(row["reward_vector"]).all():
            raise ValueError("Nonfinite reward")
    if len({v["subset_sha256"] for v in manifest["datasets"].values()}) != len(c.seeds):
        raise ValueError("Seeds reuse dataset subsets")
    for row in results:
        if not 0 <= row["test_accuracy"] <= 1:
            raise ValueError("Invalid accuracy")
        records = [t for t in trace if all(t[k] == row[k] for k in ("seed", "policy", "preference_index"))]
        if sorted(t["round"] for t in records) != list(range(1, c.rounds+1)):
            raise ValueError("Missing or duplicate rounds")
        for key in ("carbon_g", "latency_s", "energy_wh"):
            if not np.isclose(sum(t[key] for t in records), row[key]):
                raise ValueError(f"Trajectory total differs: {key}")
    actual = analyze(results, c)
    stored = json.loads((folder/"analysis.json").read_text())
    def equivalent(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))
        return a == b
    if not equivalent(actual, stored):
        raise ValueError("Analysis does not reproduce from raw results")
    return {"run": folder.name, "status": "verified", "seeds": len(c.seeds), "evaluations": count, "round_records": len(trace)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+")
    args = parser.parse_args()
    folders = []
    for item in args.folders:
        path = Path(item)
        folders.extend([path] if (path/"manifest.json").exists() else [p.parent for p in path.glob("*/manifest.json")])
    if not folders:
        parser.error("No runs found")
    for folder in folders:
        print(json.dumps(verify(folder)))


if __name__ == "__main__":
    main()
