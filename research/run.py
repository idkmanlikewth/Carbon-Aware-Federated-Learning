"""python -m research.run --download --dataset mnist --output research_runs/mnist"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile
import numpy as np
import torch
from .agent import Agent, adapt_preference
from .analysis import analyze
from .config import Config
from .data import load_data
from .environment import Environment

LEARNED = {"adaptive", "fixed", "unconditioned", "static_carbon", "raw_reward"}


def preference(base, env, adaptive):
    return adapt_preference(base, env.accuracy, env.round/env.c.rounds) if adaptive else np.asarray(base, dtype=np.float32)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def run(c, output):
    unknown = set(c.policies) - LEARNED - {"random", "carbon", "fair"}
    if unknown or len(set(c.policies)) != len(c.policies) or not c.policies:
        raise ValueError(f"Invalid policies: {unknown}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    manifest = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(c), "measurement_type": "SIMULATED_INFRASTRUCTURE; actual classifier evaluation",
                "carbon_model": "Scenario W, seconds and gCO2e/kWh; includes idle clients and 30W coordinator at 400g/kWh. Not measured hardware energy or wall-clock latency.",
                "python": platform.python_version(), "platform": platform.platform(),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "scipy")},
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
                "datasets": {}, "training": []}
    source_hash = hashlib.sha256()
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    source_paths.append(Path(__file__).resolve().parents[1]/"fl_pipeline"/"data.py")
    with zipfile.ZipFile(output/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for path in source_paths:
            name = str(path.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/")
            content = path.read_bytes()
            source_hash.update(name.encode()); source_hash.update(content)
            snapshot.writestr(name, content)
    manifest["engine_sha256"] = source_hash.hexdigest()
    write_json(output/"manifest.json", manifest)
    results, trajectories = [], []
    try:
        for seed in c.seeds:
            print(f"seed={seed}: loading {c.dataset}", flush=True)
            data = load_data(c, seed)
            manifest["datasets"][str(seed)] = data.provenance
            for policy in c.policies:
                adaptive = policy in LEARNED and policy != "fixed"
                agent = None
                if policy in LEARNED:
                    agent = Agent(c, seed+100, conditioned=policy != "unconditioned")
                    for episode in range(c.episodes):
                        env = Environment(c, data, seed+10000+episode, static_carbon=policy == "static_carbon", raw_reward=policy == "raw_reward")
                        base = c.preferences[episode % len(c.preferences)] if episode < len(c.preferences) else agent.rng.dirichlet(np.ones(4))
                        state = env.observation()
                        p = preference(base, env, adaptive)
                        for _ in range(c.rounds):
                            action = agent.act(state, p, epsilon=max(.05, .9*(1-episode/c.episodes)))
                            ns, reward, done, _ = env.step(action)
                            npref = preference(base, env, adaptive)
                            agent.learn(state, p, action, reward, ns, npref, done)
                            state, p = ns, npref
                    checkpoint = f"{seed}_{policy}.pt"
                    torch.save({"model": agent.online.state_dict(), "config": asdict(c), "seed": seed, "policy": policy}, output/checkpoint)
                    manifest["training"].append({"seed": seed, "policy": policy, "updates": agent.steps,
                                                  "loss_last": agent.losses[-1] if agent.losses else None, "checkpoint": checkpoint})
                for pi, base in enumerate(c.preferences):
                    # Every variant evaluated under dynamic intensity, including static-training ablation.
                    env = Environment(c, data, seed+100000)
                    for _ in range(c.rounds):
                        p = preference(base, env, adaptive)
                        action = agent.act(env.observation(), p) if agent else {"random": 0, "carbon": 2, "fair": 3}[policy]
                        _, _, _, info = env.step(action)
                        trajectories.append({"seed": seed, "policy": policy, "preference_index": pi, "preference": p.tolist(), **info})
                    results.append({"seed": seed, "policy": policy, "preference_index": pi, "base_preference": list(base),
                                    "test_accuracy": env.evaluate(data.test), "validation_accuracy": env.accuracy,
                                    "latency_s": env.total_latency, "carbon_g": env.total_carbon, "energy_wh": env.total_wh, "gini": info["gini"]})
                print(f"seed={seed} policy={policy} complete", flush=True)
                write_json(output/"results.json", results)
                write_json(output/"trajectories.json", trajectories)
                write_json(output/"manifest.json", manifest)
        analysis = analyze(results, c)
        write_json(output/"analysis.json", analysis)
        manifest.update(status="complete", wall_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        manifest["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.name != "manifest.json" and p.is_file()}
        write_json(output/"manifest.json", manifest)
        return results, analysis
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output/"manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion-mnist", "cifar10", "synthetic"], default="mnist")
    for name, default in [("clients", 10), ("rounds", 10), ("episodes", 10), ("train-samples", 2000), ("eval-samples", 500), ("local-epochs", 1)]:
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--policies", nargs="+", default=list(Config().policies))
    parser.add_argument("--participation", type=float, default=.2)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    args["seeds"], args["policies"] = tuple(args["seeds"]), tuple(args["policies"])
    run(Config(**args), output)


if __name__ == "__main__":
    main()
