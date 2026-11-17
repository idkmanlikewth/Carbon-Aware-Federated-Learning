"""Launch: python -m streamlit run research/dashboard.py"""
import io
import json
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Carbon-Aware FL · Research", page_icon="🌿", layout="wide")
st.title("Carbon-Aware Federated Learning")
st.caption("Preference-conditioned Double DQN · independent seeds · auditable experiments")
st.warning("Carbon, energy and infrastructure latency are modeled scenario estimates. Accuracy comes from trained classifiers. No physical measurements are collected.")

with st.sidebar:
    st.header("Experiment configuration")
    dataset = st.selectbox("Dataset", ["mnist", "fashion-mnist", "cifar10", "synthetic"])
    clients = st.slider("Virtual clients", 2, 100, 10)
    rounds = st.number_input("Federated rounds", 1, 200, 10)
    episodes = st.number_input("DQN training episodes", 1, 500, 10)
    train_samples = st.number_input("Training subset", clients, 50000, max(2000, clients))
    eval_samples = st.number_input("Validation + test subset", 20, 10000, 500)
    seeds = st.text_input("Independent seeds (space separated)", "11 22 33")
    policies = st.multiselect("Policies / ablations", ["adaptive", "fixed", "unconditioned", "static_carbon", "raw_reward", "random", "carbon", "fair"], default=["adaptive", "fixed", "random", "carbon"])
    download = st.checkbox("Download official datasets if needed", False)
    st.caption("Preference order: accuracy, latency, carbon, fairness. Three declared preferences are evaluated per policy.")
    if "process" in st.session_state and st.session_state.process.poll() is None:
        st.info("Experiment running. Refresh to inspect progress.")
        if st.button("Stop experiment"):
            st.session_state.process.terminate()
            st.session_state.process.wait(timeout=10)
            target = Path(st.session_state.run_path)/"manifest.json"
            if target.exists():
                stopped = json.loads(target.read_text())
                if stopped.get("status") == "running":
                    stopped["status"] = "cancelled"
                    target.write_text(json.dumps(stopped, indent=2), encoding="utf-8")
            st.rerun()
    elif st.button("Run experiment", type="primary", disabled=not policies):
        try:
            from research.config import Config
            seed_list = tuple(int(s) for s in seeds.split())
            cfg = Config(dataset=dataset, clients=clients, rounds=int(rounds), episodes=int(episodes), train_samples=int(train_samples), eval_samples=int(eval_samples), seeds=seed_list, policies=tuple(policies))
            target = ROOT/"research_runs"/uuid.uuid4().hex[:12]
            target.parent.mkdir(exist_ok=True)
            log_path = target.parent/(target.name+".log")
            command = [sys.executable, "-m", "research.run", "--dataset", dataset, "--clients", str(clients), "--rounds", str(rounds), "--episodes", str(episodes), "--train-samples", str(train_samples), "--eval-samples", str(eval_samples), "--seeds", *map(str, seed_list), "--policies", *policies, "--output", str(target)]
            if download:
                command.append("--download")
            with log_path.open("w", encoding="utf-8") as log:
                st.session_state.process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            st.session_state.run_path = str(target)
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    st.button("Refresh runs")

paths = sorted(list((ROOT/"research_runs").glob("*/manifest.json")) + list((ROOT/"evidence").glob("*/manifest.json")))
if not paths:
    st.info("Run an experiment to populate the dashboard. No placeholder results are displayed.")
    st.stop()
selected = st.selectbox("Recorded run", paths, format_func=lambda p:str(p.parent.relative_to(ROOT)))
folder = selected.parent
manifest = json.loads(selected.read_text(encoding="utf-8"))
st.write(f"Status: **{manifest['status']}** · Dataset: **{manifest['config']['dataset']}** · Clients: **{manifest['config']['clients']}**")
log = folder.parent/(folder.name+".log")
if log.exists():
    with st.expander("Execution log", expanded=manifest["status"] != "complete"):
        st.code(log.read_text(encoding="utf-8")[-12000:])
if manifest["status"] != "complete":
    st.info("Only completed runs are analyzed. Refresh for status changes.")
    st.json(manifest)
    st.stop()
results = pd.DataFrame(json.loads((folder/"results.json").read_text()))
analysis = json.loads((folder/"analysis.json").read_text())
trajectories = pd.DataFrame(json.loads((folder/"trajectories.json").read_text()))
overview, tradeoff, replay, validation, provenance = st.tabs(["Overview", "Pareto analysis", "Round replay", "Statistics & ablations", "Provenance & export"])
with overview:
    a, b, c = st.columns(3)
    a.metric("Independent seeds", results.seed.nunique())
    b.metric("Evaluated policies", results.policy.nunique())
    c.metric("Maximum test accuracy", f"{results.test_accuracy.max():.1%}")
    st.dataframe(results, use_container_width=True)
    st.plotly_chart(px.box(results, x="policy", y="test_accuracy", points="all", color="policy"), use_container_width=True)
    st.caption("Points include preferences within each seed; statistical inference aggregates at the seed level.")
with tradeoff:
    st.caption("Frontiers maximize accuracy and minimize both normalized costs. Hypervolume uses the predeclared reference (0, −1, −1).")
    seed = st.selectbox("Seed for frontier", sorted(results.seed.unique()), key="pareto_seed")
    points = results[results.seed == seed].copy()
    frontier = {(r["policy"], r["preference_index"]) for r in analysis["pareto"] if r["seed"] == int(seed)}
    points["nondominated_within_policy"] = [(r.policy, r.preference_index) in frontier for r in points.itertuples()]
    st.plotly_chart(px.scatter_3d(points, x="carbon_g", y="latency_s", z="test_accuracy", color="policy", symbol="nondominated_within_policy", hover_data=["preference_index"]), use_container_width=True)
    st.dataframe(pd.DataFrame(analysis["hypervolume"]), use_container_width=True)
with replay:
    policy = st.selectbox("Policy", list(results.policy.unique()))
    seed = st.selectbox("Seed", sorted(results.seed.unique()))
    pref = st.selectbox("Preference index", sorted(results.preference_index.unique()))
    trace = trajectories[(trajectories.policy == policy) & (trajectories.seed == seed) & (trajectories.preference_index == pref)]
    metric = st.selectbox("Round metric", ["validation_accuracy", "carbon_g", "latency_s", "energy_wh", "gini"])
    st.plotly_chart(px.line(trace, x="round", y=metric, markers=True), use_container_width=True)
    round_idx = st.slider("Replay round", 1, int(manifest["config"]["rounds"]), 1) if manifest["config"]["rounds"] > 1 else 1
    row = trace[trace["round"] == round_idx].iloc[0]
    chart = pd.DataFrame({"client": range(len(row.intensity_g_kwh)), "intensity_g_kwh": row.intensity_g_kwh})
    chart["state"] = ["selected" if i in row.selected else "available" if i in row.available else "offline" for i in chart.client]
    st.plotly_chart(px.bar(chart, x="client", y="intensity_g_kwh", color="state"), use_container_width=True)
    st.write("Observed preference", row.preference)
with validation:
    st.info(analysis["inference"])
    st.dataframe(pd.DataFrame([{"policy": p, **s} for p, s in analysis["summary"].items()]), use_container_width=True)
    st.json(analysis["paired_tests"])
    st.caption("fixed: disables adaptation; unconditioned: zeroes preference input; static_carbon: stationary training intensity; raw_reward: disables normalization. Other learned ablations retain adaptation. All evaluation environments remain dynamic.")
with provenance:
    st.json(manifest)
    st.download_button("Download results CSV", results.to_csv(index=False), "results.csv", "text/csv")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in folder.iterdir():
            if file.is_file():
                archive.write(file, file.name)
    st.download_button("Download complete evidence ZIP", buffer.getvalue(), folder.name+".zip", "application/zip")
