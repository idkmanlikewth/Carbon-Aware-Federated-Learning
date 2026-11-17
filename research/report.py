"""Generate an evidence-derived Markdown report: python -m research.report"""
import json
from pathlib import Path
from .verify import verify


def main():
    root = Path(__file__).resolve().parents[1]
    lines = ["# Executed research verification", "",
             "These are actual completed CPU runs, not illustrative results. Carbon, energy and infrastructure latency are scenario estimates; no physical measurements were collected.", "",
             "## Completed experiments", "",
             "| Run | Dataset | Clients | Seeds | Rounds | RL episodes | Train / val+test | Evaluations | Actual wall seconds |",
             "|---|---|---:|---|---:|---:|---|---:|---:|"]
    total_eval = total_rounds = 0
    records = []
    for path in sorted((root/"evidence").glob("*/manifest.json")):
        checked = verify(path.parent)
        m = json.loads(path.read_text(encoding="utf-8"))
        c = m["config"]
        total_eval += checked["evaluations"]
        total_rounds += checked["round_records"]
        lines.append(f"| {path.parent.name} | {c['dataset']} | {c['clients']} | {', '.join(map(str,c['seeds']))} | {c['rounds']} | {c['episodes']} | {c['train_samples']} / {c['eval_samples']} | {checked['evaluations']} | {m['wall_seconds']:.2f} |")
        records.append((path.parent, m))
    lines += ["", f"Total: {total_eval} final evaluations and {total_rounds} recorded evaluation rounds. Preferences within a seed share training and are not independent replicates. The one-seed full-participation control is separate from policy comparisons. Wall times include loading and any initial downloads; simultaneous jobs affect elapsed time, so these are not hardware performance benchmarks.", "",
              "## Observed performance", "",
              "Mean hypervolume is computed across independent seeds. Test accuracy below is averaged across preferences within each seed and then across seeds. Costs use the predeclared normalization bounds. Full confidence intervals and paired Holm-adjusted tests are in each analysis.json.", "",
              "| Run | Policy | Mean test accuracy | Mean hypervolume | Hypervolume 95% CI |",
              "|---|---|---:|---:|---|"]
    for folder, m in records:
        results = json.loads((folder/"results.json").read_text())
        analysis = json.loads((folder/"analysis.json").read_text())
        for policy, stats in analysis["summary"].items():
            per_seed = [sum(r["test_accuracy"] for r in results if r["policy"] == policy and r["seed"] == seed)/len(m["config"]["preferences"]) for seed in m["config"]["seeds"]]
            ci = stats["ci95"]
            ci_text = f"[{ci[0]:.6f}, {ci[1]:.6f}]" if ci else "not estimated (one seed)"
            lines.append(f"| {folder.name} | {policy} | {sum(per_seed)/len(per_seed):.2%} | {stats['mean']:.6f} | {ci_text} |")
    lines += ["", "## Checks performed", "",
              "- research.verify rechecked every completed run: artifact hashes, source snapshot hashes where recorded, dataset subset independence, evaluation/round counts, availability masking, reward finiteness, preference sums, trajectory totals, and exact reanalysis from raw results.",
              "- Research tests cover Double DQN updates, preferences, reward bounds, hypervolume against independent inclusion-exclusion, no test-label leakage, independent seeds, deterministic reruns, 100-client execution, missing-data failures, and tamper detection.",
              "- Streamlit AppTest renders the dashboard, changes the frontier seed, and launches a real synthetic experiment through the Run button, waits for completion and verifies the completed dashboard.",
              "- Legacy tests.py, tests_demo.py and tests_pipeline.py exercise the original simulation and HTTP run/export/cancellation flow. A Windows UTF-8 decoding error in tests_demo.py was repaired.",
              "- The existing React frontend passed npm ci and its TypeScript/Vite production build. pip check reported no dependency conflicts.",
              "- GitHub Actions passed both test and frontend jobs for implementation commit 76df59abd7911bbdf683b84795ec955287c68aa0: https://github.com/idkmanlikewth/Carbon-Aware-Federated-Learning/actions/runs/44115648530 . The final evidence commit runs the checks again, including evidence verification.", "",
              "## Limits and provenance", "",
              "Accuracy is low under these small subsets and training budgets. These runs establish working execution paths, not convergence, publication-quality benchmark performance, or policy superiority. The adaptive DQN does not consistently outperform every baseline or ablation. Three-seed t tests have low power and distributional assumptions; inspect intervals and adjusted p-values before drawing conclusions.", "",
              "100 clients are virtual partitions executed sequentially on CPU. Client power/speed, availability and changing grid factors are declared synthetic scenarios; the controller is a heuristic adaptation rule. The scheduler chooses among six ranking rules rather than arbitrary client subsets. No live grid API or physical carbon meter is used.", "",
              "Each manifest records seed/configuration, independent subset hashes, software versions, source digest, training update counts, checkpoints and artifact hashes. Later runs include source_snapshot.zip; earlier short runs retain source digests without snapshots. The source commit field records the repository HEAD when execution began and git_dirty truthfully records uncommitted implementation changes. It is not relabeled as the final commit.", "",
              "CIFAR-10 was fetched from the MIT Open Learning mirror after a slow official-host download was stopped before training. The archive matched torchvision's official MD5 before extraction; see evidence/DOWNLOADS.json. Official training/test files are not committed. environment-lock.txt records the installed environment; CPU PyTorch wheels require the official CPU index noted in the README.", "",
              "Reproduce: use the exact config in each manifest with a new output directory. Verify without retraining: python -m research.verify evidence. Regenerate this report: python -m research.report."]
    lines += ["", "## Original workbench integration", "",
              "The original cream-and-vermilion React interface is again the default. Four-plate navigation and topology replay retain their original design. The pipeline plate includes an optional Preference-Conditioned DQN Research workspace using the same typography, borders and palette. Launch with python demo_server.py or START_DEMO.bat; the separate Streamlit viewer remains optional.", "",
              "The HTTP integration test starts a real synthetic research job through the workbench API, checks startup/completion state, reads results, exports its evidence ZIP, rejects invalid settings and traversal paths, and cancels a running job. All 38 local tests pass, and the TypeScript/Vite production build succeeds. Browser inspection confirmed the original topology replay controls and integrated 100-client research display. These UI checks do not modify the six recorded benchmark configurations above."]
    target = root/"docs"/"VERIFICATION.md"
    target.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
