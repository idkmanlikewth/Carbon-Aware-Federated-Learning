# Carbon-Aware Federated Learning

**A Multi-Objective Reinforcement Learning Model for Optimizing the Latency–Carbon Trade-off in Federated Computing.**

CARMA-FL (Carbon-Aware Risk-Adaptive Multi-Objective Reinforcement Learning for Federated Computing) is a candidate research framework. Novelty and superiority are not assumed. The original four-plate cream-and-vermilion workbench is preserved.

## Run the original interface

Python 3.11 recommended. Install dependencies into a virtual environment, then launch:

```bash
python -m pip install -r requirements-research.txt
python demo_server.py
```

Windows: START_DEMO.bat / START_RESEARCH.bat. Linux/macOS: bash START_DEMO.sh. The UI opens at http://127.0.0.1:8765. Choose **8-Layer MORL Pipeline → Preference-Conditioned DQN Research** (or /research). CARMA-FL is the default research engine; the earlier weak-budget protocol is retained as an explicitly archived option. Topology Replay and the other original plates retain their design.

For NVIDIA CUDA 12.4, install the compatible wheels in your environment before the remaining dependencies:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-research.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The equivalent CPU wheel index is https://download.pytorch.org/whl/cpu. CARMA device auto chooses available CUDA, otherwise CPU. An explicit cuda request fails if unavailable. Launch the server using the CUDA environment, or set CARMA_PYTHON to its Python executable. environment-cuda-lock.txt records the tested GPU environment; environment-lock.txt records the earlier CPU environment. The optional separate Streamlit viewer remains at python -m streamlit run research/dashboard.py.

## Convergence before scheduling

The revised conventional FedAvg CPU development gates used a stronger classifier, train-only standardization, Adam, shuffled batches, larger subsets and more local work. Observed held-out test accuracies were MNIST 96.2%, Fashion-MNIST 86.0%, and CIFAR-10 66.0%. These are one-seed learning gates, not ten-seed scheduler comparisons or claims of convergence to a global optimum. A separate three-seed validation-only factorial diagnostic isolates architecture, optimizer and standardization at a common budget.

Training and validation are disjoint subsets of official training data. Official test data is used only for final evaluation. Missing benchmark data fails explicitly; synthetic is an explicitly named smoke fixture. The original chance-level evidence is preserved, not overwritten or presented as scientifically adequate.

## CARMA-FL mechanisms

- Shared permutation-equivariant per-client Q scorer with pooled context, masked sequential cohort selection and fixed parameter count for 4–100 virtual clients.
- Preference-conditioned Double DQN, replay, target network, terminal masking, Huber loss and independent policy retraining for each seed.
- Client-specific causal historical cost estimation, split calibration, online residual updates and empirically audited prediction intervals for duration, energy, carbon intensity and bandwidth; availability history is also estimated.
- Empirical CVaR risk features/reward, with a documented marginal cohort surrogate rather than an unsupported joint-risk guarantee.
- Convergence-aware preference/constraint urgency using validation accuracy, training loss, loss improvement, and remaining time/carbon budgets.
- Soft accuracy constraints with projected Lagrangian penalties. Latency and carbon remain the cost objectives. Violations are retained; feasibility is not guaranteed.
- Meter-anchored virtual profiles, replaceable measured-power inputs and a historical-observation CSV interface. The reported 0.065 kWh / 190 min reference is preserved with unverified instrument details. PC2–PC4 and timing ratios remain scenarios.
- Public historical UK carbon-intensity replay with a preserved raw API response and separate training/evaluation windows.

Infrastructure energy, latency and emissions are modeled, not physical readings. Classifier outcomes and wall overhead are observed. Virtual clients run on one host; descriptors do not imply physical devices. Coordinator-only overhead energy excludes GPU energy and is not a full measured carbon footprint.

## Locked campaign and resuming

```bash
python -m carma.campaign --output carma_runs/ccgrid2027 --device cuda --plan-only
python -m carma.campaign --output carma_runs/ccgrid2027 --device cuda --workers 3
python -m carma.status
```

The saved matrix has 24 configurations and 1,110 policy evaluations: ten independent seeds per configuration, all baseline/ablation comparisons on three datasets, non-IID strengths, static/dynamic carbon, heterogeneity, availability, uncertainty, unseen shifts, reward sensitivity, real carbon-trace replay, matched 4/10/20/50/100-client scaling, and longer policy-training diagnostics. This is coverage across factors, not a full factorial cross of every factor. **A plan is not a completed experiment.** The report and dashboard show actual completed counts and label interim records.

Rerun the same campaign command to resume. Atomic checkpoints preserve completed seed-policy pairs. Runtime source, configuration and input hashes must match; interrupted current policies restart from their independent seeds. Concurrent GPU studies have contention-dependent wall timings, explicitly recorded. Keep the machine and experiment processes running during a chat pause; no account reset is required for local computation itself.

Run a single saved JSON configuration in a new directory:

```bash
python -m carma.run --config carma_runs/ccgrid2027/main-mnist.config.json --output carma_runs/new-study
python -m carma.run --config carma_runs/ccgrid2027/main-mnist.config.json --output carma_runs/new-study --resume
```

## Comparisons and honest inference

Baselines: conventional FedAvg, random, round-robin, fastest, carbon-greedy, weighted heuristic, standard DQN, preference-conditioned Double DQN, Oort-style, and CARMA-FL. Six independently switched ablations remove calibrated prediction, CVaR risk, adaptation, accuracy constraints, preference conditioning, or learned selection. Oort-style is simplified; no published algorithm is falsely labeled a faithful reproduction.

Primary analysis is accuracy-feasible 2D latency/carbon hypervolume with fixed reference bounds. A final validation-infeasible point contributes zero. Target costs remain censored when never reached. Reports retain final test accuracy, target-hit/final-feasibility rates, total modeled costs, violations, tail CVaR, fairness, interval coverage, policy training and inference overhead. Independent-seed confidence intervals and paired permutation tests with per-study Holm correction are included. No carbon savings are claimed by accepting failed accuracy.

[Research protocol](docs/CARMA_PROTOCOL.md) · [Literature and novelty review](docs/CARMA_LITERATURE.md) · [Executed results and current campaign status](docs/CARMA_RESULTS.md)

## Verify evidence and application

```bash
python -m pytest tests/test_carma.py tests/test_research.py tests/test_dashboard.py tests/test_workbench.py -q
python tests.py
python tests_demo.py
python tests_pipeline.py
python -m research.verify evidence
python -m carma.verify carma_evidence
python -m carma.report
cd frontend
npm ci
npm run build
```

CARMA verification requires completed schema-2 studies, checks artifact/source hashes, seed independence, actual round counts, availability, target censoring, totals, coverage and reproducible analysis. Until the first locked study completes, carma_evidence contains only explicitly labeled convergence diagnostics and planned/status files. Completed studies are added by carma.report. Published evidence omits large classifier states while preserving their hashes; policy states and raw metrics are included. Full checkpoints remain in the original local runs and complete evidence bundle. Use --require-checkpoints on original runs.

The earlier research engine remains under research/, with archived evidence under evidence/ and its [earlier verification report](docs/VERIFICATION.md). Those results and protocol are separate from CARMA-FL. [Original demo instructions](docs/LEGACY_DEMO.md) remain available.
