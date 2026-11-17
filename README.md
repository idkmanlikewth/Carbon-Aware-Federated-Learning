# Carbon-Aware Federated Learning Research Lab

Preference-conditioned Double DQN, adaptive multi-objective rewards, dynamic virtual clients, independent-seed experiments and an interactive research dashboard.

Accuracy comes from trained classifiers. Carbon, energy and infrastructure latency are modeled scenario estimates, not physical measurements. The included short runs verify execution and reproducibility; they do not establish convergence or DQN superiority.

## Install and launch

Python 3.11 recommended. Windows: double-click START_RESEARCH.bat (also the default START_DEMO.bat / START_WINDOWS.bat). Linux/macOS: bash START_DEMO.sh.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-research.txt
python -m streamlit run research/dashboard.py --server.address 127.0.0.1
```

For CPU-only PyTorch, first install torch==2.5.1 torchvision==0.20.1 using the official CPU wheel index https://download.pytorch.org/whl/cpu. The engine runs on CPU with deterministic algorithms and one PyTorch thread.

Choose dataset, clients, training budget, independent seeds and ablations. Enable download for the first benchmark run. Completed evidence appears under Recorded run. Refresh runs while a job executes; exports include raw results, analysis, manifest and model checkpoints.

## Reproduce experiments

```bash
python -m research.run --dataset mnist --download --clients 10 --rounds 5 --episodes 8 --train-samples 1000 --eval-samples 200 --seeds 11 22 33 --output research_runs/mnist
python -m research.run --dataset fashion-mnist --download --output research_runs/fashion
python -m research.run --dataset cifar10 --download --output research_runs/cifar
python -m research.run --dataset mnist --download --clients 100 --output research_runs/scale100
```

Use a new output directory for every run. Data are cached under data/ and excluded from Git. Missing benchmark data cause an explicit error; no synthetic substitution occurs. The synthetic dataset option is an explicitly labeled smoke fixture.

Every independent seed regenerates subsets, Dirichlet non-IID partitions, classifier initialization, replay, policy training and infrastructure traces. Policies share evaluation data, model initialization and precomputed exogenous traces within a seed. Validation drives reward and adaptation; held-out test accuracy is evaluated only after the final round. Validation and final test are disjoint halves of a sampled official test subset; official training data supply clients.

## Implemented methods

- Preference-conditioned Double DQN: ten observation features plus four preference weights, a 64-by-64 MLP, replay, target network, Double DQN targets, terminal masking, Huber loss, gradient clipping and epsilon exploration.
- Scalable actions: six rules rank available clients by random score, duration, modeled carbon, participation, dataset size or a balanced score. The DQN chooses a rule, which selects a fixed fraction of clients. Network size is independent of client count. This is a restricted scheduling action space, not enumeration of every possible subset.
- Adaptive MORL: a declared feedback controller raises accuracy preference when validation accuracy falls behind a rising target, then renormalizes the simplex. Current and next preferences are stored for Bellman updates. The controller is heuristic, with no constraint guarantee.
- Normalized reward: accuracy gain / 0.1, negative latency / 120 seconds, negative carbon / 5 grams and negative participation Gini. Clipping bounds components; every round records clipping and preferences. Bounds are fixed before evaluation.
- Pareto/hypervolume: exact three-objective analysis maximizes final test accuracy and minimizes cumulative normalized latency and carbon. The fixed maximization reference is (0, -1, -1). Out-of-reference points are reported and excluded, without silently clipping costs. Fairness is reported and rewarded outside this three-objective hypervolume.
- Statistics: independent-seed hypervolume, means, 95% Student-t intervals, paired two-sided t tests against adaptive DQN, Holm correction and paired standardized effects. Preferences within a seed are not independent replicates. Three seeds are low power.

Eight policies run by default: adaptive DQN; fixed preference (no adaptation); unconditioned DQN; static-intensity training; raw rewards; random; carbon greedy; participation fairness. Each learned ablation changes one feature relative to adaptive DQN. All evaluate under the same dynamic conditions.

## Verification and evidence

```bash
python -m pytest tests/test_research.py tests/test_dashboard.py -q
python tests.py
python tests_demo.py
python tests_pipeline.py
cd frontend
npm ci
npm run build
```

CI checks research tests, dashboard rendering, the existing pipeline and the frontend build. Executed evidence under evidence/ includes manifests, dataset hashes, source digests, versions, artifact hashes, traces, results, analysis and checkpoints. The recorded pre-upgrade commit and dirty flag reflect when experiments actually ran; the source digest identifies the engine used.

See [research protocol](docs/RESEARCH_PROTOCOL.md) and [executed verification report](docs/VERIFICATION.md).

The prior NumPy demo and React console remain available through python demo_server.py or START_LEGACY.bat / START_LEGACY.sh. Their synthetic results and historical provenance belong to that legacy demo, and are not consumed by the research engine. See [legacy instructions](docs/LEGACY_DEMO.md).

Implementation references: [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning), [Fashion-MNIST loader](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html), [CIFAR-10 loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
