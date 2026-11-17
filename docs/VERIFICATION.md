# Executed research verification

These are actual completed CPU runs, not illustrative results. Carbon, energy and infrastructure latency are scenario estimates; no physical measurements were collected.

## Completed experiments

| Run | Dataset | Clients | Seeds | Rounds | RL episodes | Train / val+test | Evaluations | Actual wall seconds |
|---|---|---:|---|---:|---:|---|---:|---:|
| cifar10 | cifar10 | 10 | 11, 22, 33 | 5 | 8 | 1000 / 200 | 72 | 30.07 |
| fashion-mnist | fashion-mnist | 10 | 11, 22, 33 | 5 | 8 | 1000 / 200 | 72 | 46.91 |
| mnist | mnist | 10 | 11, 22, 33 | 5 | 8 | 1000 / 200 | 72 | 41.14 |
| mnist-100 | mnist | 100 | 11, 22, 33 | 5 | 8 | 1000 / 200 | 72 | 39.16 |
| mnist-extended | mnist | 10 | 11, 22, 33 | 20 | 20 | 2000 / 600 | 72 | 533.88 |
| mnist-learning-control | mnist | 10 | 44 | 30 | 1 | 2000 / 600 | 3 | 49.66 |

Total: 363 final evaluations and 2970 recorded evaluation rounds. Preferences within a seed share training and are not independent replicates. The one-seed full-participation control is separate from policy comparisons. Wall times include loading and any initial downloads; simultaneous jobs affect elapsed time, so these are not hardware performance benchmarks.

## Observed performance

Mean hypervolume is computed across independent seeds. Test accuracy below is averaged across preferences within each seed and then across seeds. Costs use the predeclared normalization bounds. Full confidence intervals and paired Holm-adjusted tests are in each analysis.json.

| Run | Policy | Mean test accuracy | Mean hypervolume | Hypervolume 95% CI |
|---|---|---:|---:|---|
| cifar10 | adaptive | 11.33% | 0.109026 | [-0.003756, 0.221808] |
| cifar10 | fixed | 11.33% | 0.108936 | [-0.004226, 0.222098] |
| cifar10 | unconditioned | 11.33% | 0.108762 | [-0.005131, 0.222656] |
| cifar10 | static_carbon | 11.33% | 0.109035 | [-0.003706, 0.221777] |
| cifar10 | raw_reward | 10.89% | 0.107721 | [-0.005165, 0.220608] |
| cifar10 | random | 9.00% | 0.083224 | [0.020109, 0.146340] |
| cifar10 | carbon | 11.33% | 0.108576 | [-0.003474, 0.220626] |
| cifar10 | fair | 11.33% | 0.105597 | [-0.004262, 0.215456] |
| fashion-mnist | adaptive | 7.67% | 0.071147 | [0.058368, 0.083926] |
| fashion-mnist | fixed | 12.56% | 0.137124 | [0.096786, 0.177462] |
| fashion-mnist | unconditioned | 11.33% | 0.107068 | [0.030263, 0.183872] |
| fashion-mnist | static_carbon | 11.11% | 0.117113 | [0.025434, 0.208791] |
| fashion-mnist | raw_reward | 11.33% | 0.121549 | [0.017220, 0.225878] |
| fashion-mnist | random | 13.00% | 0.119723 | [0.020280, 0.219167] |
| fashion-mnist | carbon | 10.00% | 0.095650 | [0.013917, 0.177382] |
| fashion-mnist | fair | 12.67% | 0.118703 | [0.018121, 0.219285] |
| mnist | adaptive | 11.78% | 0.123351 | [0.057315, 0.189388] |
| mnist | fixed | 11.22% | 0.109811 | [0.099326, 0.120296] |
| mnist | unconditioned | 11.67% | 0.108342 | [0.100961, 0.115722] |
| mnist | static_carbon | 12.00% | 0.110076 | [0.098602, 0.121551] |
| mnist | raw_reward | 11.89% | 0.114054 | [0.087049, 0.141059] |
| mnist | random | 9.00% | 0.082385 | [-0.034522, 0.199291] |
| mnist | carbon | 11.00% | 0.105009 | [0.079509, 0.130509] |
| mnist | fair | 10.67% | 0.099177 | [0.077976, 0.120379] |
| mnist-100 | adaptive | 11.33% | 0.103143 | [0.003873, 0.202413] |
| mnist-100 | fixed | 11.33% | 0.103143 | [0.003873, 0.202413] |
| mnist-100 | unconditioned | 11.33% | 0.103143 | [0.003873, 0.202413] |
| mnist-100 | static_carbon | 11.33% | 0.103143 | [0.003873, 0.202413] |
| mnist-100 | raw_reward | 11.22% | 0.104166 | [0.007754, 0.200577] |
| mnist-100 | random | 9.00% | 0.079308 | [0.040069, 0.118546] |
| mnist-100 | carbon | 11.00% | 0.102127 | [0.002163, 0.202091] |
| mnist-100 | fair | 11.00% | 0.097216 | [0.039822, 0.154611] |
| mnist-extended | adaptive | 10.70% | 0.116195 | [0.022486, 0.209903] |
| mnist-extended | fixed | 11.22% | 0.127461 | [0.014107, 0.240814] |
| mnist-extended | unconditioned | 11.56% | 0.105420 | [0.068160, 0.142679] |
| mnist-extended | static_carbon | 10.26% | 0.102165 | [0.085248, 0.119082] |
| mnist-extended | raw_reward | 10.44% | 0.098639 | [0.082689, 0.114589] |
| mnist-extended | random | 9.33% | 0.082795 | [0.064412, 0.101178] |
| mnist-extended | carbon | 11.33% | 0.105281 | [0.079286, 0.131277] |
| mnist-extended | fair | 9.33% | 0.083588 | [0.065494, 0.101683] |
| mnist-learning-control | random | 20.33% | 0.115888 | not estimated (one seed) |

## Checks performed

- research.verify rechecked every completed run: artifact hashes, source snapshot hashes where recorded, dataset subset independence, evaluation/round counts, availability masking, reward finiteness, preference sums, trajectory totals, and exact reanalysis from raw results.
- Research tests cover Double DQN updates, preferences, reward bounds, hypervolume against independent inclusion-exclusion, no test-label leakage, independent seeds, deterministic reruns, 100-client execution, missing-data failures, and tamper detection.
- Streamlit AppTest renders the dashboard, changes the frontier seed, and launches a real synthetic experiment through the Run button, waits for completion and verifies the completed dashboard.
- Legacy tests.py, tests_demo.py and tests_pipeline.py exercise the original simulation and HTTP run/export/cancellation flow. A Windows UTF-8 decoding error in tests_demo.py was repaired.
- The existing React frontend passed npm ci and its TypeScript/Vite production build. pip check reported no dependency conflicts.
- GitHub Actions passed both test and frontend jobs for implementation commit 76df59abd7911bbdf683b84795ec955287c68aa0: https://github.com/idkmanlikewth/Carbon-Aware-Federated-Learning/actions/runs/44115648530 . The final evidence commit runs the checks again, including evidence verification.

## Limits and provenance

Accuracy is low under these small subsets and training budgets. These runs establish working execution paths, not convergence, publication-quality benchmark performance, or policy superiority. The adaptive DQN does not consistently outperform every baseline or ablation. Three-seed t tests have low power and distributional assumptions; inspect intervals and adjusted p-values before drawing conclusions.

100 clients are virtual partitions executed sequentially on CPU. Client power/speed, availability and changing grid factors are declared synthetic scenarios; the controller is a heuristic adaptation rule. The scheduler chooses among six ranking rules rather than arbitrary client subsets. No live grid API or physical carbon meter is used.

Each manifest records seed/configuration, independent subset hashes, software versions, source digest, training update counts, checkpoints and artifact hashes. Later runs include source_snapshot.zip; earlier short runs retain source digests without snapshots. The source commit field records the repository HEAD when execution began and git_dirty truthfully records uncommitted implementation changes. It is not relabeled as the final commit.

CIFAR-10 was fetched from the MIT Open Learning mirror after a slow official-host download was stopped before training. The archive matched torchvision's official MD5 before extraction; see evidence/DOWNLOADS.json. Official training/test files are not committed. environment-lock.txt records the installed environment; CPU PyTorch wheels require the official CPU index noted in the README.

Reproduce: use the exact config in each manifest with a new output directory. Verify without retraining: python -m research.verify evidence. Regenerate this report: python -m research.report.

## Original workbench integration

The original cream-and-vermilion React interface is again the default. Four-plate navigation and topology replay retain their original design. The pipeline plate includes an optional Preference-Conditioned DQN Research workspace using the same typography, borders and palette. Launch with python demo_server.py or START_DEMO.bat; the separate Streamlit viewer remains optional.

The HTTP integration test starts a real synthetic research job through the workbench API, checks startup/completion state, reads results, exports its evidence ZIP, rejects invalid settings and traversal paths, and cancels a running job. All 38 local tests pass, and the TypeScript/Vite production build succeeds. Browser inspection confirmed the original topology replay controls and integrated 100-client research display. These UI checks do not modify the six recorded benchmark configurations above.
