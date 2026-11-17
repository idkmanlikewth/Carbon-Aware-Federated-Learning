# Research protocol

The unit of replication is an independent seed. Default evaluation preferences are accuracy-oriented (0.7,0.1,0.1,0.1), balanced (0.25,0.25,0.25,0.25) and carbon-oriented (0.1,0.1,0.7,0.1). They share a freshly trained policy within a seed. Early training episodes cycle these anchors; later episodes draw Dirichlet preferences. The DQN observes preferences directly. The unconditioned ablation zeroes that input.

Currently available clients are ranked by one of six selector rules; the agent selects the rule. Availability varies each round, with one declared always-online virtual client. Participation fraction is fixed per run. Intensity varies sinusoidally per client around scenario factors. This is synthetic infrastructure, not a grid API or meter reading.

All selected clients start from the round-global CNN. Parameters are averaged by sample count. The CNN uses two convolution layers, adaptive pooling and a ten-class head. Client computation is sequential on CPU; 100 clients means virtual partitions and profiles, not 100 physical devices or distributed workers. Streaming aggregation retains only global, local and accumulator models.

Modeled training duration is (2 + local samples * epochs * 0.03) / scenario speed * jitter seconds. Latency is the slowest selected duration plus 0.45 seconds. All clients incur idle power for the round; selected clients additionally incur active-minus-idle power during training. The coordinator consumes 30 W at a fixed 400 gCO2e/kWh scenario factor. Power (W) times seconds / 3600 gives Wh. Per-client Wh times gCO2e/kWh / 1000 gives grams. Manifest wall time is actual execution time, separate from modeled latency.

Training environments use seed+10000+episode; evaluation uses seed+100000. Training never uses final test accuracy. Seed+100 initializes every policy from identical weights for paired comparisons. Exogenous arrays are generated before selections so different policies cannot shift random timing or availability draws. Each independent seed regenerates all data and policies. Evaluation disables exploration and learning.

Validation and test are disjoint halves of a randomly sampled official test subset. Official training samples are partitioned among clients with Dirichlet label skew and deterministic rebalancing to avoid empty clients. Only validation is exposed to the scheduling loop. Test accuracy is evaluated at the final global model. No automatic hyperparameter selection uses final test data.

For each policy and seed, compute a frontier across preferences and exact hypervolume for accuracy, negative normalized latency and negative normalized carbon. Fairness is included in learning and reported separately. Predeclared per-round scales are 120 seconds and 5 grams; reference is (0,-1,-1). Reward costs are clipped; hypervolume costs are not. Report points outside the reference, which contribute zero hypervolume.

Policy variants: adaptive DQN is the reference. fixed disables only adaptation; unconditioned zeroes only preference input; static_carbon makes only training intensity stationary; raw_reward disables only reward normalization. All learned ablations otherwise retain adaptation and train from scratch. Random, carbon-greedy and fairness selectors use the same participation fraction and evaluation traces.

Statistical validation uses per-seed hypervolume, Student-t 95% intervals, paired two-sided t tests, Holm correction within each run, and paired standardized differences. A single seed has no interval. Zero-variance nonzero differences have undefined t-test/effect values, serialized as null. Preference points are not independent statistical samples.

Included short runs verify execution and reproducibility; do not infer convergence or real-world carbon reductions. Confirmatory research should preregister seeds, preferences, sample sizes, budgets and normalization bounds; increase independent seeds and training budgets; inspect clipping and baseline outcomes. Small seed counts and test subsets limit inference. No assumption that the learned policy beats baselines is made.
