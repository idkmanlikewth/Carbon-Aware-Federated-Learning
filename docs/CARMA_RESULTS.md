# CARMA-FL executed research report

**A Multi-Objective Reinforcement Learning Model for Optimizing the Latency–Carbon Trade-off in Federated Computing.**

CARMA-FL is a candidate method. Novelty, superiority and publication readiness are not assumed. Infrastructure is simulated; classifier outcomes and wall overhead are observed. The user-reported PC1 meter reference remains an unverified input, not a new measurement.

Campaign status: **running**. **0 / 24** studies are completed and checksum-verified. **114** final policy evaluations have actually been recorded, including interim studies. Statistical tables below include completed studies only; pending work is never represented as a result.

## Conventional FedAvg learning gates

These CPU development gates preceded the scheduler campaign. They are not ten-seed scheduler comparisons. Validation is drawn from official training data; test is held out until final evaluation. They predate the strict source-snapshot campaign protocol and are not retroactively relabeled as source-pinned confirmatory evidence.

| Dataset | Train images | Validation / test | Rounds × local epochs | Seed | Final validation | Final test |
|---|---:|---|---|---|---:|---:|
| mnist | 6000 | 1000 / 1000 | 10 × 2 | 11 | 95.60% | 96.20% |
| fashion-mnist | 6000 | 1000 / 1000 | 15 × 2 | 11 | 83.60% | 86.00% |
| cifar10 | 10000 | 1000 / 1000 | 30 × 2 | 11 | 67.40% | 66.00% |

## Controlled failure diagnosis

All cases use the same 6,000-image IID MNIST pool, disjoint validation set, local batches, ten rounds and two local epochs per seed. Six architecture/optimizer/standardization cases use seeds 11, 22 and 33. Test accuracy is not inspected. These are development diagnostics with three seeds, not a causal explanation for every difference from the original four-client protocol.

| Case | Mean final validation accuracy | Mean final validation loss |
|---|---:|---:|
| earlier_adam | 81.57% | 0.6114 |
| earlier_adam_raw | 73.70% | 0.8422 |
| earlier_sgd | 77.90% | 0.7051 |
| stronger_adam | 95.33% | 0.1415 |
| stronger_adam_raw | 93.33% | 0.2268 |
| stronger_sgd | 93.13% | 0.2219 |

The original chance-level results were not an adequate basis for scheduler claims. The controlled diagnostics separate architecture, optimizer and input-standardization effects at a common training budget. Data size, local epochs, participation and communication budgets also changed in the revised protocol; they are not falsely attributed to architecture alone.

## Completed independent-seed comparisons

Primary endpoint: two-dimensional normalized latency/carbon hypervolume, with final validation-accuracy feasibility required. Infeasible points contribute zero. Accuracy thresholds and cost reference bounds were saved before execution. Time/carbon to target are censored when never reached; conditional means must be interpreted with hit counts.

| Study | Policy | Seeds | Test accuracy mean | Final feasible | Target hit | Feasible HV mean [95% CI] | Policy training seconds mean |
|---|---|---:|---:|---:|---:|---|---:|

## Statistical claims and limits

Per-study analyses include Student-t confidence intervals and paired sign-flip permutation tests, Holm-corrected within the study baseline family. All seeds are independent model/policy retrainings. There is no campaign-wide multiplicity correction; exploratory scenario claims must be labeled accordingly. Raw test accuracy, every failed target, total modeled energy/latency/carbon, both constraint-violation counts, tail CVaR, fairness, per-round prediction-interval coverage and wall overhead remain in the raw JSON.

No unconditional carbon-saving claim is made from an accuracy-infeasible run. Nominal 90% marginal prediction intervals are empirically audited, not guaranteed under adaptive selection or unseen shifts. The risk reward uses a marginal-CVaR cohort surrogate, not exact joint straggler/idle CVaR. Measured policy training/inference wall time is contention-dependent because independent studies share one GPU. Coordinator-only overhead carbon uses an assumed 30 W; it excludes measured GPU energy and must not be called the full physical training footprint. Separate virtual training-rollout scenario costs are retained.

The Oort-style control is simplified. No published-method name is attached to a purported faithful reproduction. FedCAMO already contains dual carbon pricing and holistic accounting; CEFL already connects cost-aware utility to critical learning periods. Deep Sets, CVaR and constrained RL are established techniques. The candidate contribution needs method-level review and robust held-out evidence, not a novelty claim from combining them. See CARMA_LITERATURE.md and CARMA_PROTOCOL.md.

A confounded development ablation protocol was stopped and preserved outside the locked campaign. Those partial records are excluded from the tables. The locked no-constraint ablation removes only its penalty; the no-calibrated-prediction ablation retains the separate empirical CVaR mechanism.

## Reproduce and verify

Run `python -m carma.campaign --output carma_runs/ccgrid2027 --device cuda --workers 3`. Inspect first with `--plan-only`. Resume skips completed seed-policy pairs and rejects changed runtime/config/input hashes. The exact 24 configurations are in carma_evidence/plan.json. Run `python -m carma.verify carma_evidence` for published raw evidence; classifier checkpoint hashes are retained but those large states remain in original local runs/full bundle. Use `--require-checkpoints` on the originals. Policy checkpoints and source snapshots are included in published evidence. Historical UK carbon observations and raw API response are under carma_inputs; they are grid estimates, not physical measurements of the virtual clients.

Regenerate this report only from actual files: `python -m carma.report`. Completed-study plots are generated from the same verified data. Pending studies are not extrapolated.
