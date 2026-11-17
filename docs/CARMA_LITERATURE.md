# CARMA-FL prior art and novelty audit

Problem statement, unchanged: **A Multi-Objective Reinforcement Learning Model for Optimizing the Latency–Carbon Trade-off in Federated Computing.**

CARMA-FL is a candidate experimental method, not a proven novel contribution or an accepted CCGrid paper. This targeted review is not an exhaustive systematic review. No numerical result from another paper is reused as a result of this repository.

## Closest work

| Work and primary source | Relevant overlap | Consequence for CARMA-FL |
|---|---|---|
| [FedCAMO, Computer Networks, DOI 10.1016/j.comnet.2026.112486](https://www.sciencedirect.com/science/article/pii/S1389128626004986) | Carbon-budgeted combinatorial client selection, online dual pricing, utility/diversity and holistic compute/communication/idle/server costs. | Carbon budgets, Lagrange multipliers and holistic energy accounting are already prior art. They cannot independently support a novelty claim. |
| [CEFL: Carbon-Efficient Federated Learning](https://arxiv.org/abs/2310.17972) | Cost-aware utility selection and critical learning periods. | Moving learning emphasis earlier and carbon efficiency later is already a related idea. The convergence controller requires comparison, not a novelty assertion. |
| [Green Federated Learning via Carbon-Aware Client and Time Slot Scheduling, author code](https://github.com/chrdz/GreenFL) | Time-varying carbon, slack scheduling, allocation fairness and fine-tuning. | Dynamic carbon scheduling itself is established. CARMA-FL currently selects cohorts and does not reproduce GreenFL's time-slot/slack optimization. |
| [GreenFL: Carbon-efficient Federated Learning over RE Powered Edge Computing Systems, author's publication list](https://guomingtang.github.io/publications/) | A separately named ICDCS 2025 GreenFL work on renewable-powered edge FL. | “GreenFL” is ambiguous. It must be identified by exact title/DOI; this work is not silently treated as identical to the MASCOTS implementation. |
| [FedMarl, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20894) | Learned client selection balancing model accuracy, latency and communication. | Learned per-client selection and efficiency trade-offs are already prior art. CARMA-FL is a shared centralized sequential Q scorer, not a faithful MARL-agent reproduction. |
| [MODRL-FL, publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/S1570870526002040) | Stability-aware multi-objective deep RL for dynamic edge FL and resource selection. | Robustness/stability and multi-objective RL are already related contributions. Full method-level comparison is still needed; publisher abstract access is not a faithful reproduction. |
| [Oort, author paper](https://www.usenix.org/system/files/osdi21-lai.pdf) and [official code](https://github.com/SymbioticLab/Oort) | Statistical utility, device speed and exploration in participant selection. | The implemented oort_style baseline retains those concepts but omits Oort's complete pacer, clipping, blacklisting and parameterization. It must remain labeled inspired/simplified. |

The implementation includes conventional FedAvg, random, round-robin, fastest, carbon-greedy and weighted controls, standard non-Double unconditioned DQN, preference-conditioned Double DQN, Oort-style and CARMA-FL. None of the results is labeled as a faithful FedCAMO, GreenFL, CEFL, FedMarl or MODRL-FL reproduction. Reproducing published tables would require matching their data, infrastructure, budgets and protocol; changing those details invalidates a direct numerical comparison.

## Mechanisms are grounded in existing methods

- [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) motivates shared encoders and pooled context. CARMA-FL emits permutation-equivariant per-client Q values; parameter count is independent of client count. The architecture is not itself novel.
- [Risk-constrained RL with CVaR](https://mohammadghavamzadeh.github.io/PUBLICATIONS/jmlr-VaR-CVaR.pdf) establishes CVaR-based risk constraints as prior art. CARMA-FL uses empirical marginal-CVaR penalties; it does not implement that paper's algorithm or inherit its theoretical guarantees.
- [Conformal prediction guide](https://arxiv.org/abs/2107.07511) motivates held-out residual calibration. CARMA-FL initially fits ratios on 20 historical observations and calibrates on a disjoint 40, then updates prequentially. Exchangeability and fixed-predictor conditions do not generally hold after adaptive selection or under distribution shift, so reported empirical coverage is essential. No distribution-free online coverage guarantee is claimed.

## Falsifiable contribution hypothesis

The question is whether causal uncertainty features and risk-aware, convergence-conditioned cohort decisions preserve a validation-accuracy constraint while improving latency/carbon efficiency and tail behavior on unseen infrastructure regimes, compared with matched controls. This is a hypothesis, not a finding. Combining known techniques does not establish novelty.

Evidence required: independent-seed feasible-cost comparisons; consistent accuracy/hit-rate outcomes; isolated ablations; calibrated coverage and tail audits under held-out shifts; policy-training overhead; reference/budget sensitivity; and a convincing comparison to the closest carbon-budgeted and critical-period methods. Negative or null outcomes must remain in the report. If CARMA-FL does not outperform simpler controls after correction, describe it as a reproducible experimental framework and candidate method rather than asserting a new state of the art.

## Historical carbon trace

The [Great Britain Carbon Intensity API](https://carbon-intensity.github.io/api-definitions/) supplies public time-stamped national intensity estimates. The repository preserves the raw response and extracted actual-estimate CSV, uses separate training/evaluation windows, and labels the experiment a UK trace replay on virtual profiles. These values are neither local wall-meter readings nor evidence of Indian-device carbon savings.
