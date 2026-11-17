"""Mixed-direction Pareto analysis and exact 3D hypervolume, independent-seed inference."""
import numpy as np
from scipy import stats


def pareto_mask(points):
    """All objectives maximized; identical points are both nondominated."""
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite objective matrix")
    return np.array([not np.any(np.all(a >= p, axis=1) & np.any(a > p, axis=1)) for p in a])


def hypervolume(points, reference=(0., -1., -1.)):
    """Exact union of axis-aligned boxes in maximization coordinates.

    Points outside the fixed reference are excluded, never clipped into the box.
    Accuracy in [0,1], negative normalized latency/carbon. Reference is declared
    before runs; cost ratios are unbounded to expose bound violations.
    """
    a, ref = np.asarray(points, float), np.asarray(reference, float)
    if a.size == 0:
        return 0.
    if a.ndim != 2 or a.shape[1] != len(ref) or not np.isfinite(a).all() or not np.isfinite(ref).all():
        raise ValueError("Invalid hypervolume dimensions or values")
    a = a[np.all(a > ref, axis=1)] - ref
    def union(p):
        if len(p) == 0:
            return 0.
        if p.shape[1] == 1:
            return float(p.max())
        edges = np.unique(np.r_[0., p[:, 0]])
        return sum((right-left)*union(p[p[:, 0] >= right, 1:]) for left, right in zip(edges[:-1], edges[1:]))
    return float(union(a))


def interval(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    if len(a) < 2:
        return {"n": len(a), "mean": mean, "ci95": None}
    margin = float(stats.t.ppf(.975, len(a)-1) * stats.sem(a))
    return {"n": len(a), "mean": mean, "ci95": [mean-margin, mean+margin]}


def paired_comparisons(hv_rows):
    """Two-sided paired t tests on per-seed hypervolume; Holm family correction."""
    table = {}
    for row in hv_rows:
        table.setdefault(row["policy"], {})[row["seed"]] = row["hypervolume"]
    out = []
    for policy in sorted(set(table)-{"adaptive"}):
        seeds = sorted(set(table.get("adaptive", {})) & set(table[policy]))
        if len(seeds) < 2:
            continue
        diff = np.array([table["adaptive"][s]-table[policy][s] for s in seeds])
        if np.std(diff) == 0:
            p = 1. if np.all(diff == 0) else None
        else:
            p = float(stats.ttest_1samp(diff, 0).pvalue)
        out.append({"baseline": policy, "seeds": seeds, "difference": interval(diff), "p_raw": p,
                    "p_holm": None, "effect_dz": float(diff.mean()/diff.std(ddof=1)) if diff.std() > 0 else None})
    valid = sorted([r for r in out if r["p_raw"] is not None], key=lambda r:r["p_raw"])
    running = 0.
    for i, row in enumerate(valid):
        running = max(running, min(1., (len(valid)-i)*row["p_raw"]))
        row["p_holm"] = running
    return out


def analyze(results, c):
    hv_rows = []
    frontier = []
    for seed in c.seeds:
        for policy in c.policies:
            rows = [r for r in results if r["seed"] == seed and r["policy"] == policy]
            points = [[r["test_accuracy"], -r["latency_s"]/(c.rounds*c.latency_scale_s), -r["carbon_g"]/(c.rounds*c.carbon_scale_g)] for r in rows]
            mask = pareto_mask(points)
            frontier.extend({"seed": seed, "policy": policy, "preference_index": r["preference_index"]} for r, keep in zip(rows, mask) if keep)
            hv_rows.append({"seed": seed, "policy": policy, "hypervolume": hypervolume(points),
                            "outside_reference": sum(p[1] <= -1 or p[2] <= -1 for p in points)})
    summaries = {p: interval([r["hypervolume"] for r in hv_rows if r["policy"] == p]) for p in c.policies}
    return {"objectives": ["test_accuracy:max", "latency/(rounds*latency_scale):min", "carbon/(rounds*carbon_scale):min"],
            "reference": [0, -1, -1], "pareto": frontier, "hypervolume": hv_rows, "summary": summaries,
            "paired_tests": paired_comparisons(hv_rows), "inference": "Exploratory paired seed-level t tests, 95% Student-t intervals, Holm correction; small seed counts are low power. Preferences are not independent replicates."}
