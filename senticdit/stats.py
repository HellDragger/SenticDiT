"""Statistics.

`bootstrap_ci` is for single-arm summaries; comparisons go through `paired_bootstrap_diff`, which
puts the interval on the difference (two overlapping 95% intervals can still hide a significant
difference). `min_detectable_effect` reports what the design could have caught, which is what
turns "we saw nothing" into a claim.
"""
import numpy as np
from scipy import stats as _sps


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    """Single-arm summary. Fine for describing one configuration; NOT a comparison test."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_boot=10000, alpha=0.05, seed=0):
    """CI on the DIFFERENCE between two item-matched arms."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"n": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = float(np.mean(np.abs(boots - boots.mean()) >= abs(d.mean())))
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "p": p, "n": int(d.size),
            "significant": not (lo <= 0.0 <= hi)}


def describe_diff(res, name_a="A", name_b="B", unit=""):
    if res.get("n", 0) == 0:
        return "no paired observations"
    verdict = "a distinguishable" if res["significant"] else "NO distinguishable"
    return (f"{name_a} {res['mean_a']:.4g}{unit} vs {name_b} {res['mean_b']:.4g}{unit} | "
            f"diff {res['diff']:+.4g}{unit} (95% CI [{res['ci_low']:+.4g}, {res['ci_high']:+.4g}], "
            f"p={res['p']:.3f}, n={res['n']}) -> {verdict} effect")


def wilson_ci(successes, n, z=1.96):
    """Interval for a proportion. Use this for failure counts, not a bootstrapped mean of a
    variable that is censored at its ceiling."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(centre - half), float(centre + half)


def min_detectable_effect(sd, n_per_arm, alpha=0.05, power=0.80, paired=False):
    """Smallest true difference this design could have caught. Report beside every null."""
    if not np.isfinite(sd) or n_per_arm <= 0:
        return float("nan")
    z_a = _sps.norm.ppf(1 - alpha / 2)
    z_b = _sps.norm.ppf(power)
    factor = 1.0 if paired else np.sqrt(2.0)
    return float((z_a + z_b) * sd * factor / np.sqrt(n_per_arm))
