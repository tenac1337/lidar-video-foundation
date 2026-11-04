import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    "Summarize saved geometry diagnostics and reproduce the original depth rejections"
)
parser.add_argument("profile", type=Path)
b = parser.parse_args().profile
d = json.loads((b / "profile.json").read_text())
c = json.loads((b / "depth_controls.json").read_text())
control = {r["square"]: r for r in c["objects"]}
rows = d["objects"]
summary = []
for r in rows:
    x = control[r["square"]]
    e = r["experiments"]["rigid"]
    summary.append(
        dict(
            square=r["square"],
            native_samples=r["native_depth_samples"],
            own_mask_median_distance_mm=r["metrics"]["own_mask"]["median_mm"],
            fused_median_distance_mm=r["metrics"]["fused_square"]["median_mm"],
            native_core_median_height_mm=x["native_core"]["median_height_mm"],
            core_fraction_at_or_below_15mm=x["native_core"]["fraction_at_or_below_15mm"],
            model_base_height_mm=r["mesh_bounds_mm"][0][2],
            heldout_before_mm=e.get("heldout_before", {}).get("median_mm"),
            heldout_after_rigid_fit_mm=e.get("heldout_after", {}).get("median_mm"),
        )
    )
with (b / "per_object.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    w.writeheader()
    w.writerows(summary)
fig, (ax, heat) = plt.subplots(2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1, 1.8]})
xx = np.arange(len(rows))
ax.bar(
    xx - 0.18,
    [r["own_mask_median_distance_mm"] for r in summary],
    0.36,
    color="#256c91",
    label="Reference: own SAM mask, valid depth",
)
ax.bar(
    xx + 0.18,
    [r["fused_median_distance_mm"] for r in summary],
    0.36,
    color="#b95d32",
    label="Ten views: square, height >15 mm",
)
ax.axhline(49.032544975 / 4, color="#555555", ls="--", lw=1, label="Original depth limit (12.26 mm)")
ax.set_xticks(xx, [r["square"] for r in rows])
ax.set_xlabel("Provisional square label")
ax.set_ylabel("Median point-to-mesh distance (mm)")
ax.set_title("Same generated mesh, different observed depth sets")
ax.legend(fontsize=9)
frames = [r["frame"] for r in rows[0]["per_frame"]]
values = np.array(
    [[p["residual"]["median_mm"] if p["residual"] else np.nan for p in r["per_frame"]] for r in rows]
)
cm = plt.get_cmap("magma").copy()
cm.set_bad("#dddddd")
im = heat.imshow(values, cmap=cm, vmin=0, vmax=60, aspect="auto")
heat.set_yticks(range(len(rows)), [r["square"] for r in rows])
heat.set_xticks(range(len(frames)), [str(f) + (" (ref)" if f == 178 else "") for f in frames])
heat.set_xlabel("Imported frame index (197-frame capture)")
heat.set_ylabel("Provisional square label")
heat.set_title("Which views disagree? Same square and >15 mm height filter for every column")
for y in range(len(rows)):
    for x in range(len(frames)):
        v = values[y, x]
        heat.text(
            x,
            y,
            "—" if np.isnan(v) else f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black" if np.isnan(v) or v > 40 else "white",
        )
fig.colorbar(
    im, ax=heat, label="Median point-to-mesh distance (mm); gray = no selected depth", fraction=0.025
)
fig.suptitle("iPhone chess geometry profile — all 14 saved A100 meshes", fontsize=15)
fig.text(
    0.03,
    0.015,
    "Source: saved recording 2898c56d14, frame 5217 inference and ten fused views. Distances use 120,000 mesh surface samples.\nTop series differ in masks, height selection, resolution and views; their difference is not a pure multi-view effect.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig(b / "error_breakdown.png", dpi=150)
plt.close(fig)
# Independent check against the original reject strings, and consistency of the controls.
orig = json.loads((b.parent / "hybrid/verification.json").read_text())["rejected_learned_meshes"]
checks = []
for r in rows:
    reason = orig[r["square"]]
    if "measured-depth" in reason:
        expected = float(re.search(r": ([0-9.]+)m", reason).group(1)) * 1000
        delta = abs(r["original_10k_fused"]["median_mm"] - expected)
        assert delta <= 0.051, (r["square"], delta)
        checks.append(
            dict(
                square=r["square"],
                rounded_report_mm=expected,
                reproduced_mm=r["original_10k_fused"]["median_mm"],
            )
        )
assert len(checks) == 8
assert all(r["max_depth_difference_m"] == 0 and r["confidence_equal"] for r in c["original_sensor_checks"])
assert len(c["original_sensor_checks"]) == 10
assert (
    d["contracts"]["pose_max_abs_difference"]
    == d["contracts"]["K_scale_max_abs_difference"]
    == d["contracts"]["input_depth_max_abs_difference_m"]
    == 0
)
assert set(d["train_frames"]).isdisjoint(d["heldout_frames"])
assert 178 not in d["train_frames"] + d["heldout_frames"]
fits = [r["experiments"]["rigid"] for r in rows if "heldout_after" in r["experiments"]["rigid"]]
validation = dict(
    original_depth_rejections_reproduced=checks,
    original_png_frames_match=10,
    transfer_pose_intrinsics_depth_exact=True,
    training_and_heldout_disjoint=True,
    eligible_rigid_fits=len(fits),
    heldout_improved=sum(e["heldout_after"]["median_mm"] < e["heldout_before"]["median_mm"] for e in fits),
    insufficient_fit_data=[r["square"] for r in rows if "status" in r["experiments"]["rigid"]],
    median_across_objects_own_mm=float(np.median([r["own_mask_median_distance_mm"] for r in summary])),
    median_across_objects_fused_mm=float(np.median([r["fused_median_distance_mm"] for r in summary])),
    max_median_change_10k_to_120k_mm=max(
        abs(r["original_10k_fused"]["median_mm"] - r["metrics"]["fused_square"]["median_mm"]) for r in rows
    ),
    max_exact_check_p90_sampling_error_120k_mm=max(r["sampling_error_120k"]["p90_mm"] for r in rows),
    original_results_modified=False,
)
(b / "diagnostic_checks.json").write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2))
for r in c["objects"]:
    if r["square"] in ["c7", "d2", "h8"]:
        print(r["square"], "shift max median", max(x["median_height_mm"] for x in r["offset_sensitivity"]))
