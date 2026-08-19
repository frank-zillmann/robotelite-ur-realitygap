"""Architecture diagram for the per-joint position model, for the report.

Shows the shared pipeline both PerJointPositionModel and PerJointTreeModel
use: eleven input features -> one regressor per joint -> predicted residual
-> added back to the commanded position -> predicted actual_q. The
regressor box is generic (linear least-squares or gradient-boosted trees)
since the surrounding architecture is identical either way -- only the
fitting method inside that one box differs between the two models.

    python plot_model_diagram.py

Writes model_architecture.png to the current directory.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

import ur_style

FEATURES = [
    ("target_current", "A", "commanded current"),
    ("qd", "rad/s", "commanded velocity"),
    ("qdd", "rad/s²", "commanded accel."),
    ("qdd_lag4", "rad/s²", "qdd, 4 samples (~31ms) ago"),
    ("qdd_lag16", "rad/s²", "qdd, 16 samples (~125ms) ago"),
    ("qdd_lag64", "rad/s²", "qdd, 64 samples (~500ms) ago"),
    ("pos", "rad", "= target_q (this joint)"),
    ("gravity_torque", "Nm", "physics feature, § below"),
    ("vel", "raw", "movej register"),
    ("acc", "raw", "movej register"),
    ("bias", "= 1", "per-joint intercept"),
]

# Feature names highlighted as "hand-engineered, not raw-logged" -- gravity is
# a physics feature computed from the full pose, the qdd_lag* taps are this
# model's only features with memory (causal, per-recording -- see
# train_distillation_model._lag_array). Distinct colors so a reader can see
# at a glance these two additions aren't just more of the raw signal.
_PHYSICS_FEATURE = "gravity_torque"
_LAG_FEATURES = {"qdd_lag4", "qdd_lag16", "qdd_lag64"}


def _box(ax, xy, w, h, text, *, face, edge, text_color=None, fontsize=10.5,
        fontweight="normal", boxstyle="round,pad=0.02,rounding_size=0.08"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                                facecolor=face, edgecolor=edge, linewidth=1.4,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color or ur_style.NAVY, zorder=4, linespacing=1.35)


def _arrow(ax, start, end, *, color, style="-", lw=1.6, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 connectionstyle=f"arc3,rad={rad}",
                                 linestyle=style, linewidth=lw, color=color,
                                 zorder=z, shrinkA=0, shrinkB=0))


def plot_model_diagram(out_path: str = "model_architecture.png"):
    ur_style.apply()
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.6, 8.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- input feature boxes ---------------------------------------------
    # Packed to fit 11 rows (was 8) into the same vertical band [0.5, 8.24]
    # the diagram used before the qdd_lag* taps were added, so the frame/
    # title positions below don't need to move.
    in_x, in_w = 0.5, 3.35
    box_top, box_bottom = 7.85, 0.5     # leaves headroom below the "inputs" title at y=8.15
    in_h = 0.5
    step = (box_top - in_h - box_bottom) / (len(FEATURES) - 1)
    ys = [box_top - in_h - i * step for i in range(len(FEATURES))]
    in_boxes = {}
    for (name, unit, note), y in zip(FEATURES, ys):
        is_gravity = name == _PHYSICS_FEATURE
        is_lag = name in _LAG_FEATURES
        is_pos = name == "pos"
        face = (ur_style.LIGHT_BLUE if is_gravity else
               ur_style.MID_BLUE if is_lag else ur_style.GRID)
        edge = ur_style.NAVY if (is_gravity or is_lag or is_pos) else ur_style.GRAY
        _box(ax, (in_x, y), in_w, in_h,
            f"{name}  ({unit})\n{note}", face=face, edge=edge, fontsize=8.7)
        in_boxes[name] = (in_x + in_w, y + in_h / 2)

    ax.text(in_x + in_w / 2, 8.15, "inputs — per row, per joint $j$",
            ha="center", fontsize=11.5, fontweight="bold", color=ur_style.NAVY)

    # ---- regressor box ------------------------------------------------------
    reg_x, reg_y, reg_w, reg_h = 5.9, 2.55, 3.1, 3.0
    _box(ax, (reg_x, reg_y), reg_w, reg_h,
        "Regressor$_j$\n\nlinear least-squares\nOR\ngradient-boosted trees",
        face=ur_style.BLUE, edge=ur_style.NAVY, text_color="white",
        fontsize=11, fontweight="bold")
    ax.text(reg_x + reg_w / 2, reg_y - 0.32, "predicts the residual\n"
            r"$\hat{\Delta}_j \approx$ actual_q$_j$ $-$ target_q$_j$",
            ha="center", va="top", fontsize=9.7, color=ur_style.NAVY, linespacing=1.3)

    for name in FEATURES[:-1]:
        _arrow(ax, in_boxes[name[0]], (reg_x - 0.05, reg_y + reg_h / 2),
              color=ur_style.GRAY, lw=1.2)
    # bias drawn last so its arrow is visible against the cluster
    _arrow(ax, in_boxes["bias"], (reg_x - 0.05, reg_y + reg_h / 2),
          color=ur_style.GRAY, lw=1.2)

    # ---- adder node -----------------------------------------------------
    plus_c = (10.35, reg_y + reg_h / 2)
    ax.add_patch(Circle(plus_c, 0.34, facecolor="white", edgecolor=ur_style.NAVY,
                        linewidth=1.6, zorder=3))
    ax.text(*plus_c, "+", ha="center", va="center", fontsize=18,
            fontweight="bold", color=ur_style.NAVY, zorder=4)
    _arrow(ax, (reg_x + reg_w, plus_c[1]), (plus_c[0] - 0.34, plus_c[1]),
          color=ur_style.NAVY, lw=1.8)

    # ---- bypass: target_q added back unchanged ---------------------------
    pos_out = in_boxes["pos"]
    _arrow(ax, pos_out, (plus_c[0], plus_c[1] + 0.34),
          color=ur_style.MID_BLUE, style="--", lw=1.8, rad=-0.28, z=2.5)
    ax.text(7.7, 0.05,
            "target_q$_j$ (the commanded position) is added back unchanged —\n"
            "the regressor only ever has to explain the small residual, not the whole trajectory",
            ha="center", va="bottom", fontsize=9.3, color=ur_style.MID_BLUE, style="italic")

    # ---- output -----------------------------------------------------------
    out_x, out_w, out_h = 11.15, 2.05, 1.1
    out_y = plus_c[1] - out_h / 2
    _box(ax, (out_x, out_y), out_w, out_h, "predicted\nactual_q$_j$  (rad)",
        face=ur_style.NAVY, edge=ur_style.NAVY, text_color="white",
        fontsize=10.5, fontweight="bold")
    _arrow(ax, (plus_c[0] + 0.34, plus_c[1]), (out_x, out_y + out_h / 2),
          color=ur_style.NAVY, lw=1.8)

    ax.text(6.7, 8.15, "output", ha="center", fontsize=11.5,
            fontweight="bold", color=ur_style.NAVY)

    # ---- "x6 independent joints" framing ----------------------------------
    ax.add_patch(FancyBboxPatch((0.15, -0.5), 13.2, 8.75, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="none", edgecolor=ur_style.GRAY,
                                linewidth=1.2, linestyle=(0, (6, 4)), zorder=1))
    ax.text(13.15, -0.28, "× 6", ha="right", va="bottom", fontsize=13,
            fontweight="bold", color=ur_style.GRAY)
    fig.text(0.5, 0.015,
            "One independently-fit regressor per joint (base … wrist3) — six separate models, no shared weights.",
            ha="center", fontsize=10, color=ur_style.GRAY)

    fig.suptitle("Per-Joint Position Model — Architecture", fontsize=16, y=0.985,
                fontweight="bold", color=ur_style.NAVY)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    plot_model_diagram()
