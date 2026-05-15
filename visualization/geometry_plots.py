"""Geometry visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.conductors import StackedSlotCable
from data.coils import Solenoid


def plot_cable_cross_section(cable, ax=None):
    """Plot cable cross-section for CICC or stacked-slot assemblies."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 4))

    if isinstance(cable, StackedSlotCable):
        # Circular SS jacket
        ax.add_patch(plt.Circle((0, 0), cable.jacket_outer_radius,
                                fc="#2e2e2e", ec="silver", lw=1.5,
                                label="SS jacket"))

        # Copper former
        ax.add_patch(plt.Circle((0, 0), cable.former_outer_radius,
                                fc="#B5B5B5", alpha=0.5, ec='#888888', lw=1.2,
                                label=f"{cable.former_material} former"))

        # Central helium channel
        ax.add_patch(plt.Circle((0, 0), cable.helium_radius,
                                fc="#3399FF", alpha=0.7, ec='#1166CC', lw=0.8,
                                label="He channel"))
        ax.text(0, 0, "He", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

        # HTS stacks as rectangles
        block_w = cable.stack.width
        block_h = cable.stack.height
        for i, (cx, cy) in enumerate(cable.stack_positions):
            rect = mpatches.Rectangle((cx - block_w/2, cy - block_h/2), block_w, block_h,
                                      fc="#FFCC44", ec="#996600", lw=0.8,
                                      label="HTS stack" if i == 0 else "")
            ax.add_patch(rect)

        xlim = cable.jacket_outer_radius * 1.1
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-xlim, xlim)
        ax.set_aspect('equal')
        ax.set_title(f"VIPER cable cross-section\n"
                     f"({cable.N_slots} stacks, OD={cable.jacket_outer_diameter*1e3:.0f} mm)", fontsize=9)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.grid(alpha=0.15)
        # Add scale bar
        scale_length = 5e-3  # 5 mm
        ax.plot([-xlim + 1e-3, -xlim + 1e-3 + scale_length], [-xlim + 1e-3, -xlim + 1e-3], 'k-', lw=2)
        ax.text(-xlim + 1e-3 + scale_length/2, -xlim + 2e-3, f'{scale_length*1e3:.0f} mm', ha='center', va='bottom', fontsize=8)
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    s = cable.conduit_side
    jk = cable.jacket_thick
    r_avail = s / 2 - jk

    ax.add_patch(plt.Rectangle((-s/2, -s/2), s, s,
                                fc="#2e2e2e", ec="silver", lw=1.5,
                                label="SS jacket"))

    r_he = r_avail * 0.28
    ax.add_patch(plt.Circle((0, 0), r_he,
                              fc="#3399FF", alpha=0.7, ec='#1166CC', lw=0.8,
                              label="He channel"))
    ax.text(0, 0, "He", ha='center', va='center',
            fontsize=7, color='white', fontweight='bold')

    r_tape = r_avail * 0.62
    w = cable.tape.width
    t = cable.tape.thickness * 4
    w = min(w, r_avail * 0.9)

    angles = np.linspace(0, 2*np.pi, cable.N_tapes, endpoint=False)
    for i, theta in enumerate(angles):
        cx = r_tape * np.cos(theta)
        cy = r_tape * np.sin(theta)
        half_w = w / 2
        half_t = t / 2
        corners_local = np.array([
            [-half_w, -half_t],
            [ half_w, -half_t],
            [ half_w,  half_t],
            [-half_w,  half_t],
        ])
        rot = np.array([[np.cos(theta + np.pi/2), -np.sin(theta + np.pi/2)],
                        [np.sin(theta + np.pi/2),  np.cos(theta + np.pi/2)]])
        corners = corners_local @ rot.T + [cx, cy]
        ax.add_patch(plt.Polygon(corners, fc="#FFCC44", ec="#996600", lw=0.8,
                                  label="HTS tape" if i == 0 else ""))

    ax.set_xlim(-s * 0.62, s * 0.62)
    ax.set_ylim(-s * 0.62, s * 0.62)
    ax.set_aspect('equal')
    ax.set_title(f"CICC cross-section\n"
                 f"({cable.N_tapes} HTS tapes, "
                 f"{cable.conduit_side*1e3:.0f} mm conduit)", fontsize=9)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.15)
    if standalone:
        plt.tight_layout(); plt.show()
    return ax


def plot_solenoid_cross_section(sol: Solenoid, ax=None):
    """Plot solenoid r-z cross-section."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 7))

    d = sol.cable.conduit_side
    for lay in sol.layers:
        ax.add_patch(plt.Rectangle(
            (lay.r_center - d/2, lay.z_center - d/2), d, d,
            fc="#FFCC44", ec="#996600", lw=0.4, alpha=0.9))

    ax.axhspan(-sol.height/2, sol.height/2,
               xmin=0, xmax=sol.r_inner / (sol.r_outer * 1.4),
               fc="#3399FF", alpha=0.15, label="Warm bore")
    ax.axvline(0, color='gray', lw=0.8, ls='--', label='Symmetry axis')

    ax.set_xlim(0, sol.r_outer * 1.35)
    ax.set_ylim(-sol.height * 0.65, sol.height * 0.65)
    ax.set_xlabel("r  (m)"); ax.set_ylabel("z  (m)")
    ax.set_title(f"Solenoid r-z cross-section\n"
                 f"{sol.n_radial} radial × {sol.n_axial} axial = "
                 f"{sol.total_turns} turns", fontsize=9)
    ax.set_aspect('equal')
    ax.legend(fontsize=8); ax.grid(alpha=0.15)

    y_ann = -sol.height / 2 * 1.2
    ax.annotate('', xy=(sol.r_outer, y_ann), xytext=(sol.r_inner, y_ann),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
    ax.text((sol.r_inner + sol.r_outer)/2, y_ann - sol.height*0.06,
            f"Δr = {(sol.r_outer-sol.r_inner)*100:.1f} cm",
            ha='center', fontsize=8, color='red')

    if standalone:
        plt.tight_layout(); plt.show()
    return ax


def plot_geometry_overview(sol: Solenoid, cable):
    """Plot combined geometry overview."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Solenoid Geometry Overview", fontsize=13, fontweight='bold')
    plot_cable_cross_section(cable, ax=ax1)
    plot_solenoid_cross_section(sol, ax=ax2)
    plt.tight_layout()
    plt.savefig("solenoid_geometry.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: solenoid_geometry.png")