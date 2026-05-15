"""Magnetic field visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.coils import Solenoid
from data.solutions import MagneticFieldSolution


def plot_bfield(sol: MagneticFieldSolution, solenoid: Solenoid):
    """Plot magnetic field solution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Magnetic Field  B(r, z)", fontsize=13, fontweight='bold')

    R2d, Z2d = np.meshgrid(sol.r, sol.z, indexing='ij')

    ax = axes[0]
    cm = ax.contourf(Z2d*100, R2d*100, sol.Bmag, levels=40, cmap='plasma')
    plt.colorbar(cm, ax=ax, label='|B| (T)', shrink=0.85)
    ax.streamplot(sol.z*100, sol.r*100,
                  sol.Bz, sol.Br,
                  color='white', density=1.0,
                  linewidth=0.6, arrowsize=0.8)
    ax.add_patch(plt.Rectangle(
        (-solenoid.height/2*100, solenoid.r_inner*100),
        solenoid.height*100, (solenoid.r_outer - solenoid.r_inner)*100,
        fc='none', ec='cyan', lw=1.5, ls='--', label='Winding'))
    ax.set_xlabel("z (cm)"); ax.set_ylabel("r (cm)")
    ax.set_title("|B| + field lines")
    ax.legend(fontsize=8)

    ax = axes[1]
    vmax = np.percentile(np.abs(sol.Bz), 98)
    cm2 = ax.contourf(Z2d*100, R2d*100, sol.Bz,
                        levels=np.linspace(-vmax, vmax, 41), cmap='RdBu_r')
    plt.colorbar(cm2, ax=ax, label='Bz (T)', shrink=0.85)
    ax.set_xlabel("z (cm)"); ax.set_ylabel("r (cm)")
    ax.set_title("Bz component (signed)")

    ax = axes[2]
    Bz_axis = sol.Bz[0, :]
    ax.plot(sol.z*100, Bz_axis, color='#FF6B35', lw=2, label='Bz  (r=0)')
    ax.axhline(15.0, color='gray', ls='--', lw=1.2,
               label='Target 15 T')
    ax.axvspan(-solenoid.height/2*100, solenoid.height/2*100,
               alpha=0.12, color='blue', label='Winding span')
    ax.set_xlabel("z (cm)"); ax.set_ylabel("Bz (T)")
    ax.set_title("On-axis field profile")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("solenoid_bfield.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: solenoid_bfield.png")