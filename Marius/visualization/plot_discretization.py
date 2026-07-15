"""Exact (matrix-exponential) vs. Euler discretization of the TES state equation.

The TES is the leaky integrator
    dE/dt = -E/tau_loss + eta_in * Q_in - Q_out / eta_out,
which core.py discretizes exactly under a zero-order hold on the fluxes:
    a  = exp(-dt/tau_loss),   b1 = eta_in * tau_loss * (1 - a),   b2 = -(1/eta_out) * tau_loss * (1 - a).
Explicit Euler instead uses a_E = 1 - dt/tau_loss, b1_E = eta_in * dt, b2_E = -dt/eta_out.

The exact discretization is the reference here: for a flux held constant over each step -- which is
exactly what the optimization models assume -- it reproduces the continuous solution at the grid
points for *any* step size, so there is nothing to compare it against.  Euler is the only scheme
with an error, and it is measured against the exact one.

Two figures:
  1. discretization_decay.png       -- free decay (no fluxes), from Marius/notebooks/discretization.ipynb.
     The state-matrix error is O(dt^2 / tau_loss^2) and stays invisible for the dt used here.
  2. discretization_alternating.png -- alternating charging/discharging square wave.  The input
     matrices carry the much larger O(dt/tau_loss) relative error, so Euler drifts off the exact
     trajectory, and the drift grows with dt.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.optimization.core import tau_loss, eta_in_TES, eta_out_TES, E_nom_TES
from src.visualization.style import apply_style

OUT_DIR = ROOT / "Marius" / "visualization"

# Continuous-time state-space model: dE/dt = A*E + B @ [Q_in, Q_out]
A = -1.0 / tau_loss
B = np.array([eta_in_TES, -1.0 / eta_out_TES])

# Flux schedule: start empty, charge up to E_PEAK, then leave the store standing for the rest of the
# week.  Two things are visible at once, and nothing else is in the way:
#   * the charging phase is where Euler's input-matrix error enters -- it ends the phase above the
#     exact charge, by eps * E_PEAK with eps = |db/b|;
#   * the idle phase is the homogeneous response, a pure exp(-t/tau_loss) decay.  Over 150 h (0.75
#     time constants) its curvature is plain.  Cycling the store every 20 h hides that completely:
#     the state then never traverses more than 0.1 time constants at a stretch, over which an
#     exponential is a straight line to within half a percent.
# E_PEAK leaves headroom against E_nom on purpose: the MILP constrains E_TES <= E_nom, so a profile
# that rode the bound and let Euler poke above it would be one no solver could ever return.
T_CHARGE = 10.0                                 # charging phase [h]
T_IDLE = 158.0                                  # idle phase: free decay, no flux [h]
T_FINAL = T_CHARGE + T_IDLE                     # 168 h = N * DELTA_T, the full model horizon
E_START = 0.0                                   # start empty [kWh]
E_PEAK = 0.90 * E_nom_TES                       # charge reached at the end of the charging phase [kWh]
# The step sizes only have to divide the charging phase, so the flux switches on a step boundary.
# They need not divide the horizon: simulate() takes a shorter final step where they do not, which
# is how a 10 h grid still ends exactly on 168 h.
DT_LIST = [1.0, 10.0]                           # step sizes shown as trajectories
ZOOM_X = (T_CHARGE - 8.0, T_CHARGE + 24.0)      # inset window around the end of the charging phase
ZOOM_Y = (E_PEAK - 55.0, E_PEAK + 30.0)
DT_SWEEP = [0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 12.0]

C_EXACT = "#2166AC"   # blue  -- exact discretization
C_EULER = "#D6604D"   # red   -- Euler approximation
C_CHARGE = "#4DAC26"  # green -- charging phase shading


def save_figure(fig: plt.Figure, stem: str) -> Path:
    """Save to OUT_DIR/<stem>.png and .pdf."""
    png, pdf = OUT_DIR / f"{stem}.png", OUT_DIR / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def exact_matrices(dt: float) -> tuple[float, np.ndarray]:
    """Zero-order-hold discretization: a = exp(A*dt), b = B * int_0^dt exp(A*tau) dtau."""
    a = np.exp(A * dt)
    return a, B * (a - 1.0) / A


def euler_matrices(dt: float) -> tuple[float, np.ndarray]:
    """Explicit-Euler discretization: first-order truncation of the exact matrices."""
    return 1.0 + A * dt, B * dt


def db_over_b(dt: float) -> float:
    """Relative error of Euler's input matrix.  Same for both entries of b, so report the first."""
    b = exact_matrices(dt)[1][0]
    return (euler_matrices(dt)[1][0] - b) / b


def step_lengths(dt: float, t_final: float) -> list[float]:
    """Steps of length dt, with a shorter final step if dt does not divide the horizon.

    Both schemes take the step length as an argument, so a ragged last step is perfectly well
    defined -- and it is what lets a 10 h grid still land exactly on the 168 h horizon.
    """
    n_full = int(np.floor(t_final / dt + 1e-9))
    remainder = t_final - n_full * dt
    return [dt] * n_full + ([remainder] if remainder > 1e-9 else [])


def simulate(dt: float, matrices, flux, t_final: float = T_FINAL, x0: float | None = None):
    """Roll the discrete state equation forward.  `flux(t)` returns [Q_in, Q_out] for a step."""
    x = E_START if x0 is None else x0
    ts, xs, t = [0.0], [x], 0.0
    for h in step_lengths(dt, t_final):
        a, b = matrices(h)
        x = a * x + b @ flux(t)
        t += h
        ts.append(t)
        xs.append(x)
    return np.array(ts), np.array(xs)


def no_flux(t: float) -> np.ndarray:
    return np.zeros(2)


def charge_then_idle(q: float):
    """Charge at q for T_CHARGE hours, then no flux at all for the rest of the horizon."""
    def flux(t: float) -> np.ndarray:
        return np.array([q, 0.0]) if t < T_CHARGE - 1e-9 else np.zeros(2)
    return flux


def charging_flux() -> float:
    """The flux that takes the store from empty to E_PEAK over T_CHARGE hours.

    Held constant over the phase, the exact solution is a*0 + b1*q, so the flux follows directly.
    """
    return E_PEAK / exact_matrices(T_CHARGE)[1][0]


Q_FLUX = charging_flux()  # charging flux [kW]


def print_matrix_deviation() -> None:
    """Relative error of the Euler matrices w.r.t. the exact ones, per step size."""
    print(f"{'dt [h]':>8} {'dA/A':>12} {'db1/b1':>12} {'db2/b2':>12}")
    for dt in DT_LIST:
        a, b = exact_matrices(dt)
        a_E, b_E = euler_matrices(dt)
        print(f"{dt:>8.2f} {(a - a_E) / a:>12.2e} "
              f"{(b[0] - b_E[0]) / b[0]:>12.2e} {(b[1] - b_E[1]) / b[1]:>12.2e}")


def plot_decay() -> Path:
    """Free decay of a full TES: exact vs. Euler, with no flux at all."""
    runs = {}
    for dt in DT_LIST:
        ts, xs_exact = simulate(dt, exact_matrices, no_flux, x0=E_nom_TES)
        _, xs_euler = simulate(dt, euler_matrices, no_flux, x0=E_nom_TES)
        runs[dt] = (ts, xs_exact, xs_euler)

    # Zoom the tail: with no flux the two schemes separate slowly, and the whole of the gap has
    # accumulated by the end of the horizon.  The y window is taken from the data actually inside
    # the x window, so the curves fill the inset instead of running off the top of it.
    zoom_x = (T_FINAL - 2 * max(DT_LIST), T_FINAL + 0.1 * max(DT_LIST))
    inside = [xs[ts >= zoom_x[0]] for ts, *pair in runs.values() for xs in pair]
    lo = min(xs.min() for xs in inside)
    hi = max(xs.max() for xs in inside)
    zoom_y = (lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))

    apply_style(width_cm=22, aspect=2.6, grid=True, strict=True)
    fig, axes = plt.subplots(1, len(DT_LIST), sharex=True, sharey=True, constrained_layout=True)

    for ax, dt in zip(axes, DT_LIST):
        ts, xs_exact, xs_euler = runs[dt]
        peak = np.abs(xs_euler - xs_exact).max()

        ax.axhline(E_nom_TES, color="black", linewidth=1.4, linestyle=":", label=r"$E_{\rm{nom}}$")
        ax.plot(ts, xs_exact, color=C_EXACT, linewidth=1.5, label="Exact")
        ax.plot(ts, xs_euler, color=C_EULER, linewidth=1.5, linestyle="--", label="Euler")

        ax.set_title(rf"$\Delta t = {dt:g}$ h  (Euler off by {peak:.1f} kWh)")
        ax.set_xlabel(r"Time $t$ [h]")
        ax.set_xticks(np.arange(0, T_FINAL + 1, 24))
        ax.legend(loc="upper right")

        # Lower left: the only corner the decay curve leaves free.
        every = max(1, int(round(5.0 / dt)))
        axins = ax.inset_axes([0.09, 0.10, 0.30, 0.34])
        axins.plot(ts, xs_exact, color=C_EXACT, linewidth=1.5, marker="o", markersize=3, markevery=every)
        axins.plot(ts, xs_euler, color=C_EULER, linewidth=1.5, linestyle="--", marker="s", markersize=3, markevery=every)
        axins.set_xlim(*zoom_x)
        axins.set_ylim(*zoom_y)
        axins.tick_params(axis="both", labelsize=6, pad=1)
        axins.minorticks_off()
        ax.indicate_inset_zoom(axins, edgecolor="0.6")

        print(f"dt = {dt:>5.2f} h:  free decay, Euler off by up to {peak:>5.2f} kWh; "
              f"ends at {xs_euler[-1]:.1f} vs {xs_exact[-1]:.1f} kWh (exact)")

    axes[0].set_ylabel(r"TES charge $E_{\rm{TES}}$ [kWh]")
    fig.suptitle("TES free decay (no charging or discharging)")

    return save_figure(fig, "discretization_decay")


def plot_alternating() -> Path:
    """Full-depth cycling -- the worst case for Euler -- against the exact discretization."""
    flux = charge_then_idle(Q_FLUX)

    apply_style(width_cm=22, aspect=2.6, grid=True, strict=True)
    fig, axes = plt.subplots(1, len(DT_LIST), sharex=True, sharey=True, constrained_layout=True)

    for ax, dt in zip(axes, DT_LIST):
        ts, xs_exact = simulate(dt, exact_matrices, flux)
        _, xs_euler = simulate(dt, euler_matrices, flux)
        peak = np.abs(xs_euler - xs_exact).max()

        ax.axvspan(0.0, T_CHARGE, color=C_CHARGE, alpha=0.12, linewidth=0, label="charging")
        ax.axhline(E_nom_TES, color="black", linewidth=1.4, linestyle=":", label=r"$E_{\rm{nom}}$")

        ax.plot(ts, xs_exact, color=C_EXACT, linewidth=1.5, label="Exact")
        ax.plot(ts, xs_euler, color=C_EULER, linewidth=1.5, linestyle="--", label="Euler")

        ax.set_title(rf"$\Delta t = {dt:g}$ h  (Euler off by {peak:.1f} kWh)")
        ax.set_xlabel(r"Time $t$ [h]")
        ax.set_xticks(np.arange(0, T_FINAL + 1, 24))

        # Zoom the end of the charging phase, where Euler's error is widest.  Both panels use the
        # same window, so the indicator rectangle is visible (and comparable) in both -- sizing it
        # per-dt would shrink the dt = 1 h box to a sub-pixel smear.  Markers are thinned to one
        # every ~5 h: at dt = 1 h a marker per step is taller than the gap between steps, and the
        # overlapping squares read as a staircase that is not in the data.
        every = max(1, int(round(5.0 / dt)))
        axins = ax.inset_axes([0.50, 0.11, 0.30, 0.34])
        axins.plot(ts, xs_exact, color=C_EXACT, linewidth=1.5, marker="o", markersize=3, markevery=every)
        axins.plot(ts, xs_euler, color=C_EULER, linewidth=1.5, linestyle="--", marker="s", markersize=3, markevery=every)
        axins.set_xlim(*ZOOM_X)
        axins.set_ylim(*ZOOM_Y)
        axins.tick_params(axis="both", labelsize=6, pad=1)
        axins.minorticks_off()
        ax.indicate_inset_zoom(axins, edgecolor="0.6")

        print(f"dt = {dt:>5.2f} h:  peak |E_Euler - E_exact| = {peak:>6.2f} kWh  "
              f"({peak / E_nom_TES:.2%} of E_nom, b error {abs(db_over_b(dt)):.2%}), "
              f"Euler stays within [{xs_euler.min():.1f}, {xs_euler.max():.1f}] kWh")

    axes[0].set_ylabel(r"TES charge $E_{\rm{TES}}$ [kWh]")
    for ax in axes:
        ax.legend(loc="upper right")
    fig.suptitle("TES diff.eq. discretization comparison")

    return save_figure(fig, "discretization_alternating")


def plot_convergence() -> Path:
    """Euler's error against the step size: first order in dt, exactly as the b-matrix predicts.

    This is the case for the exact discretization: its error is zero at *every* dt (it is the
    reference), while Euler's grows linearly with dt -- and it costs nothing extra, since a and b
    are constants computed once.
    """
    flux = charge_then_idle(Q_FLUX)

    peaks, swings = [], []
    for dt in DT_SWEEP:
        _, xs_exact = simulate(dt, exact_matrices, flux)
        _, xs_euler = simulate(dt, euler_matrices, flux)
        peaks.append(np.abs(xs_euler - xs_exact).max())
        swings.append(np.ptp(xs_exact))

    rel = np.array(peaks) / np.array(swings)
    predicted = np.array([abs(db_over_b(dt)) for dt in DT_SWEEP])

    apply_style(width_cm=16, aspect="golden", grid=True, strict=True)
    fig, ax = plt.subplots(constrained_layout=True)

    ax.loglog(DT_SWEEP, rel, color=C_EULER, marker="s", markersize=5, linestyle="none",
              label="Euler: measured peak error")
    ax.loglog(DT_SWEEP, predicted, color=C_EULER, linewidth=1.2, linestyle="--",
              label=r"Euler: predicted, $|\Delta b / b| \approx \Delta t / 2\tau_{\rm{loss}}$")

    ax.set_xlabel(r"Step size $\Delta t$ [h]")
    ax.set_ylabel(r"Peak $|E_{\rm{Euler}} - E_{\rm{exact}}|$ / swing $[-]$")
    ax.set_ylim(4e-4, 1e-1)
    ax.set_title("Euler's error is first order in the step size; the exact scheme has none")
    ax.legend(loc="upper left")

    # The exact scheme's error is identically zero at every dt, which cannot be drawn on a log
    # axis -- and faking it with a line at 1e-15 would stretch the scale over 15 dead decades.
    ax.text(0.97, 0.06, "Exact discretization: error $\\equiv 0$ at every $\\Delta t$\n"
                        "(exact for a zero-order-hold flux, at no extra cost)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C_EXACT,
            bbox=dict(facecolor="white", edgecolor=C_EXACT, alpha=0.9, pad=3.0))

    # Mark the step size the models actually run at, and a coarse aggregated one.
    for dt_mark, note in [(1.0, "hourly model"), (12.0, "aggregated")]:
        i = DT_SWEEP.index(dt_mark)
        ax.annotate(rf"$\Delta t = {dt_mark:g}$ h: {rel[i]:.2%} ({note})",
                    xy=(dt_mark, rel[i]), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=C_EULER)

    path = save_figure(fig, "discretization_convergence")

    print("\nconvergence sweep (peak Euler error as a fraction of the swing):")
    for dt, r, p in zip(DT_SWEEP, rel, predicted):
        print(f"  dt = {dt:>5.2f} h:  measured {r:>7.3%}   predicted |db/b| {p:>7.3%}")
    return path


if __name__ == "__main__":
    print_matrix_deviation()
    print()
    print(f"Saved {plot_alternating()}")
    print(f"Saved {plot_decay()}")
    print(f"Saved {plot_convergence()}")
