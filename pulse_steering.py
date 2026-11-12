"""Integrate a phenomenological two-state enantiomer steering model."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


def simulate_laser_steering(
    H0_matrix: NDArray[np.complex128],
    mu: float = 1.2,
    E0: float = 1.8,
    omega: float = 0.3,
    sigma: float = 3.0,
    t0: float = 10.0,
    t_span: tuple[float, float] = (0.0, 20.0),
    *,
    plot_path: str | Path | None = None,
    plot: bool = True,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Evolve a coherent two-state R/S wavepacket under a Gaussian pulse.

    The basis is ``(|R>, |S>)`` and ``H(t) = H0 - mu E(t) sigma_z``, with
    ``hbar = 1`` and a cosine-carrier Gaussian field. The initial pure state
    ``(|R> + |S>)/sqrt(2)`` has equal populations but is coherent; it is not an
    incoherent racemic ensemble. The resulting enantiomeric excess therefore
    describes this effective two-state model, not a molecular prediction.

    Returns the times, R and S populations, and final enantiomeric excess in %.
    If ``plot_path`` is supplied, the trajectory is saved there; otherwise the
    plot is displayed. Set plot=False and verbose=False for parameter sweeps.
    """
    h0 = np.asarray(H0_matrix, dtype=np.complex128)
    if h0.shape != (2, 2):
        raise ValueError(f"H0_matrix must have shape (2, 2), got {h0.shape}.")
    if not np.isfinite(h0).all():
        raise ValueError("H0_matrix must contain only finite values.")
    if not np.allclose(h0, h0.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError("H0_matrix must be Hermitian.")
    # A scalar energy shift only changes global phase; removing it avoids
    # resolving a large electronic reference energy during every pulse trial.
    h0 = h0 - np.trace(h0) / 2.0 * np.eye(2)

    if not all(isfinite(value) for value in (mu, E0, omega, sigma, t0)):
        raise ValueError("Pulse parameters must be finite real numbers.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if (
        len(t_span) != 2
        or not all(isfinite(value) for value in t_span)
        or t_span[0] >= t_span[1]
    ):
        raise ValueError("t_span must contain finite, increasing start and end times.")

    times = np.linspace(t_span[0], t_span[1], 801, dtype=np.float64)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    initial_state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)

    def schrodinger_rhs(time: float, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        envelope = np.exp(-0.5 * ((time - t0) / sigma) ** 2)
        field = E0 * envelope * np.cos(omega * (time - t0))
        hamiltonian = h0 - mu * field * sigma_z
        return -1j * hamiltonian @ state

    solution = solve_ivp(
        schrodinger_rhs,
        t_span,
        initial_state,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Schrodinger integration failed: {solution.message}")

    states = solution.y.T
    norms = np.sum(np.abs(states) ** 2, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Wavefunction norm drift exceeded the allowed tolerance.")
    populations = np.abs(states) ** 2 / norms[:, np.newaxis]
    population_r = populations[:, 0].astype(np.float64, copy=False)
    population_s = populations[:, 1].astype(np.float64, copy=False)
    final_ee = float(100.0 * (population_r[-1] - population_s[-1]))

    if verbose:
        print(f"Final enantiomeric excess (effective model): {final_ee:.2f}%")
    if not plot:
        return times, population_r, population_s, final_ee

    figure, axis = plt.subplots()
    axis.plot(times, population_r, label=r"$P_R$")
    axis.plot(times, population_s, label=r"$P_S$")
    axis.set_xlabel("Time (model units)")
    axis.set_ylabel("Population")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"Final enantiomeric excess: {final_ee:.2f}%")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        destination = Path(plot_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
        plt.close(figure)

    return times, population_r, population_s, final_ee


def optimize_laser_parameters(
    H_0: NDArray[np.complex128], mu: float = 1.2, *,
    plot_path: str | Path = "chbrclf_stereocontrol.png",
) -> dict[str, float]:
    """Maximize signed final R excess on a 15 by 15 pulse grid.

    All trials use t0=32 and a common readout at t=64 (four maximum
    widths on each side). This optimizes a specified readout, not persistent
    localization: field-free tunneling resumes after the pulse. The gap is
    calculated from both eigenvalues, including any diagonal bias.

    In this Hamiltonian the resonant rotating-wave Rabi rate is mu*E0
    (hbar=1). The envelope pi area is only a search scale: an energy-basis
    pi pulse does not itself guarantee localized R/S population inversion.
    """
    if not isfinite(mu) or mu == 0:
        raise ValueError("mu must be finite and nonzero.")
    # Reuse the simulator's input checks before computing the spectral gap.
    simulate_laser_steering(H_0, mu=mu, E0=0, plot=False, verbose=False)
    levels = np.linalg.eigvalsh(H_0)
    omega = float(levels[1] - levels[0])
    best: dict[str, float] = {"ee": -float("inf")}
    for sigma in np.linspace(1.0, 8.0, 15):
        nominal = np.pi / (abs(mu) * sigma * np.sqrt(2 * np.pi))
        for scale in np.linspace(0.5, 2.0, 15):
            amplitude = float(nominal * scale)
            *_, ee = simulate_laser_steering(
                H_0, mu=mu, E0=amplitude, omega=omega, sigma=float(sigma),
                t0=32.0, t_span=(0.0, 64.0), plot=False, verbose=False,
            )
            if ee > best["ee"]:
                best = dict(E0=amplitude, sigma=float(sigma), omega=omega,
                            t0=32.0, t_start=0.0, t_end=64.0, ee=ee)
    simulate_laser_steering(
        H_0, mu=mu, E0=best["E0"], omega=omega, sigma=best["sigma"],
        t0=best["t0"], t_span=(best["t_start"], best["t_end"]),
        plot_path=plot_path,
    )
    print(f"Best grid parameters: {best}")
    return best
