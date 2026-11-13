"""VQE-to-laser Toy Demonstration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from scipy.optimize import minimize


# Step 1: a supplied 4x4 toy matrix, not an ab initio molecular calculation.
H_mol = np.array(
    [
        [-0.800, 0.000, 0.000, 0.000],
        [0.000, -1.137, 0.082, 0.000],
        [0.000, 0.082, -0.450, 0.000],
        [0.000, 0.000, 0.000, -0.200],
    ],
    dtype=float,
)


# Step 2: prepare a two-qubit trial state using exactly three gates.
def make_vqe_circuit(theta: float) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.ry(theta, 1)
    circuit.cx(1, 0)
    return circuit


def energy_for_angle(parameters: np.ndarray) -> float:
    state = Statevector(make_vqe_circuit(float(parameters[0]))).data
    return float(np.real(state.conj().T @ H_mol @ state))


optimization = minimize(energy_for_angle, x0=[0.0], method="BFGS", tol=1e-12)
theta = float(optimization.x[0])
E_ideal = float(optimization.fun)
ideal_state = Statevector(make_vqe_circuit(theta)).data
rho_ideal = np.outer(ideal_state, ideal_state.conj())

# This is a simple mixed-state noise estimate, not a hardware noise simulation.
rho_noisy = 0.98 * rho_ideal + (0.02 / 4) * np.eye(4)
E_noisy = float(np.trace(H_mol @ rho_noisy).real)


# Step 3: use the toy VQE result to set the model's resonance reference.
E_excited = -0.950
exact_gap = E_excited - E_ideal
laser_frequency = exact_gap  # hbar = 1, so energy and angular frequency share units.
vqe_detuning = abs(laser_frequency - exact_gap)
noise_detuning = abs(E_ideal - E_noisy)
mistuned_detuning = 0.06


# Step 4: repeat a simple three-level pump-and-relax cycle six times.
LEFT, RIGHT, EXCITED = range(3)
INITIAL_DENSITY = np.diag([0.5, 0.5, 0.0]).astype(complex)
CYCLE_COUNT = 6


def distillation_curve(detuning: float) -> np.ndarray:
    """Return right-state population after each pump and 50/50 relaxation."""
    omega = 0.02
    t_pulse = np.pi / (2 * omega)
    H_pulse = np.array(
        [
            [0.0, 0.0, omega],
            [0.0, 0.0, 0.0],
            [omega, 0.0, detuning],
        ],
        dtype=complex,
    )
    unitary = expm(-1j * H_pulse * t_pulse)

    # Keep existing ground-state populations; split excited population equally.
    keep_ground = np.diag([1.0, 1.0, 0.0]).astype(complex)
    decay_left = np.zeros((3, 3), dtype=complex)
    decay_right = np.zeros((3, 3), dtype=complex)
    decay_left[LEFT, EXCITED] = 1 / np.sqrt(2)
    decay_right[RIGHT, EXCITED] = 1 / np.sqrt(2)
    relaxation = (keep_ground, decay_left, decay_right)

    density = INITIAL_DENSITY.copy()
    right_populations = [float(density[RIGHT, RIGHT].real)]
    for _ in range(CYCLE_COUNT):
        density = unitary @ density @ unitary.conj().T
        density = sum(
            (operator @ density @ operator.conj().T for operator in relaxation),
            start=np.zeros((3, 3), dtype=complex),
        )
        ground_population = density[LEFT, LEFT].real + density[RIGHT, RIGHT].real
        right_populations.append(
            float(density[RIGHT, RIGHT].real / ground_population)
        )
    return np.array(right_populations)


curves = {
    "VQE-tuned laser": (vqe_detuning, distillation_curve(vqe_detuning)),
    "2% depolarizing-noise model": (
        noise_detuning,
        distillation_curve(noise_detuning),
    ),
    "Mistuned laser (no VQE)": (
        mistuned_detuning,
        distillation_curve(mistuned_detuning),
    ),
}

figure, axis = plt.subplots(figsize=(8, 5))
cycles = np.arange(CYCLE_COUNT + 1)
for label, (detuning, populations) in curves.items():
    axis.plot(
        cycles,
        100 * populations,
        marker="o",
        linewidth=2,
        label=f"{label} (detuning={detuning:.4f})",
    )
axis.set(
    title="Toy-model right-state population after repeated laser cycles",
    xlabel="Distillation cycle",
    ylabel="Right-state population (%)",
    xticks=cycles,
    ylim=(45, 101),
)
axis.grid(True, alpha=0.3)
axis.legend()
figure.tight_layout()
plot_path = Path(__file__).resolve().with_name("chiral_distillation.png")
figure.savefig(plot_path, dpi=180)
plt.close(figure)


print(f"VQE circuit gates: {sum(make_vqe_circuit(theta).count_ops().values())}")
print(f"Best angle: {theta:.4f} radians")
print(f"E_ideal: {E_ideal:.6f} Ha (toy matrix units)")
print(f"E_noisy: {E_noisy:.6f} Ha (2% mixed-state estimate)")
print(f"Exact model energy gap: {exact_gap:.6f} Ha (hbar = 1)")
for label, (detuning, populations) in curves.items():
    print(
        f"{label}: detuning={detuning:.6f}, "
        f"final right-state population={100 * populations[-1]:.2f}%"
    )
print(f"Saved chart to {plot_path}")
