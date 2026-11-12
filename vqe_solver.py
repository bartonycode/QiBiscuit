"""Variational and exact eigensolvers for the reduced molecular Hamiltonian."""

from __future__ import annotations

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import (
    NumPyMinimumEigensolver,
    VQE,
)
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.minimum_eigensolvers.vqe import VQEResult
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


def run_vqe_simulation(
    reduced_problem: ElectronicStructureProblem,
    mapper: ParityMapper,
) -> tuple[VQEResult, MinimumEigensolverResult]:
    """Run UCCSD VQE and an exact NumPy benchmark on the active-space problem.

    Printed eigenvalues are for the mapped reduced electronic operator (in
    Hartree); they do not include interpretation as total molecular energies.
    """
    qubit_hamiltonian = mapper.map(reduced_problem.hamiltonian.second_q_op())
    if qubit_hamiltonian.num_qubits != 2:
        raise ValueError(
            "VQE expects the requested two-qubit active-space Hamiltonian; "
            f"got {qubit_hamiltonian.num_qubits} qubits."
        )

    initial_state = HartreeFock(
        num_spatial_orbitals=reduced_problem.num_spatial_orbitals,
        num_particles=reduced_problem.num_particles,
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=reduced_problem.num_spatial_orbitals,
        num_particles=reduced_problem.num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
    )
    if ansatz.num_qubits != qubit_hamiltonian.num_qubits:
        raise ValueError(
            "The UCCSD ansatz and Hamiltonian have different qubit counts."
        )

    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=SLSQP(maxiter=250, ftol=1e-8),
    )
    vqe_result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)

    exact_solver = NumPyMinimumEigensolver()
    exact_result = exact_solver.compute_minimum_eigenvalue(qubit_hamiltonian)

    vqe_energy = float(np.real(vqe_result.eigenvalue))
    exact_energy = float(np.real(exact_result.eigenvalue))
    print(f"VQE reduced electronic ground-state energy: {vqe_energy:.12f} Ha")
    print(f"Exact reduced electronic ground-state energy: {exact_energy:.12f} Ha")
    print(f"Absolute energy difference: {abs(vqe_energy - exact_energy):.3e} Ha")
    return vqe_result, exact_result


def measure_vqe_readout(
    result: VQEResult, *, shots: int = 8192, seed: int = 42,
    error_probability: float = 0.02,
) -> dict:
    """Measure theta* with synthetic independent symmetric readout flips.

    Returns sampled basis-state assignment errors and the VQE distribution's
    total variation distance from its exact computational-basis probabilities.
    These are simulator metrics, not device calibrations or energy errors.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError

    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if not np.isfinite(error_probability) or not 0 <= error_probability <= 1:
        raise ValueError("error_probability must lie in [0, 1].")
    bound = result.optimal_circuit.assign_parameters(result.optimal_parameters)
    if bound.num_qubits != 2 or bound.num_parameters:
        raise ValueError("Expected a fully bound two-qubit VQE circuit.")
    ideal = Statevector.from_instruction(bound).probabilities_dict()
    noise = NoiseModel()
    p = error_probability
    noise.add_all_qubit_readout_error(ReadoutError([[1-p, p], [p, 1-p]]))
    backend = AerSimulator(noise_model=noise, max_parallel_threads=1)
    measured = bound.copy()
    measured.measure_all()
    circuits = [measured]
    for state in range(4):
        calibration = QuantumCircuit(2)
        for qubit in range(2):
            if state & (1 << qubit):
                calibration.x(qubit)
        calibration.measure_all()
        circuits.append(calibration)
    compiled = transpile(circuits, backend, seed_transpiler=seed, optimization_level=1)
    sampled = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    counts = sampled.get_counts(0)
    labels = [format(state, "02b") for state in range(4)]
    assignment_errors = {
        label: 1 - sampled.get_counts(index + 1).get(label, 0) / shots
        for index, label in enumerate(labels)
    }
    return {
        "shots": shots, "seed": seed, "readout_flip_probability": p,
        "optimal_parameters": {str(k): float(v) for k, v in result.optimal_parameters.items()},
        "transpiled_depth": compiled[0].depth(), "counts": counts,
        "assignment_errors": assignment_errors,
        "total_variation_distance": float(0.5 * sum(
            abs(counts.get(label, 0) / shots - ideal.get(label, 0)) for label in labels
        )),
    }
