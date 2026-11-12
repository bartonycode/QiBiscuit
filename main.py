"""Run the CHBrClF active-space, VQE, and effective pulse-steering pipeline."""

from __future__ import annotations

import numpy as np
import json
from pathlib import Path

from environment import check_environment
from molecule_driver import build_chbrclf_qubit_hamiltonian
from pulse_steering import optimize_laser_parameters
from vqe_solver import run_vqe_simulation, measure_vqe_readout


def main() -> None:
    """Run the molecular calculation and save the pulse population plot."""
    for package, package_version in check_environment().items():
        print(f"{package}=={package_version}")

    _, reduced_problem, mapper, qubit_hamiltonian = (
        build_chbrclf_qubit_hamiltonian()
    )
    vqe_result, exact_result = run_vqe_simulation(reduced_problem, mapper)
    readout = measure_vqe_readout(vqe_result)

    # Treat the gap between the two lowest electronic levels as an effective
    # R/S tunnelling splitting; the molecular operator does not encode chirality.
    electronic_levels = np.linalg.eigvalsh(qubit_hamiltonian.to_matrix())
    if electronic_levels.size < 2:
        raise RuntimeError("At least two electronic levels are needed for steering.")
    lower, upper = electronic_levels[:2]
    level_gap = float(upper - lower)
    mean_energy = float((upper + lower) / 2.0)
    effective_h0 = np.array(
        [[mean_energy, level_gap / 2.0], [level_gap / 2.0, mean_energy]],
        dtype=np.complex128,
    )
    pulse = optimize_laser_parameters(effective_h0)
    report = {
        "energy_difference_Ha": float(abs(vqe_result.eigenvalue - exact_result.eigenvalue)),
        "pulse_effective_model": pulse,
        "synthetic_readout": readout,
    }
    Path("validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
