"""Build a two-qubit active-space Hamiltonian for CHBrClF."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit


def build_chbrclf_qubit_hamiltonian(
) -> tuple[
    ElectronicStructureProblem,
    ElectronicStructureProblem,
    ParityMapper,
    SparsePauliOp,
]:
    """Build and return the full and reduced CHBrClF electronic problems.

    The geometry is a tetrahedral Cartesian model in angstroms with distinct
    C-H, C-Br, C-Cl, and C-F bond lengths. The active-space transformer retains
    two electrons in two spatial orbitals and freezes the remaining orbitals.
    """
    atom = "\n".join(
        (
            "C  0.000000  0.000000  0.000000",
            "H  0.629312  0.629312  0.629312",
            "Br -1.120207 -1.120207  1.120207",
            "Cl -1.021911  1.021911 -1.021911",
            "F  0.779423 -0.779423 -0.779423",
        )
    )
    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    electronic_problem = driver.run()

    transformer = ActiveSpaceTransformer(
        num_electrons=2,
        num_spatial_orbitals=2,
    )
    reduced_problem = transformer.transform(electronic_problem)
    mapper = ParityMapper(num_particles=reduced_problem.num_particles)
    qubit_hamiltonian = mapper.map(reduced_problem.hamiltonian.second_q_op())

    if qubit_hamiltonian.num_qubits != 2:
        raise RuntimeError(
            "Expected the active-space ParityMapper to produce a 2-qubit "
            f"Hamiltonian; got {qubit_hamiltonian.num_qubits} qubits."
        )

    return electronic_problem, reduced_problem, mapper, qubit_hamiltonian
