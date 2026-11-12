# Friday presentation notes

## Slide 5 — Code Implementation

Previously reported VQE versus exact reduced-operator energy difference:
4.181 × 10⁻⁹ Ha. This demonstrates numerical solver agreement on the same
two-qubit active-space Hamiltonian. It does not independently establish basis
set accuracy, active-space accuracy, or the physical validity of the chirality
model. The active-space calculation uses two electrons in two spatial orbitals.

Use `validation_results.json` for the latest rerun difference and synthetic
readout metrics. The optimal VQE parameters are bound to the ansatz, transpiled
for Aer, and sampled with independent 2% symmetric readout flips per qubit.
Four computational basis calibration circuits estimate assignment error;
total variation distance compares sampled VQE counts with ideal probabilities.
These are synthetic readout metrics, not IBM hardware calibration results.

## Slide 6 — Pulse Control Dynamics

Embed `chbrclf_stereocontrol.png`. The 225-candidate grid reached 99.9314%
final signed R excess (P_R ≈ 99.9657%, P_S ≈ 0.0343%) at t=64 model units.
Parameters: E0=0.29840813, sigma=4, omega=0.28324903, t0=32; hbar=1.
See `pulse_results.json` for the standalone sweep and `validation_results.json`
for the integrated rerun. Small run-to-run electronic differences are possible.

The initial state is the coherent superposition (|R>+|S>)/sqrt(2), not an
incoherent racemate. A maximally mixed racemate remains unchanged by unitary
two-state evolution. The electronic gap is used as a phenomenological tunneling
splitting; it has not been calculated from a molecular inversion barrier.
Field-free tunneling resumes after the pulse, so the result is readout-time
localization, not persistent control or demonstrated coherent destruction of
tunneling.

For H=H0−mu E(t) sigma_z, the energy gap sets the candidate carrier frequency.
The rotating-wave envelope area mu E0 sigma sqrt(2 pi) supplies a search scale
in model units. It is not the integral of the oscillatory field, which includes
a Gaussian carrier suppression factor. An energy-eigenbasis pi pulse alone
does not guarantee localized R/S transfer from the specified initial state.

No slide deck was present in this workspace; these notes provide insertion text.

## Run

Install `requirements.txt` into your Python environment and run `python main.py`.
The pipeline writes the optimized plot and `validation_results.json`.
Aer noise API reference: https://qiskit.github.io/qiskit-aer/apidocs/aer_noise
