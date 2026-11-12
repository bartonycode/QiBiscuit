"""Report the scientific Python dependencies used by this project."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import sys


_REQUIRED_DISTRIBUTIONS = (
    "qiskit",
    "qiskit-aer",
    "qiskit-nature",
    "qiskit-algorithms",
    "pyscf",
    "scipy",
    "matplotlib",
    "numpy",
)


def check_environment() -> dict[str, str]:
    """Return installed dependency versions or raise an actionable error."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")

    installed: dict[str, str] = {}
    missing: list[str] = []
    for distribution in _REQUIRED_DISTRIBUTIONS:
        try:
            installed[distribution] = version(distribution)
        except PackageNotFoundError:
            missing.append(distribution)

    if missing:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install them with `python -m pip install -r requirements.txt`."
        )
    return installed


if __name__ == "__main__":
    for package, package_version in check_environment().items():
        print(f"{package}=={package_version}")
