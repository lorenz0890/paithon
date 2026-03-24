import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from tests.support.files import PROJECT_ROOT


def _python_supports_local_build(python_executable: str) -> bool:
    script = (
        "import importlib.util, re, setuptools, sys\n"
        "match = re.match(r'^(\\d+)', setuptools.__version__)\n"
        "major = int(match.group(1)) if match else 0\n"
        "has_build = importlib.util.find_spec('build') is not None\n"
        "sys.exit(0 if has_build and major >= 61 else 1)\n"
    )
    completed = subprocess.run(
        [python_executable, "-c", script],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _build_python() -> str:
    candidates = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), sys.executable]
    for candidate in candidates:
        if Path(candidate).exists() and _python_supports_local_build(candidate):
            return candidate
    pytest.skip("no local Python interpreter has both the build frontend and setuptools>=61 available")


def test_package_builds_with_expected_metadata(tmp_path):
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [
            _build_python(),
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist_dir),
            str(PROJECT_ROOT),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )

    wheel_paths = sorted(dist_dir.glob("*.whl"))
    sdist_paths = sorted(dist_dir.glob("*.tar.gz"))
    assert len(wheel_paths) == 1
    assert len(sdist_paths) == 1

    wheel_path = wheel_paths[0]
    sdist_path = sdist_paths[0]
    assert wheel_path.name.startswith("paithon_jig-0.1.0-")
    assert sdist_path.name == "paithon_jig-0.1.0.tar.gz"

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode("utf-8")
    assert "Name: paithon-jig" in metadata
    assert "License-File: LICENSE" in metadata

    with tarfile.open(sdist_path, "r:gz") as sdist:
        names = sdist.getnames()
    assert any(name.endswith("/README.md") for name in names)
    assert any(name.endswith("/LICENSE") for name in names)
