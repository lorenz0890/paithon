from pathlib import Path
import subprocess
import sys

import pytest

from tests.support.files import PROJECT_ROOT


def _validation_python() -> str:
    candidates = [PROJECT_ROOT / ".venv" / "bin" / "python", Path(sys.executable)]
    script = (
        "import importlib.util, sys\n"
        "sys.exit(0 if importlib.util.find_spec('setuptools.config.pyprojecttoml') else 1)\n"
    )
    for candidate in candidates:
        if candidate.exists():
            completed = subprocess.run(
                [str(candidate), "-c", script],
                cwd="/tmp",
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return str(candidate)
    pytest.skip("no local Python interpreter provides setuptools pyproject validation support")


def test_pyproject_contains_publish_metadata():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "paithon-jig"' in text
    assert 'license = {file = "LICENSE"}' in text
    assert 'build-backend = "setuptools.build_meta"' in text
    assert "[project.urls]" in text


def test_pyproject_validates_with_setuptools():
    subprocess.run(
        [
            _validation_python(),
            "-c",
            (
                "from setuptools.config.pyprojecttoml import read_configuration;"
                "read_configuration('pyproject.toml', True, False, None)"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_release_files_exist():
    expected_paths = [
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "RELEASING.md",
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml",
        PROJECT_ROOT / ".github" / "workflows" / "publish-testpypi.yml",
        PROJECT_ROOT / ".github" / "workflows" / "publish.yml",
    ]
    for path in expected_paths:
        assert path.exists(), str(path)


def test_template_uses_publish_distribution_name():
    text = (PROJECT_ROOT / "templates" / "real_project" / "pyproject.toml").read_text(encoding="utf-8")
    assert '"paithon-jig"' in text
