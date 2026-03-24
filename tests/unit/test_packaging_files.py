from pathlib import Path

from tests.support.files import PROJECT_ROOT


def test_pyproject_contains_publish_metadata():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "paithon-jig"' in text
    assert 'license = {file = "LICENSE"}' in text
    assert 'build-backend = "setuptools.build_meta"' in text


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
