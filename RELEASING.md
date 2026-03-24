# Releasing

This repo is prepared for a GitHub plus PyPI flow with:

- a GitHub CI workflow in `.github/workflows/ci.yml`
- a manual TestPyPI workflow in `.github/workflows/publish-testpypi.yml`
- a tag-driven PyPI workflow in `.github/workflows/publish.yml`

The distribution name is `paithon-jig`. The import name remains `paithon`.

## First-Time Setup

1. Create the GitHub repository.
2. Initialize Git locally if needed:

```bash
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

3. In GitHub, enable Actions for the repo.
4. In TestPyPI and PyPI, configure trusted publishing for this repository and the matching workflow file.
5. After the real GitHub URL exists, add it to `pyproject.toml` under `[project.urls]`.

## Local Release Checks

Run these before publishing:

```bash
pytest -q
cd /tmp
python -m build /path/to/PAIthon
```

If `twine` is installed locally, also run:

```bash
python -m twine check dist/*
```

If you are building from a restricted or offline shell, use:

```bash
cd /tmp
python -m build --no-isolation /path/to/PAIthon
```

## TestPyPI

Use the `Publish TestPyPI` workflow from the GitHub Actions UI.

After a successful publish, verify installation:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple paithon-jig
```

## PyPI

1. Update `version` in `pyproject.toml`.
2. Commit the version change.
3. Tag the release:

```bash
git tag v0.1.0
git push origin main --tags
```

4. The `Publish PyPI` workflow will build and publish the release.

## Notes

- Trusted publishing is the intended path. These workflows do not require storing a long-lived PyPI token in GitHub secrets.
- The default workflows publish only from GitHub; they do not publish from a local machine.
- `paithon-jig` is the package users install. `import paithon` remains the runtime import surface.
