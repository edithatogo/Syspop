# Jules's Guide to the Syspop Repository

This document summarizes the recent changes and provides a quick guide for working with your `Syspop` fork.

## Recent Changes & Setup

I have performed the following actions to set up your `Syspop` fork:

1.  **Forked Repository:** Your `Syspop` repository (`https://github.com/edithatogo/Syspop`) is now correctly configured as the `origin` remote for your local `syspop` submodule.
2.  **Ruff Linting:** A GitHub Actions workflow (`.github/workflows/ruff.yml`) has been added to automatically run `Ruff` linting on pushes and pull requests to your `main` branch. This ensures code style and quality.
3.  **MyPy Type Checking:** A GitHub Actions workflow (`.github/workflows/mypy.yml`) has been added to automatically run `MyPy` type checking on pushes and pull requests to your `main` branch. This helps ensure type correctness.
4.  **Pull Request to Upstream:** A pull request (`https://github.com/jzanetti/Syspop/pull/23`) has been created from your `ruff-fixes-pr-2` branch to the original `jzanetti/Syspop` repository, proposing the Ruff formatting and linting fixes.

## Working with Your Syspop Fork

### Keeping Your Fork Synced

To keep your `main` branch in sync with the original `Syspop` repository (`upstream/main`):

```bash
git checkout main
git pull upstream main
git push origin main
```

### Making New Changes

When making new changes you intend to contribute back to the original `Syspop` repository:

1.  **Create a new branch:**
    ```bash
    git checkout main
    git pull upstream main # Ensure your main is up-to-date
    git checkout -b my-new-feature-branch
    ```
2.  **Make your changes and commit them.**
3.  **Push your branch to your fork:**
    ```bash
    git push -u origin my-new-feature-branch
    ```
4.  **Create a Pull Request:** Use the GitHub UI or `gh cli` to create a pull request from `edithatogo/Syspop:my-new-feature-branch` to `jzanetti/Syspop:main`.

### Running Linters and Type Checkers Locally

Before pushing, you can run Ruff and MyPy locally to catch issues early:

```bash
# From the syspop directory
pip install ruff mypy
ruff check . --fix
ruff format .
mypy syspop/python/ syspop/etc/scripts_nz/
```

This setup should streamline your development workflow for the `Syspop` library. Feel free to ask if you have more questions!
