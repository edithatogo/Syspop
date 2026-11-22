# Contributing to Syspop

Syspop develops synthetic and simulated population workflows in Python and R. Start with the [README](README.md), [pixi.toml](pixi.toml), and the existing task definitions before changing an analysis or environment.

## Before you start

Search existing issues and pull requests before starting, and discuss substantial changes or new dependencies in an issue. Use an issue template when one is offered. For a bug, include the affected commit or version, environment, a minimal reproduction, and expected versus observed behavior; remove sensitive information from logs and screenshots.

## Development and validation

The committed Pixi tasks provide these checks:

```sh
pixi run lint
pixi run test
```

The task named `test` runs static type checking with mypy; it does not establish simulation correctness. The lint task combines Ruff, Bandit, and Vulture. Record any findings and environment limitations. Inspect the [makefile](makefile) before using its environment targets: some clear environments or package caches. Do not run publication or upload targets as part of routine validation.

For synthesis changes, use small synthetic inputs, fixed seeds where applicable, and explicit expected aggregate constraints. Compare output schemas and population totals, and report numerical tolerances. Keep R dependency changes consistent with the existing R environment files and explain any changes to population assumptions.

## Pull requests

Work on a focused branch from the current default branch. Open a pull request explaining the problem, the resulting behavior, and any compatibility or data-format changes. Link the relevant issue and complete any existing PR template.

Add focused regression coverage for changed behavior and update affected examples or documentation. Describe the checks actually run, their results, and any checks you could not run. For documentation-only changes, verify links, examples, and rendered formatting. Use the repository's existing checks and review/merge process; preserve required checks and resolve actionable feedback without weakening the checks.

## Security and sensitive information

Follow the [account security policy](https://github.com/edithatogo/.github/blob/main/SECURITY.md). Report vulnerabilities privately through the repository's **Report a vulnerability** option if available; otherwise request a private reporting contact without posting exploit details. Never include credentials, personal information, restricted datasets, or confidential records in issues, commits, fixtures, or diagnostic output.

## Licensing and conduct

Be respectful and constructive, and follow the [account code of conduct](https://github.com/edithatogo/.github/blob/main/CODE_OF_CONDUCT.md). Preserve attribution and third-party notices. Contribute only material you are entitled to share; resolve unclear permissions before adding third-party content. This guide does not change licensing terms or introduce a CLA or DCO requirement.

## References

- [GitHub contribution guideline conventions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors)
- [OpenSSF contribution and quality criteria](https://www.bestpractices.dev/en/criteria)
