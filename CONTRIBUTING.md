# Contributing to Brute Force Plotter

Thank you for your interest in contributing to Brute Force Plotter! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality Tools](#code-quality-tools)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install UV
pip install uv

# Clone the repository
git clone https://github.com/eyadsibai/brute-force-plotter.git
cd brute-force-plotter

# Install dependencies (including dev dependencies)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### Using pip (Traditional)

```bash
# Clone the repository
git clone https://github.com/eyadsibai/brute-force-plotter.git
cd brute-force-plotter

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov ruff pre-commit
```

## Code Quality Tools

This project uses several tools to maintain code quality:

### Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and formatter written in Rust. It replaces multiple tools (flake8, isort, pyupgrade, etc.) with a single, fast tool.

**Linting:**
```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

**Formatting:**
```bash
# Check formatting
ruff format --check .

# Auto-format code
ruff format .
```

### Bandit (Security Checks)

[Bandit](https://github.com/PyCQA/bandit) is a tool for finding common security issues in Python code. While configured in `pyproject.toml`, it runs separately from pre-commit hooks.

**Usage:**
```bash
# Install bandit
pip install bandit

# Run security checks
bandit -c pyproject.toml -r src/

# Run on specific file
bandit -c pyproject.toml src/brute_force_plotter.py
```

### Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit, ensuring code quality standards are met.

**Setup:**
```bash
# Install pre-commit hooks
pre-commit install
```

**Usage:**
```bash
# Hooks run automatically on git commit
git commit -m "Your message"

# Run manually on all files
pre-commit run --all-files

# Run manually on staged files
pre-commit run

# Update hooks to latest versions
pre-commit autoupdate
```

The pre-commit configuration includes:
- Ruff linting and formatting
- Trailing whitespace removal
- End-of-file fixer
- YAML, JSON, and TOML validation
- Large file checks
- Merge conflict detection

**Note:** Bandit security checks are configured but not included in pre-commit hooks. Run Bandit separately as shown above.

**Troubleshooting:** If you experience network timeouts during the first `git commit` after installing pre-commit hooks, you can:
- Skip the hooks for that commit: `git commit --no-verify -m "Your message"`
- Or manually run the checks before committing: `ruff check --fix . && ruff format .`

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/description` - for new features
- `fix/description` - for bug fixes
- `docs/description` - for documentation changes
- `refactor/description` - for code refactoring

### Commit Messages

Write clear, descriptive commit messages:
- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Add a blank line before the detailed description (if needed)
- Reference issues and pull requests when relevant

Example:
```
Add correlation matrix visualization

- Implement Pearson correlation heatmap
- Implement Spearman correlation heatmap
- Add configuration option for correlation method

Fixes #123
```

## Testing

Currently, the project has minimal test coverage. When adding new features:

1. Consider adding tests if you're implementing a new feature
2. Ensure your changes don't break existing functionality
3. Test with the example dataset:

```bash
# Using UV
uv run python -m src example/titanic.csv example/titanic_dtypes.json example/output

# Using pip
python3 -m src example/titanic.csv example/titanic_dtypes.json example/output
```

### Running Tests

```bash
# Using UV
uv run pytest

# Using pip
pytest
```

## Submitting Changes

1. **Update your branch** with the latest changes from main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run code quality checks:**
   ```bash
   # Lint and format
   ruff check --fix .
   ruff format .
   
   # Or use pre-commit
   pre-commit run --all-files
   ```

3. **Test your changes:**
   ```bash
   # Run with example data
   python3 -m src example/titanic.csv example/titanic_dtypes.json example/output
   ```

4. **Push your changes:**
   ```bash
   git push origin your-branch
   ```

5. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots (if applicable)
   - Test results

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (enforced by ruff)
- Maximum line length: 88 characters (Black-compatible)
- Use double quotes for strings
- Use type hints where appropriate

### Code Organization

- Keep functions focused on a single responsibility
- Use descriptive variable and function names
- Add docstrings for modules, classes, and functions
- Organize imports:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports

### Documentation

- Update README.md if adding new features or changing behavior
- Add docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Comment complex logic or non-obvious code

### Example Code Style

```python
import os
from pathlib import Path

import pandas as pd
import seaborn as sns


def plot_distribution(data: pd.DataFrame, column: str, output_path: Path) -> None:
    """
    Plot distribution for a numeric column.

    Args:
        data: DataFrame containing the data
        column: Name of the column to plot
        output_path: Directory where the plot will be saved
    """
    # Implementation here
    pass
```

## Questions?

If you have questions or need help:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the README.md for usage examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Brute Force Plotter! 🎉
