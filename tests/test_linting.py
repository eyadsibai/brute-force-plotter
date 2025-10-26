"""
Tests for code quality and linting checks.

This module ensures that all code in the repository passes ruff linting
and formatting checks.
"""

from pathlib import Path
import subprocess
import sys

import pytest


class TestRuffLinting:
    """Tests for ruff linting checks."""

    @pytest.mark.unit
    def test_ruff_check_passes(self):
        """Test that ruff check passes on all code."""
        # Get the repository root (parent of tests directory)
        repo_root = Path(__file__).parent.parent

        # Run ruff check
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Assert that ruff check passed
        assert result.returncode == 0, (
            f"Ruff linting failed with errors:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    @pytest.mark.unit
    def test_ruff_format_check_passes(self):
        """Test that ruff format check passes on all code."""
        # Get the repository root (parent of tests directory)
        repo_root = Path(__file__).parent.parent

        # Run ruff format check
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Assert that ruff format check passed
        assert result.returncode == 0, (
            f"Ruff format check failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
            f"Some files need formatting. Run 'ruff format .' to fix."
        )
