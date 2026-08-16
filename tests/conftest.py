"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def tmp_runs(tmp_path):
    return tmp_path / "runs"
