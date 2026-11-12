"""Kiểm thử các tiện ích theo dõi run, không cần đăng nhập W&B."""
import random
import sys
import types

import numpy as np
import pytest
import torch

from randumb.tracking import (
    Timer,
    environment_info,
    init_run,
    peak_ram_gb,
    relative_deviation,
    set_seed,
)


def test_set_seed_repeats_all_generators():
    set_seed(3)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(3)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first


def test_relative_deviation():
    assert relative_deviation(55.0, 50.0) == pytest.approx(10.0)
    assert relative_deviation(45.0, 50.0) == pytest.approx(-10.0)
    with pytest.raises(ValueError):
        relative_deviation(1.0, 0.0)


def test_environment_info_has_versions():
    info = environment_info()
    assert info["version_torch"] is not None
    assert "python" in info and "gpu_name" in info


def test_timer_and_peak_ram():
    with Timer() as timer:
        sum(range(1000))
    assert timer.seconds >= 0
    ram = peak_ram_gb()
    assert ram is None or ram > 0


def test_init_run_passes_config_and_disabled_mode(monkeypatch):
    captured = {}
    fake_wandb = types.SimpleNamespace(init=lambda **kwargs: captured.update(kwargs) or "run")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_ENTITY", "team-x")

    run = init_run("E1-test", "E1", ["lab:reproduction"], {"dim": 10}, enabled=False)

    assert run == "run"
    assert captured["entity"] == "team-x" and captured["mode"] == "disabled"
    assert captured["config"]["dim"] == 10 and "python" in captured["config"]
    assert captured["tags"] == ["lab:reproduction"]


def test_init_run_defaults_to_team_project(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=lambda **kw: captured.update(kw)))
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    init_run("E1-test", "E1", [], {}, enabled=False)

    assert captured["entity"] == "24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm"
    assert captured["project"] == "randumb-reproduction-24TNT"
