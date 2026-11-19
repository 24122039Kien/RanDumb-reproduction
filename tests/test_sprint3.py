"""Kiểm tra lịch run, calibration không dùng test và chống resume kết quả sai."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def suite():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sprint3.py"
    spec = importlib.util.spec_from_file_location("run_sprint3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_for(suite, job):
    return dict(name=job["name"], config=suite.expected_config(job, "test-source", "cuda", 512),
                final_accuracy=40.0, mean_variance=2e-5, lambda_effective=1e-5,
                wandb_url="https://wandb.ai/test/project/runs/test-run")


def test_complete_manifest_and_no_implicit_parameter_changes(suite, tmp_path):
    jobs = suite.make_jobs()
    assert len(jobs) == len({j["name"] for j in jobs}) == 71
    e1 = [j for j in jobs if j["group"] == "E1"]
    assert len(e1) == 15
    assert {j["seed"] for j in e1} == {0, 1, 2}
    assert {j["dim"] for j in e1} == {25000}
    for job in e1:
        cmd = suite.build_command(job, tmp_path / "run.json", "cuda", 512)
        assert cmd[cmd.index("--gamma") + 1] == "1.0"
        assert cmd[cmd.index("--norm") + 1] == "l2"
        assert "--max-train" not in cmd and "--max-eval" not in cmd
    e8 = [j for j in jobs if j["name"].startswith("E8-B")]
    assert len(e8) == 24
    assert {j["shrink"] for j in e8} == {"oas", "oas_per_class", "none"}


def test_calibration_uses_train_variance_only(suite, tmp_path):
    fixed = suite.make_jobs("E8-B")[0]
    record = record_for(suite, fixed)
    path = tmp_path / (fixed["name"] + ".json")
    path.write_text(json.dumps(record), encoding="utf-8")
    relative = next(j for j in suite.make_jobs("E8") if j["dataset"] == "cifar10"
                    and j["dim"] == 25000 and j["shrink"] == "oas" and j["lam_mode"] == "relative")
    resolved = suite.resolve_job(relative, tmp_path, "test-source", "cuda", 512)
    assert resolved["lam"] == pytest.approx(0.5)
    assert resolved["calibration_run"] == record["wandb_url"]
    record["final_accuracy"] = 99.0
    path.write_text(json.dumps(record), encoding="utf-8")
    assert suite.resolve_job(relative, tmp_path, "test-source", "cuda", 512)["lam"] == resolved["lam"]


@pytest.mark.parametrize("change", ["source", "gamma", "url", "accuracy"])
def test_resume_rejects_stale_or_incomplete_results(suite, tmp_path, change):
    job = suite.make_jobs("E1")[0]
    record = record_for(suite, job)
    if change == "source":
        record["config"]["source_sha256"] = "old-source"
    elif change == "gamma":
        record["config"]["gamma_value"] = 0.001
    elif change == "url":
        del record["wandb_url"]
    else:
        record["final_accuracy"] = None
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        suite.completed_record(path, job, "test-source", "cuda", 512)


def test_summary_marks_missing_seeds_and_retains_previous_sessions(suite, tmp_path):
    import csv

    jobs = suite.make_jobs("E1")[:3]
    for job in jobs[:2]:
        (tmp_path / (job["name"] + ".json")).write_text(json.dumps(record_for(suite, job)), encoding="utf-8")
    records = suite.collect_records(tmp_path, "test-source", "cuda", 512)
    suite.write_summary(records, tmp_path)
    with (tmp_path / "e1_aggregate.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "incomplete"
    assert row["mean_accuracy"] == ""
    last = record_for(suite, jobs[2])
    last["final_accuracy"] = 43
    (tmp_path / (jobs[2]["name"] + ".json")).write_text(json.dumps(last), encoding="utf-8")
    records = suite.collect_records(tmp_path, "test-source", "cuda", 512)
    suite.write_summary(records, tmp_path)
    with (tmp_path / "e1_aggregate.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "complete"
    assert float(row["mean_accuracy"]) == 41
    assert float(row["sample_std"]) == pytest.approx(3 ** 0.5)


def test_suite_executes_once_then_resumes_without_new_runs(suite, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(suite, "code_version", lambda: {"source_sha256": "test-source"})
    jobs = {j["name"]: j for j in suite.make_jobs("E1")}

    def fake_run(argv, **kwargs):
        name = argv[argv.index("--name") + 1]
        calls.append(name)
        path = Path(argv[argv.index("--output") + 1])
        path.write_text(json.dumps(record_for(suite, jobs[name])), encoding="utf-8")

    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    args = ["--group", "E1", "--dataset", "mnist", "--out-dir", str(tmp_path), "--execute"]
    suite.main(args)
    assert len(calls) == 3
    suite.main(args)
    assert len(calls) == 3


def test_e8a_covers_normalization_and_gamma_from_instruction(suite, tmp_path):
    """Docs/Instruction.md mục 4.6: ba cách chuẩn hóa × γ = 1 và các bội số của scale, D = 5.000 và D lớn."""
    jobs = suite.make_jobs("E8-A")
    assert len(jobs) == 32
    assert {j["dataset"] for j in jobs} == {"cifar10", "cifar100"} and {j["dim"] for j in jobs} == {5000, 25000}
    small = {(j["norm"], j["gamma"]) for j in jobs if j["dim"] == 5000 and j["dataset"] == "cifar10"}
    assert small == {(n, g) for n in ("zscore", "l2", "unit") for g in ("1.0", "scale*0.5", "scale", "scale*2")} - {("l2", "1.0")}
    large = {(j["norm"], j["gamma"]) for j in jobs if j["dim"] == 25000 and j["dataset"] == "cifar100"}
    assert ("zscore", "scale") in large and ("zscore", "1.0") in large and ("l2", "1.0") not in large
    # Tổ hợp L2 + γ = 1 đã có trong E8-B, cùng các tham số còn lại.
    assert any(j["norm"] == "l2" and j["gamma"] == "1.0" and j["shrink"] == "oas" and j["lam_mode"] == "fixed"
               for j in suite.make_jobs("E8-B"))
    job = next(j for j in jobs if j["norm"] == "zscore" and j["gamma"] == "scale*0.5")
    cmd = suite.build_command(job, tmp_path / "run.json", "cuda", 512)
    assert cmd[cmd.index("--gamma") + 1] == "scale*0.5" and cmd[cmd.index("--norm") + 1] == "zscore"
    assert cmd[cmd.index("--shrink") + 1] == "oas" and cmd[cmd.index("--lam") + 1] == "1e-05" and "--flip" in cmd
    expected = suite.expected_config(job, "src", "cuda", 512)
    assert expected["gamma"] == "scale*0.5" and "gamma_value" not in expected


def test_resolve_gamma_multiples_of_scale():
    import torch
    from randumb.pipeline import resolve_gamma

    x = torch.randn(10, 4, 4, 3)
    base = resolve_gamma("scale", x)
    assert resolve_gamma("scale*2", x) == pytest.approx(2 * base)
    assert resolve_gamma("scale*0.5", x) == pytest.approx(0.5 * base)
    assert resolve_gamma("1.0", x) == 1.0
    with pytest.raises(ValueError):
        resolve_gamma("scale*-1", x)
    with pytest.raises(ValueError):
        resolve_gamma("scalex2", x)
