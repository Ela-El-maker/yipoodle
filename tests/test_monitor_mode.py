import json
from pathlib import Path

from src.apps.monitor_mode import run_monitor_mode, unregister_monitor


def test_monitor_mode_creates_spec_and_runs_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def _fake_load(_path: str) -> dict:
        return {"topics": [], "paths": {}}

    def _fake_run(_cfg: str) -> str:
        out = tmp_path / "runs" / "monitor" / "audit" / "sample" / "runs" / "20260224T150000Z"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    monkeypatch.setattr("src.apps.monitor_mode.load_automation_config", _fake_load)
    monkeypatch.setattr("src.apps.monitor_mode.run_automation", _fake_run)

    out_path = tmp_path / "runs" / "monitor" / "nvidia.json"
    payload = run_monitor_mode(
        question="Monitor NVIDIA stock and notify me",
        schedule="0 */6 * * *",
        automation_config_path="config/automation.yaml",
        out_path=str(out_path),
        register_schedule=False,
    )

    assert payload["mode"] == "monitor"
    assert payload["monitor_bootstrap_ok"] is True
    assert payload["baseline_run_id"] == "20260224T150000Z"
    assert Path(payload["monitor_spec_path"]).exists()
    assert Path(payload["generated_automation_config"]).exists()

    topic_spec = json.loads(Path(payload["monitor_spec_path"]).read_text(encoding="utf-8"))
    assert topic_spec["schedule"] == "0 */6 * * *"
    assert "nvidia stock" in topic_spec["query"].lower()


def test_monitor_mode_baseline_failure_is_non_fatal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("src.apps.monitor_mode.load_automation_config", lambda _path: {"topics": [], "paths": {}})

    def _boom(_cfg: str) -> str:
        raise RuntimeError("baseline failed")

    monkeypatch.setattr("src.apps.monitor_mode.run_automation", _boom)

    payload = run_monitor_mode(
        question="Track PIX outages",
        schedule="0 */6 * * *",
        automation_config_path="config/automation.yaml",
        out_path=str(tmp_path / "runs" / "monitor" / "pix.json"),
        register_schedule=False,
    )

    assert payload["monitor_bootstrap_ok"] is False
    assert payload["baseline_run_id"] is None
    assert "baseline failed" in str(payload["monitor_bootstrap_error"])


def test_monitor_mode_registers_schedule_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.apps.monitor_mode.load_automation_config", lambda _path: {"topics": [], "paths": {}})
    monkeypatch.setattr("src.apps.monitor_mode.run_automation", lambda _cfg: str(tmp_path / "runs" / "x" / "20260224T150000Z"))

    captured: dict[str, str] = {}

    def _fake_upsert(*, name: str, schedule: str, generated_config_path: str) -> str:
        captured["name"] = name
        captured["schedule"] = schedule
        captured["generated_config_path"] = generated_config_path
        return f"{schedule} echo ok # yipoodle-monitor:{name}"

    monkeypatch.setattr("src.apps.monitor_mode._upsert_monitor_crontab", _fake_upsert)

    payload = run_monitor_mode(
        question="Monitor NVIDIA stock",
        schedule="0 */6 * * *",
        automation_config_path="config/automation.yaml",
        out_path=str(tmp_path / "runs" / "monitor" / "nvidia_sched.json"),
        register_schedule=True,
    )

    assert payload["schedule_register_requested"] is True
    assert payload["schedule_registered"] is True
    assert payload["schedule_error"] is None
    assert "yipoodle-monitor" in str(payload["schedule_entry"])
    assert captured["schedule"] == "0 */6 * * *"


def test_monitor_mode_schedule_registration_fail_open(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.apps.monitor_mode.load_automation_config", lambda _path: {"topics": [], "paths": {}})
    monkeypatch.setattr("src.apps.monitor_mode.run_automation", lambda _cfg: str(tmp_path / "runs" / "x" / "20260224T150000Z"))
    monkeypatch.setattr(
        "src.apps.monitor_mode._upsert_monitor_crontab",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("crontab unavailable")),
    )

    payload = run_monitor_mode(
        question="Track PIX outages",
        schedule="0 */6 * * *",
        automation_config_path="config/automation.yaml",
        out_path=str(tmp_path / "runs" / "monitor" / "pix_sched.json"),
        register_schedule=True,
    )

    assert payload["schedule_register_requested"] is True
    assert payload["schedule_registered"] is True
    assert payload["schedule_backend_used"] == "file"
    assert payload["schedule_entry"] is not None
    assert "crontab unavailable" in str(payload["schedule_error"])
    assert payload["monitor_bootstrap_ok"] is True


def test_monitor_mode_schedule_backend_crontab_no_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.apps.monitor_mode.load_automation_config", lambda _path: {"topics": [], "paths": {}})
    monkeypatch.setattr("src.apps.monitor_mode.run_automation", lambda _cfg: str(tmp_path / "runs" / "x" / "20260224T150000Z"))
    monkeypatch.setattr(
        "src.apps.monitor_mode._upsert_monitor_crontab",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("crontab unavailable")),
    )

    payload = run_monitor_mode(
        question="Track PIX outages",
        schedule="0 */6 * * *",
        automation_config_path="config/automation.yaml",
        out_path=str(tmp_path / "runs" / "monitor" / "pix_sched_crontab.json"),
        register_schedule=True,
        schedule_backend="crontab",
    )

    assert payload["schedule_register_requested"] is True
    assert payload["schedule_registered"] is False
    assert payload["schedule_backend_used"] is None
    assert payload["schedule_entry"] is None
    assert "crontab unavailable" in str(payload["schedule_error"])
    assert payload["monitor_bootstrap_ok"] is True


def test_monitor_mode_schedule_backend_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.apps.monitor_mode.load_automation_config", lambda _path: {"topics": [], "paths": {}})
    monkeypatch.setattr("src.apps.monitor_mode.run_automation", lambda _cfg: str(tmp_path / "runs" / "x" / "20260224T150000Z"))
    monkeypatch.setattr(
        "src.apps.monitor_mode._upsert_monitor_crontab",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("crontab should not be called")),
    )

    payload = run_monitor_mode(
        question="Track PIX outages",
        schedule="0 */6 * * *",
        automation_config_path="config/automation.yaml",
        out_path=str(tmp_path / "runs" / "monitor" / "pix_sched_file.json"),
        register_schedule=True,
        schedule_backend="file",
    )

    assert payload["schedule_register_requested"] is True
    assert payload["schedule_registered"] is True
    assert payload["schedule_backend_used"] == "file"
    assert payload["schedule_error"] is None
    assert payload["schedule_entry"] is not None
    assert Path(str(payload["schedule_entry"])).exists()
    assert payload["monitor_bootstrap_ok"] is True


def test_monitor_unregister_removes_files_and_schedule(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = tmp_path / "runs" / "monitor" / "topics" / "nvidia_stock.json"
    gen = tmp_path / "runs" / "monitor" / "generated" / "nvidia_stock.automation.yaml"
    sched = tmp_path / "runs" / "monitor" / "schedules" / "nvidia_stock.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    gen.parent.mkdir(parents=True, exist_ok=True)
    sched.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("{}", encoding="utf-8")
    gen.write_text("{}", encoding="utf-8")
    sched.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.apps.monitor_mode._remove_monitor_crontab", lambda **_kwargs: True)

    payload = unregister_monitor(name_or_question="nvidia stock", delete_files=True)

    assert payload["name"] == "nvidia_stock"
    assert payload["schedule_removed"] is True
    assert payload["schedule_error"] is None
    assert payload["spec_removed"] is True
    assert payload["generated_config_removed"] is True
    assert payload["schedule_registry_removed"] is True
    assert not spec.exists()
    assert not gen.exists()
    assert not sched.exists()


def test_monitor_unregister_schedule_fail_open(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.apps.monitor_mode._remove_monitor_crontab",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("crontab permission denied")),
    )
    payload = unregister_monitor(name_or_question="pix outage monitor", delete_files=False)
    assert payload["name"] == "pix_outage_monitor"
    assert payload["schedule_removed"] is False
    assert "permission denied" in str(payload["schedule_error"])
