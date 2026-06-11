from qwen3_rlvr.logging.resource_monitor import ResourceMonitor, sample_resources


def test_sample_resources_returns_snapshot():
    snap = sample_resources(elapsed_s=1.0, label="test")
    assert snap.elapsed_s == 1.0
    assert snap.label == "test"
    assert snap.ram_total_gb > 0


def test_resource_monitor_summary(tmp_path):
    out = tmp_path / "resources.json"
    monitor = ResourceMonitor(out, interval_s=0.01)
    monitor._start_time = 0.0
    monitor.record()
    monitor.record()
    summary = monitor.save()
    assert summary.num_samples == 2
    assert out.exists()
