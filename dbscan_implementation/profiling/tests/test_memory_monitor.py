from pathlib import Path

from dbscan_implementation.profiling.memory_monitor import MemorySampler


def test_memory_sampler_collects_and_writes_trace(tmp_path: Path) -> None:
    sampler = MemorySampler(interval_seconds=0.001)
    with sampler:
        sum(range(100_000))

    summary = sampler.summary()
    assert summary["memory_samples"] >= 1
    assert summary["memory_peak_mb"] >= 0

    output = tmp_path / "memory.csv"
    sampler.write_csv(output)
    assert output.read_text(encoding="utf-8").startswith("seconds,rss_mb\n")
