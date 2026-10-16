"""Muestreo temporal de la memoria RSS del proceso actual."""

import csv
import threading
import time
from pathlib import Path


def read_rss_mb() -> float:
    """Devuelve la memoria residente del proceso en MiB en Linux."""
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("No se encontró VmRSS en /proc/self/status")


class MemorySampler:
    """Muestrea RSS periódicamente y puede guardar una serie temporal CSV."""

    def __init__(self, interval_seconds: float = 0.02) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds debe ser mayor que 0")
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemorySampler":
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.samples.append((
                time.perf_counter() - self._started,
                read_rss_mb(),
            ))
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            raise RuntimeError("El sampler no recogió ninguna muestra")
        times = [sample[0] for sample in self.samples]
        values = [sample[1] for sample in self.samples]
        area = sum(
            (right_time - left_time) * (left_value + right_value) / 2
            for (left_time, left_value), (right_time, right_value)
            in zip(self.samples, self.samples[1:])
        )
        peak_index = max(range(len(values)), key=values.__getitem__)
        return {
            "memory_samples": len(values),
            "memory_mean_mb": sum(values) / len(values),
            "memory_p95_mb": sorted(values)[int(0.95 * (len(values) - 1))],
            "memory_peak_mb": values[peak_index],
            "memory_time_to_peak_s": times[peak_index],
            "memory_area_mb_s": area,
            "memory_duration_s": times[-1],
        }

    def write_csv(self, path: Path) -> None:
        """Guarda las muestras para dibujar la evolución posteriormente."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["seconds", "rss_mb"])
            writer.writerows(self.samples)
