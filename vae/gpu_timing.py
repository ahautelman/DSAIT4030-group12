from collections import deque

import torch


class GPUTimer:
    def __init__(self, window: int = 500, device=None):
        self.window = window
        self.cuda = torch.cuda.is_available()
        self.device = device if device is not None else (
            torch.cuda.current_device() if self.cuda else None
        )

        self._pending = deque()        # (start_event, end_event) not yet read
        self._completed_ms = deque(maxlen=window)
        self._current_start = None

        self._cpu_start = None

    def __enter__(self):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._current_start = start
        else:
            import time
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((self._current_start, end))
            self._current_start = None
            self._drain_completed()
        else:
            import time
            self._completed_ms.append((time.perf_counter() - self._cpu_start) * 1000.0)
        return False

    def _drain_completed(self):
        while self._pending:
            start, end = self._pending[0]
            if end.query():
                self._completed_ms.append(start.elapsed_time(end))
                self._pending.popleft()
            else:
                break

    def _flush(self):
        if not self._pending:
            return
        self._pending[-1][1].synchronize()
        while self._pending:
            start, end = self._pending.popleft()
            self._completed_ms.append(start.elapsed_time(end))

    def mean(self) -> float:
        self._flush()
        if not self._completed_ms:
            return 0.0
        return sum(self._completed_ms) / len(self._completed_ms)

    def reset(self):
        self._pending.clear()
        self._completed_ms.clear()
