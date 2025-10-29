"""Simple background video stream reader with thread-safe last-frame access."""

from __future__ import annotations

import cv2, time, threading
from typing import Optional

class VideoStream:
    """Continuously read frames from a video source in a background thread.

    Access the latest frame with read(), which returns a copy to avoid races.
    """

    def __init__(self, url: str):
        self.url: str = url
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame = None
        self.lock = threading.Lock()
        self.running: bool = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VideoStream":
        if self.running:
            return self
        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="VideoStream", daemon=True)
        self._thread.start()
        return self

    def _open(self) -> None:
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        self.cap = cv2.VideoCapture(self.url)

    def _loop(self) -> None:
        self._open()
        fails = 0
        while self.running and not self._stop.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.4)
                self._open()
                continue
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 40:
                    self._open(); fails = 0
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            # Avoid hanging on shutdown
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass

    # Context manager helpers for safer lifecycle handling
    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
