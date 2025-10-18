import cv2, time, threading

class VideoStream:
    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        if self.running:
            return self
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _open(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        self.cap = cv2.VideoCapture(self.url)

    def _loop(self):
        self._open()
        fails = 0
        while self.running:
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

    def stop(self):
        self.running = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
