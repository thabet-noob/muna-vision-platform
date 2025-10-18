import time, threading
from orchestration.flows.train_flow import train_flow
from orchestration.flows.inference_flow import inference_flow
from orchestration.flows.monitor_flow import monitor_flow

def every(seconds, fn):
    def loop():
        while True:
            try:
                fn()
            except Exception as e:
                print("[orchestrator] error:", e)
            time.sleep(seconds)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    print("[orchestrator] starting local runners...")
    # daily train (every 6 hours here for demo)
    every(6 * 3600, train_flow)
    # inference sample every minute
    every(60, inference_flow)
    # monitoring every 2 minutes
    every(120, monitor_flow)
    # keep alive
    while True:
        time.sleep(3600)
