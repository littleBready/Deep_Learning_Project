import os
import signal
for pid in pids:
    try:
        os.kill (pid, signal.SIGKILL)
    except:
        pass
    