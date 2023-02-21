import narla
import subprocess


class Job:
    def __init__(self, settings: narla.Settings, process: subprocess.Popen):
        self._settings = settings
        self._process: subprocess.Popen = process

    def is_done(self) -> bool:
        is_done = self._process.poll() is not None

        return is_done
