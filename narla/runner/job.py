import narla
import subprocess


class Job:
    """
    A Job is executed by the Runner

    :param settings: Settings for an individual MultiAgentNetwork
    :param process: Process the Job is running on
    """
    def __init__(self, settings: narla.Settings, process: subprocess.Popen):
        self._settings = settings
        self._process: subprocess.Popen = process

    def is_done(self) -> bool:
        """
        If ``True`` the Job has completed
        """
        is_done = self._process.poll() is not None

        return is_done
