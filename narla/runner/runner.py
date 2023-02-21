import os
import sys
import time
import tqdm
import narla
import subprocess
from typing import Dict, List


class Runner:
    def __init__(self, all_settings: List[narla.Settings], available_gpus: List[int] = (0,), jobs_per_gpu: int = 1):
        self._all_settings = all_settings
        self._available_gpus = available_gpus
        self._jobs_per_gpu = jobs_per_gpu

        self._jobs_on_gpus: Dict[int, List[narla.runner.Job]] = {
            gpu: [] for gpu in available_gpus
        }
        self._progress_bar = tqdm.tqdm(
            total=len(all_settings),
            position=0,
            leave=True
        )

    def execute(self):
        while self._all_settings:
            self._fill_free_gpus()

            self._wait_for_free_gpu()

        self._wait_for_running_jobs()

        self._progress_bar.close()

    @staticmethod
    def _execute_job(settings: narla.Settings, gpu: int):
        trial_path = narla.io.format_trial_path(settings)
        narla.io.make_directories(trial_path)

        log_file = os.path.join(trial_path, "log.txt")

        process = subprocess.Popen(
            args=[f"{sys.executable} main.py " + settings.to_command_string()],
            shell=True,
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT
        )

        job = narla.runner.Job(
            settings=settings,
            process=process
        )
        time.sleep(5)

        return job

    def _fill_free_gpus(self):
        for gpu, running_jobs in self._jobs_on_gpus.items():
            while len(running_jobs) < self._jobs_per_gpu:
                settings = self._all_settings.pop(0)

                job = self._execute_job(
                    settings=settings,
                    gpu=gpu
                )
                running_jobs.append(job)

    def is_done(self) -> bool:
        for running_jobs in self._jobs_on_gpus.values():
            if running_jobs:
                return False

        return True

    def _remove_completed_jobs(self) -> bool:
        for running_jobs in self._jobs_on_gpus.values():
            for job_index in reversed(range(len(running_jobs))):
                job = running_jobs[job_index]

                if job.is_done():
                    running_jobs.pop(job_index)
                    self._progress_bar.update(1)

                    return True

        return False

    def _wait_for_free_gpu(self):
        while True:
            job_removed = self._remove_completed_jobs()

            if job_removed:
                return

    def _wait_for_running_jobs(self):
        while not self.is_done():
            self._remove_completed_jobs()
