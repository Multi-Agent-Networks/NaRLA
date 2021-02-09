import os
import random
import argparse
import itertools
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--max_concurrent',type=int, default=1, help='Number of concurrent processes')
args = parser.parse_args()

NUM_REPEATS  = range(0,10)
REWARD_TYPE  = ['task','bio']
NUM_LAYERS   = range(1,21,3)
NUM_NODES    = range(15, 100, 25)

params = list(
    itertools.product(
        NUM_REPEATS, REWARD_TYPE, NUM_LAYERS, NUM_NODES
    )
)
random.shuffle(params)

pbar             = tqdm(total=len(params))
current_jobs     = []
template_command = 'python main.py --repeat_num {} --reward_type {} --num_layers {} --num_nodes {} --verbose'

while params:
    # RUN JOBS WHILE NOT AT LIMIT
    while len(current_jobs) < args.max_concurrent:
        param   = params.pop()
        if param[-2] == 1 and param[-1] != 15:
            continue

        command = template_command.format(*param)
        proc    = subprocess.Popen([command],shell=True,stdout=open('logfile.txt', 'a'), stderr=subprocess.STDOUT)

        # STORE PROCESS ID TO MONITOR JOB
        current_jobs.append(proc)

    # WHILE AT CAPACITY CHECK JOB STATUS
    while len(current_jobs) == args.max_concurrent:
        for i in reversed(range(args.max_concurrent)):
            # IF JOB IS DONE REMOVE FROM LIST
            if current_jobs[i].poll() is not None:
                current_jobs.pop(i)
                pbar.update(1)
