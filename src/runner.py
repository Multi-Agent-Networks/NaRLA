import os
import argparse
import itertools
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--max_concurrent',type=int, default=16, help='Number of concurrent processes')
args = parser.parse_args()

NUM_REPEATS  = range(0,10)
NEURON_TYPE  = ['DQN', 'PG']
UPDATE_EVERY = [1]
REWARD_TYPE  = ['all','task','bio','bio_then_all']
UPDATE_TYPE  = ['sync'] # , 'async'
ENVIRONMENT  = ['binary', 'cart']
NUM_LAYERS   = [0, 1, 2, 5]

params = list(
    itertools.product(
        NUM_REPEATS, NEURON_TYPE, UPDATE_EVERY, REWARD_TYPE, UPDATE_TYPE, ENVIRONMENT, NUM_LAYERS
    )
)

pbar             = tqdm(total=len(params))
current_jobs     = []
template_command = 'python main.py --repeat_num {} --neuron_type {} --update_every {} --reward_type {} --update_type {} --env {} --num_layers {}'

while params:
    # RUN JOBS WHILE NOT AT LIMIT
    while len(current_jobs) < args.max_concurrent:
        param   = params.pop()
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
