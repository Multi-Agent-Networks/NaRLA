# NaRLA
Giving Up Control: Neurons as Reinforcement Learning Agents

## Run main
```bash 
python3 main.py \
  --results_directory Results \
  --environment CART_POLE \
  --neuron_type DEEP_Q
```

## Execute runner
Run `main.py` with a product of all the settings
```bash 
python3 scripts/run_jobs.py \
  --settings.results_directory RunnerResults \
  --environments CART_POLE \
  --gpus 0 \
  --jobs_per_gpu 2 \
  --neuron_types DEEP_Q ACTOR_CRITIC \
  --number_of_layers 1 2 3 4 5 6 7 8 9 10 
```
