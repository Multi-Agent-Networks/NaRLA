# NaRLA

![](https://github.com/Multi-Agent-Networks/NaRLA/actions/workflows/sphinx.yml/badge.svg)

For installation instructions and API documentation please refer to the [docs](https://multi-agent-networks.github.io/NaRLA/)

![network.png](figures%2Fnetwork.png)


## Run main
```bash 
narla main.py \
  --results_directory Results \
  --environment CART_POLE \
  --neuron_type DEEP_Q
```

## Execute runner
Run `main.py` with a product of all the settings
```bash 
narla scripts/run_jobs.py \
  --settings.results_directory RunnerResults \
  --environments CART_POLE \
  --gpus 0 1 2 3 \
  --jobs_per_gpu 5 \
  --neuron_types DEEP_Q ACTOR_CRITIC \
  --number_of_layers 1 2 3 4 5 6 7 8 9 10 
```

## Citation
This repository accompanies the paper [*Giving Up Control: Neurons as Reinforcement Learning Agents*](https://arxiv.org/abs/2003.11642)

```
@misc{ott2020giving,
    title={Giving Up Control: Neurons as Reinforcement Learning Agents},
    author={Jordan Ott},
    year={2020},
    eprint={2003.11642},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```