## :warning: :warning: :warning:   This repository is a fork of the [ZhaozhiQIAN/D-CODE-ICLR-2022](https://github.com/ZhaozhiQIAN/D-CODE-ICLR-2022.git) and serves to verify additional approximation results for the proposed loss function.  

## Quick start
Follow the installation guidelines as per [ZhaozhiQIAN/D-CODE-ICLR-2022](https://github.com/ZhaozhiQIAN/D-CODE-ICLR-2022.git) 
```bash
$ git clone --recursive [repo-url]
cd path/to/project
python -m venv d-code
source d-code/bin/activate (Linux)
pip install -r requirements.txt
python -u run_experiments.py
```

# D-CODE-ICLR-2022
Code for [D-CODE: Discovering Closed-form ODEs from Observed Trajectories (ICLR 2022)](https://openreview.net/forum?id=wENMvIsxNN).



## Installation

Clone this repository and all submodules (e.g. using `git clone --recursive`).
Python 3.6+ is recommended. Install dependencies as per [`requirements.txt`](./requirements.txt).

## Replicating Experiments

Shell scripts to replicate the experiments can be found in [`run_all.sh`](./run_all.sh).

To run all the synthetic data experiments:
```bash
$ bash run_all.sh
```
You may also run the experiment steps individually, see [`run_all.sh`](./run_all.sh). 
To then produce the figures, run the Jupyter notebooks `Result Summary.ipynb`, `Fig3.ipynb`, `Fig5.ipynb`, `rebuttal.ipynb`.


## Citing

If you use this code, please cite the associated paper:

```
@inproceedings{NEURIPS2021,
  author = {Qian, Zhaozhi and Kacprzyk, Krzysztof and van der Schaar, Mihaela},
  booktitle = {International Conference on Learning Representations},
  title = {D-CODE: Discovering Closed-form ODEs from Observed Trajectories},
  url = {https://openreview.net/pdf?id=wENMvIsxNN},
  volume = {10},
  year = {2022}
}
```
