# FLGuard Evaluations

Just some of my own evaluations of the FLGuard algorithm proposed by [Nguyen et al. (2021)](https://arxiv.org/abs/2101.02281).

## Executing

First install the requirements:
```sh
pip install -r requirements.txt
```

Then the main experiments can be run with:
```sh
python main.py
```

## Quick recreation

The main experiment can be quickly recreated using docker with:
```sh
docker run ghcr.io/codymlewis/flguard:latest
```


## Results and Analysis

![FLGuard Evaluation Results](/plot.png)

We observed that the learning performance of the FLGuard algorithm is significant diminished when the size of local gradients is too
small (either due to small learning rate or small number of local training epochs). For example, in the case of SGD with a learning rate
of 0.01 on the MNIST dataset, the FLGuard algorithm only attains an accuracy of approximately 10% despite executing for 5000 global rounds.

Through experimentation on altered forms of the FLGuard algorithm, we found that this issue is the result of the clipping step located on
Line 11 of the paper's algorithm. However, completely removing the clipping step results must be ruled out as a feasible option as it would result in the loss
of some important properties of the algorithm, that are presented in the original paper.