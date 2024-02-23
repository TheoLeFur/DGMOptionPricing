# DGMOptionPricing
We implement a version of the Deep Galerkin Method for solving high dimensional quasilinear parabolic partial differential equations. Applications include numerical pricing of American Options with payoff depending on a large number of stocks, but also for controlling SPDEs like the stochastic heat equation. 
We refer to [DGM](https://arxiv.org/pdf/1708.07469.pdf).

# I. Getting started

Clone the repository using:
```
git clone https://github.com/TheoLeFur/DGMOptionPricing.git
```
and download the requirements using
```
pip install -r requirements.txt
```
To run the model:
```
python3 main.py
```
