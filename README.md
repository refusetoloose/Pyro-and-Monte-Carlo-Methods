# Pyro and Monte Carlo Methods

This repository contains the code and materials for Pyro and Monte Carlo Methods. In this lab, we explore Pyro, a probabilistic programming library, and Monte Carlo methods applied to machine learning problems.

## Table of Contents

- [Lab Setup](#lab-setup)
- [Lab Activities](#lab-activities)
  - [Coin Toss](#coin-toss)
  - [Monte Carlo](#monte-carlo)
- [Resources](#resources)

## Lab Setup

Follow the instructions below to set up your environment and prepare for the lab activities:

1. **Download Data Files**: Download all the data files listed in the Materials and Equipment section of the lab instructions and put them in a folder on your computer created for Lab 7.

2. **Open in VS Code**: Open the folder in Visual Studio Code. The initial structure should resemble the screen shown in Figure 1 of the lab instructions.

3. **Create Virtual Environment**: Create a virtual environment named `venv` and configure VS Code to use it as the default Python interpreter.

4. **Install Dependencies**: Install PyTorch and the required dependencies specified in `requirements.txt`. Refer to Lab 2, Activity 1, Steps 3 to 6 for reference.

5. **Setup Complete**: Your environment is now set up and ready to proceed with the lab activities.

## Lab Activities

### Coin Toss

In this section, we use the Pyro library's primitives to model a Bernoulli distribution for a fair coin toss, utilizing PyTorch tensors to increase simulation speed.

- Complete the `coin_toss()` function.
- Complete the `coin_toss_tensor()` function.
- Run all the code cells and compare the speed of the last two cells. Which one is significantly faster?

### Monte Carlo

In this section, we work with the MNIST dataset and explore machine learning techniques using decision trees and logistic regression classifiers. We then use the Monte Carlo method to compare the performance of these classifiers.

- Create and fit a DecisionTreeClassifier object.
- Create and fit a LogisticRegression object.
- Print the accuracy score for both classifiers.
- Complete the `model()` and `monte_carlo()` functions.
- Run the code cells and compare the simulation results of the classifiers. Which performs better and why?

## Resources

- [An Introduction to Models in Pyro](http://pyro.ai/examples/intro_part_i.html)
- [torch.ones Documentation](https://pytorch.org/docs/stable/torch.html#torch.ones)
- [Bernoulli Distribution](http://docs.pyro.ai/en/0.2.1-release/distributions.html#bernoulli)
- [Uniform Distribution](http://docs.pyro.ai/en/0.2.1-release/distributions.html#uniform)
- [Simple Hamiltonian Monte Carlo Kernel (HMC)](http://docs.pyro.ai/en/0.2.1-release/mcmc.html#pyro.infer.mcmc.HMC)
- [Markov Chain Monte Carlo Algorithm (MCMC)](http://docs.pyro.ai/en/0.2.1-release/mcmc.html#pyro.infer.mcmc.MCMC)

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

## License
Distributed under the MIT License. See LICENSE for more information.
