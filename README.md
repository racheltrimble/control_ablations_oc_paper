# control_ablations_oc_paper

Infrastructure for running ablation and sweep experiments on different control algorithms. Intended for flexible usage across heuristic control, RL and optimal control. This variation is for replicating experiments on a paper comparing optimal control performance to various simple heurisitcs.

## Installation
In the root directory:
pip install .

The optimal control code is based on casadi and requires the installation of the ma97 HSL linear solver. This can be installed by running the following the instructions here:
https://github.com/casadi/casadi/wiki/Obtaining-HSL

For any code using the optimal control functionality, the compiled library needs to be linked into the control ablations optimal control implementation by setting:

    from control_ablations import config
    config.hsl_path = 'absolute_path_to_hsl_lib'

## Repository Structure
The repository contains two main submodules:
- ablation_infra: Contains the underlying infrastructure for running experiments.
- generic_targets: Contains base classes implementing different types of control algorithms.

## Code Structure and Terminology
The purpose of the control_ablations code is to enable a researcher to run a "study". A study is a series of computational tests run with different settings each time (for example a sweep or an ablation).
A study works by running a series of "target" objects with the different settings. These targets are intended for testing different control algorithms against one another and so each one implements a specific control algorithms and executes it on a particular environment according to the supplied settings. The target is responsible for up to four phases of operation - tuning (e.g. determining the best hyperparameters), training (e.g. for learning algorithms), evaluation and plotting outputs that are required per trained iteration.
The study uses a Runner object to create and run the targets for to allow for local execution or e.g. on a slurm cluster.
After all the targets have been run the study calls an ablation analyser which is used to analyse data from across runs on the different targets. This is split into a series of plot_blocks which must be implemented by the user.
The storage of data and analysis files is managed by a hierarchy of IO objects.
