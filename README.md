### Overview
This repository contains the prototpye implementation of the proposed incremental A* approach presented in the paper 
***Online Process Monitoring Using Incremental State-Space Expansion: An Exact Algorithm*** 
by Daniel Schuster and Sebastiaan J. van Zelst.

Corresponding author: Daniel Schuster ([Mail](mailto:daniel.schuster@fit.fraunhofer.de?subject=github-incremental_a_star_approach))

Preprint: [https://arxiv.org/abs/2002.05945](https://arxiv.org/abs/2002.05945)

This prototype implementation is using a fork of the [pm4py library](https://pm4py.fit.fraunhofer.de). 


### Repository Structure
* The main proposed algorithm is implemented in 
`pm4py/algo/conformance/alignments/incremental_a_star/incremental_a_star_approach.py`.
* In `pm4py/algo/conformance/alignments/experiments/experiments.py` is an example script how to run the conducted experiments.

### Installation
For platform-specific installation instructions, please visit [http://pm4py.pads.rwth-aachen.de/installation/](http://pm4py.pads.rwth-aachen.de/installation/)