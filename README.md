Code written by Duncan Barlow at Universite de Bordeaux. Some code taken from other sources but cited.

# README for PDD Neural Network

## Clone

use command:

     git clone https://github.com/DuncanBarlow/PDDOptimisation.git

via https.

## Quick start

Change directory into "python\_scripts".

To generate training data in file "Data/Data\_input" with "10" examples:

     python training_data_generation.py ../Data/Data_input 10 run_type=full

To run the optimisation suite use:

     python optimize.py ../Data/Data_output 100 2 10 1 10 1 10 0 12345 ../Data/Data_input

A brief guide to the meaning is given in the table below but the feature is still underdevelopment so read the source code for a more accurate understanding.
     #                   dir                 iex  init_type  bayes_opt grad_descent random_sampler random_seed  dir
     #python optimize.py ../Data/Data_output 100   0-2 10     0-1 10     0-1  10        0           12345      ../Data/Data_input

## Additional Install

     conda create -n <write_environment_name_here> "scipy>=1.9.1" jupyterlab netcdf4 numpy
     conda activate <write_environment_name_here>
     conda install -c conda-forge healpy bayesian-optimization

### To run data generation you will need:
Ifriit (University of Rochester, inverse ray tracing module) will need to be installed. Requests to acol@lle.rochester.edu.
You will need the python modules: healpy and netcdf4
These can be installed via conda using:

     conda config --add channels conda-forge
     conda install netcdf4 healpy

### To run the optimizers you will need:
You will need python modules: bayesian-optimization

     conda install -c conda-forge bayesian-optimization

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
