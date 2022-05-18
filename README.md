Code written by Duncan Barlow at the Universite de Bordeaux. Some code taken from other sources but cited.

# README for PDD Neural Network

## Quick start

To generate training data in file "Data" with 10 examples:

python training\_data\_generation.py Data 10

To generate 1 neural network run for 10 epochs from the training data in file "Data":

python neural\_network\_generation.py Data 10 1

## Additional Install

### To run data generation you will need:
Ifriit (Univeriste de Bordeaux, inverse ray tracing module) will need to be installed. Requests to arnaud.colaitis@u-bordeaux.fr.
You will need the python modules: healpy and netcdf4
These can be installed via conda using:

conda config --add channels conda-forge
conda install healpy
conda install netcdf4

## To run the neural network you will need:
You will need the python module: tensorflow.
These can be installed via conda using:

conda install tensorflow

