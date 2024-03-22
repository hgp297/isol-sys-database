# isol-sys-database

A database of isolated steel frames and their performance under earthquakes.

## Description

This repository is aimed at generating and analyzing isolated steel frames for their performance.
Most of the repository is dedicated towards automating the design of steel moment and braced frames isolated with friction or lead rubber bearings.
The designs are then automatically constructed in OpenSeesPy and subjected to a full nonlinear time history analysis.
The database is revolved around generating series of structures spanning the range of a few random design variables dictating over/under design of strength and displacement capacity variables, along with a few isolator design parameters.

Decision variable prediction via the SimCenter toolbox is also available (work in progress).

An analysis folder is available with some scripts performing data visualization and machine learning predictions.
The database is utilized to generate inverse design targeting specific structural performance.

### Dependencies
* Structural software:
	* OpenSeesPy 3.4.0
	* Python 3.9

* Data structure management:
	* Pandas 1.1.5+
	* Numpy 1.22.4+
	* Scipy 1.12.0+

* Machine learning analyses (required for design of experiment, inverse design):
	* Scikit-learn

* Visualization:
	* Matplotlib
	* Seaborn

* Decision-variable prediction:
	* Pelicun 3.1+


### Usage
The database generation is handled through main_\* scripts available in the ```src``` folder.
```src/analyses/``` contains scripts for data visualization and results processing.
Some past results are available in ```tfp-mf``` and ```loss```.

The project is still currently just a project for myself. Packaging for general usage is a future goal.

## Personal notes:
A reminder that this database is dependent on the OpenSees compatible with Python=3.9.
See opensees_build/locations/ for location of a working Opensees.pyd code.

## Research tools utilized

* [OpenSeesPy](https://github.com/zhuminjie/OpenSeesPy)
* [SimCenter Pelicun](https://github.com/NHERI-SimCenter/pelicun)