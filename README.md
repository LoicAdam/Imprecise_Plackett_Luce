# Learning of robust preferences

Building a robust model to learn preferences that can be partial or with missing labls.

# How it works.

## What is needed

Some files use the python library choix: https://pypi.org/project/choix/.

## Datasets (in data).

The data folder contains multiple datasets that can be used to learn preferences.

For all of them (except Nascar), they consist of multiple features (X)
and of rankings of multiple labels (Y).

* **data_ranking.py**: contains methods to read the files.
* **create_data.py**: create sushi and nascar data sets.

## Plackett-Luce

To find the probability of a ranking, a statistical model is fitted on the rankings:
the Plackett-Luce model. More information on the model can be found on *Analyzing and
Modeling Rank Data* of John I. Marden.

## Learning with Plackett-Luce (in model)

### Precise case

Only full rankings are given. No abstention from prediction is possible.

Two algorithms:
* **IB_pl.py**: Instance-Based, based on local predictions.
* **GLM_pl.py**: General Linear Models, based on global predictions. Note: GLM does not seem to work well.

**opti_pl.py** contains two optimisation algorithms (MM an Markov) for the IB algorithm.

### Imprecise case

Abstention from ordering labels is possible, i.e. it is possible to not say if
a label is preferred to another or the opposite.

Two algorithms:
* **abstention_pl.py**: classic abstention method, where a threshold is applied to
the preferences to only order labels with a high preference level.
* **contour_pl.py**: based on contour likelihood function to modelise imprecise probabilities.

## Synthetic PL

**synthetic_pl.py** contains a script to generate a synthetic dataset from PL parameters
and then compare the two methods for imprecise predictions.

## Evaluation (in model)
 
* **cv_pl.py**: cross validation methods for different experiments.
* **evaluation.py**: metrics to evaluation the different results
* **tools_pl.py**: methods to perturb the data and other tools.

## Examples

* **compare_imprecise_all.py** and **compare_imprecise_single.py**: compare contour likelihood algorithm with abstention algorithm.
* **cut_example.py** to see how the generation method works.
* **cv_glm_all.py** and **cv_ib_all.py** evaluation of GLM and IB algorithms.
* **data_evolution.py** to see the evolution of prediction's quality with the amount of data.
* **inference.py** show how our method infere.
* **missing_cv_evolution.py** evaluation of performances with missing labels.
* **synthetic_pl.py** synthetic data sets.
