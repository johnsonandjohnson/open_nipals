# `open_nipals` PCA and PLS package

This package implements the nonlinear iterative partial least squares (NIPALS) algorithm for principal component analysis (PCA) and partial least squares (PLS) regression in a `scikit-learn` compatible fashion. 
In contrast to orthodox methods for PCA and PLS, the NIPALS algorithm is an iterative method, allowing free tuning of desired numerical performance and precision.
Moreover, it naturally integrates with Nelson's Single Component Projection method for missing data imputation.

# Quickstart

Install the package with `pip install open-nipals`.

Training a `NipalsPCA` model can look as simple as:
```python
from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPCA import NipalsPCA

# input data frame df

# standard-scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# train PCA model
model = NipalsPCA()
transformed_data = model.fit_transform(X=data)
```

A minimal example of fitting a `NipalsPLS` model:
```python
from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPLS import NipalsPLS

# input data frames df_x, df_y

# standard-scale data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaled_x_data = scaler_x.fit_transform(df_x)
scaled_y_data = scaler_y.fit_transform(df_y)

# train PLS model
model = NipalsPLS()
transformed_x_data, transformed_y_data = model.fit_transform(X=scaled_x_data, y=scaled_y_data)
```

# Key Features and API Overview

## Preprocessing

Both the `NipalsPCA` and `NipalsPLS` classes expect a numpy array as an input with rows as samples and columns as features. 
Additionally, these array columns should have zero mean for best performance; typically this is done with a sklearn `StandardScaler` object. 
Note that it is *highly* encouraged to mean-center the input data before training an `open_nipals` model on it.

Note: If the input data is a `pandas` dataframe, you can fit and instantiate an `ArrangeData` object which will ensure all future datasets come to the appropriate shape and column order.

```python
from open_nipals.utils import ArrangeData
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load some arbitrary data
df = pd.read_csv('my_data.csv')

# Invoke preprocessing pipeline
arrdat = ArrangeData()
scaler = StandardScaler()

# Both scaler and arrdat should be saved for future use
data = scaler.fit_transform(arrdat.fit_transform(df))

# data is ready to model
```

## Model fitting and transforming

The number of components can be specified as an argument to the constructor, with the default `n_components=2`. 
After fitting, components can be added or removed by the `set_components()` method without having to fit the entire model again from scratch.
Components that were once fitted but not needed any more are saved for possible later use. 

Functions of the `scikit-learn` API implemented by `open_nipals`:

- `fit()` for model fitting
- `transform()` for transforming data given a fitted model
- `fit_transform()` as a combination of `fit()` and `transform()`
- a pseudo-inverse transformation `inverse_transform()`, making the model predict how the data would look like


## PLS prediction

One particular feature of PLS models is that they can predict dependent variables. To this end, run `model.predict()`, where either a matrix of X data `X`, 
or a matrix of X scores `scores_x` need to be given as arguments, e.g. 
```python
predicted_y_data = model.predict(X=data_x)
```


## Distances

In-model distances (IMD) and out-of-model distances (OOMD) are metrics of model accuracy.
They can be calculated for PCA and PLS models with:
```python
# Must be scaled data
hotelling_t2 = model.calc_imd(input_array = data)

# also must be scaled, default metric is QResiduals or 'QRes'
dmodx = model.calc_oomd(input_array = data, metric = "DModX")
```

## Explainability

Similar to `scikit-learn`, the attribute `explained_variance_ratio_` measures the ratio of variance that each component of the model explains.
`NipalsPLS` has two of those arrays, one for the X and one for the y data.
Note that the NIPALS algorithm avoids calculating eigenvalues, therefore they are not accessible as the `explained_variance_` attribute.

Additionally, the regression vector can be calculated for a `NipalsPLS` model with `get_reg_vector()`.
The regression vector is a measure of how relevant each X feature is for the prediction of the y data.


# References

PLS algorithm implemented from Chapter 6 of:
> Chiang, Leo H., Evan L. Russell, and Richard D. Braatz.
> Fault detection and diagnosis in industrial systems.
> Springer Science & Business Media, 2000.

One of the most concise definitions can be found in this paper on page 7:
> Geladi, P.; Kowalski, B. R. Partial Least-Squares Regression: A Tutorial.
> Analytica Chimica Acta 1986, 185, 1–17.
> https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
> Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F. Missing data methods
> in PCA and PLS: Score calculations with incomplete observations.
> Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.


# Contributing

If you would like to contribute to `open_nipals`, please check out our [github repo](https://github.com/johnsonandjohnson/open_nipals).
For contribution guidelines please refer to the `CONTRIBUTING.md` in the repo, or the [contributor's guide](https://open-nipals.readthedocs.io/en/stable/contributing.html) in the online documentation.

# License

`open_nipals` is distributed under the BSD 3-clause license.

# Citation
This documentation refers to [`open_nipals v2.0.1`](https://github.com/johnsonandjohnson/open_nipals/tree/v2.0.1). 
An archived version of the code can be found under this DOI [10.5281/zenodo.18375840](https://10.5281/zenodo.18375840).
