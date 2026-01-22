# NIPALS PCA and PLS package
This package implements the nonlinear iterative partial least squares (NIPALS) algorithm for principal component analysis (PCA) and partial least squares (PLS) regression. 

## Nipals PCA
Implements the NIPALS algorithm for principal components analysis in python. 

One of the most concise definitions can be found in this paper on page 7:
    Geladi, P.; Kowalski, B. R. Partial Least-Squares Regression: A Tutorial.
    Analytica Chimica Acta 1986, 185, 1–17.
    https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
    Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F. Missing data methods
    in PCA and PLS: Score calculations with incomplete observations.
    Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.

## Nipals PLS
Implements the NIPALS algorithm for partial least squares regression in python. 

Algorithm implemented from Chapter 6 of:
    Chiang, Leo H., Evan L. Russell, and Richard D. Braatz.
    Fault detection and diagnosis in industrial systems.
    Springer Science & Business Media, 2000.

Alternative algorithm derivation from:
    Geladi, P.; Kowalski, B. R.
    Partial Least-Squares Regression: A Tutorial.
    Analytica Chimica Acta 1986, 185, 1–17.
    https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
    Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F.
    Missing data methods in PCA and PLS: Score calculations
    with incomplete observations.
    Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.


# Installation
The package can be installed via the python package index (PyPI) or from the cloned git repo. 
If you would like to only use the package without needing acccess to the code, the first version is recommended.
If you would like to modify the code and/or contribute to the package, it is recommended to install via cloning the git repository.

## PyPI
You can install the package from PyPI with `pip install open-nipals`.

## Git repository
1. Clone git repository
2. Open a command line
3. Navigate to git repository
4. Run `pip install .`

NOTE: if you plan on working on the package please see [CONTRIBUTING.md](./CONTRIBUTING.md) for installation and environment setup instructions

# Example usage

## Preprocessing
Both the `NipalsPCA` and `NipalsPLS` classes expect a numpy array as an input with rows as samples and columns as features. Additionally, these array columns should have zero mean for best performance; typically this is done with a sklearn `StandardScaler` object. 

Note: If the input data is a `pandas` dataframe, you can train an `ArrangeData` object which will ensure all future datasets come to the appropriate shape and column order.

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

## Model fitting
The number of components can be specified as an argument to the constructor, with the default `n_components=2`. After fitting, components can be added by the `set_components()` method without having to fit the entire model again from scratch.

### PCA
```python
from open_nipals.nipalsPCA import NipalsPCA

# data is the numpy data matrix generated in the preprocessing
model = NipalsPCA()
transformed_data = model.fit_transform(X=data)
```

### PLS
```python
from open_nipals.nipalsPLS import NipalsPLS

# note the X/Y data blocks would need to have
# separate arrangeData and StandardScaler objects
model = NipalsPLS()
transformed_x_data, transformed_y_data = model.fit_transform(X=data_x, y=data_y)
```

## Distances
In-model distances (IMD) and out-of-model distances (OOMD) can be calculated for both PCA and PLS models
```python
# Must be scaled data
hotelling_t2 = model.calc_imd(input_array = data)

# also must be scaled, default metric is QResiduals or 'QRes'
dmodx = model.calc_oomd(input_array = data, metric = "DModX")
```

## PLS prediction
One particular feature of PLS models is that they can predict dependent variables. To this end, run `model.predict()`, where either a matrix of X data `input_x`, 
or a matrix of X scores `scores_x` need to be given as arguments, e.g. 
```python
predicted_y_data = model.predict(X=data_x)
```


