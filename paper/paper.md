---
title: 'open_nipals: An sklearn-compatible python package for NIPALS dimensional reduction'
tags:
  - Python
  - multivariate analysis
  - machine learning
  - dimensional reduction
  - principal component analysis
  - partial least square regression
authors:
  - name: Niels Schlusser
    orcid: 0000-0002-3534-2153
    equal-contrib: true
    affiliation: 1
  - name: Ryan Wall
    orcid: 0000-0003-2051-5899
    equal-contrib: true
    affiliation: 2
  - name: David R. Ochsenbein
    orcid: 0000-0001-8066-9561
    affiliation: 1
affiliations:
 - name: Cilag GmbH International, Schaffhausen, Switzerland
   index: 1
 - name: Johnson & Johnson Innovative Medicine, Titusville, New Jersey, USA
   index: 2

date: 28 April 2025
bibliography: paper.bib

---

# Summary

`open_nipals` is a python package that implements the Nonlinear Iterative Partial Least Squares (NIPALS) algorithm [@Geladi1986] for Partial Least Squares (PLS) regression as well as Principle Component Analysis (PCA). 
It employs the data transformation methods `fit()` and `transform()` from scikit-learn [@Pedregosa2011] and leverages Nelson's Single Component Projection (SCP) method for the imputation of missing data [@Nelson1996]. 
The NIPALS algorithm represents an alternative to the common Singular Value Decomposition (SVD) procedure for both PCA and PLS implemented in `scikit-learn` [@Pedregosa2011]. It is an iterative procedure that processes the data and internal matrices vector-wise and iteratively. When combined with SCP, NIPALS allows natural handling of missing data and setting tailored accuracy goals.

# Statement of Need

Python has emerged as a popular and comparatively simple programming environment for the development of machine learning and data science applications.
Packages like `numpy` for vector operations [@Harris2020], `pandas` for the handling of tabular data [@pandas2020], and `scikit-learn` for orthodox machine learning techniques like Random Forests, Support Vector Machines (SVM), and Principal Component Analyses (PCA) [@Pedregosa2011] promote python's success in extracting patterns from big and complex data sets.
However, `scikit-learn` relies on Singular Value Decomposition (SVD) for its PCA and PLS classes, with negative effects on performance for applications like batch manufacturing, where missing data is common [@Nelson1996].
PCA and PLS models require unit scaled and mean centered input data, a feature that is nicely implemented in `scikit-learn`'s `StandardScaler` class.  
To this end, we felt the need to complement `scikit-learn` with an implementation of the NIPALS algorithm for PCA and PLS. 

# Related Software

To our knowledge, the only other maintained open-source python package that implements the NIPALS algorithm for PCA and PLS is Salvador García Muñoz' `pyphi` [@Garcia2019]. 
Our implementation is different in the following aspects:
1. `open_nipals` follows the template of `scikit-learn`, which allows:
   1. Integration with other `scikit-learn` modules, e.g. the `StandardScaler`
   2. Accumulation of multiple transformation steps into a `sklearn.pipeline`.
2. `open_nipals` uses Nelson's single component projection method [@Nelson1996] for score calculation in the face of missing values.
3. The utility class of `open_nipals` contains `ArrangeData`, another `scikit-learn` style data transformer object that ensures correct ordering and quantity of input columns.

# Functionality

Wherever possible, `open_nipals` inherits structures from parent classes in `scikit-learn`. 
In principle, its functionality can be split into three parts: 
1. Utility functions for data preprocessing
2. Principal Component Analysis
3. Partial Least Squares regression. 

We decided to combine PCA and PLS functionality into one package, such that they can share common utility functions, e.g. -- but not limited to -- the `ArrangeData` class and matrix multiplication with missing values.

## Data Preprocessing, and Utility Functions

It is strongly encouraged to mean-center the input data for both PCA and PLS, and scale their variance to unity, e.g. with `sklearn`'s `StandardScaler`. Moreover, the `ArrangeData` class of `open_nipals` ensures correct ordering of the input columns, as well as proper formatting.
A code example for preprocessing could therefore look like:
```python
from open_nipals.utils import ArrangeData
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('my_data.csv')

# Create objects
arrdat = ArrangeData()
scaler = StandardScaler()

# Fit and transform data using both objects
data = scaler.fit_transform(arrdat.fit_transform(df))
```

## PCA

Principal Component Analyses with `open_nipals` utilize a `NipalsPCA` transformer object, that can be fitted to and transform input data (and both at once), e.g. with:
```python
from open_nipals.nipalsPCA import NipalsPCA

model = NipalsPCA()
transformed_data = model.fit_transform(data)
```
The number of fitted components can be specified with the `n_components` argument in the constructor, which defaults to `n_components=2`. 
After having constructed the object, components can be added or subtracted using the `set_components()` function. 
Once fitted, components are stored so they do not have to be fitted again. 
This saves compute time should the developer decide to use lower number of components than are fitted and later move back to a higher number of principal components.
A pseudo-inverse can be calculated with the `inverse_transform()` function. 
The distance of a given data point from the average of the training data within the PCA model (in-model distance, IMD) can be calculated with `calc_imd()`, where $\mathrm{Hotelling's\, T^2}$ [@Hotelling1931] is implemented and could be extended to other IMD metrics (e.g. Mahalanobis Distance). 
Conversely, the out-of-model distance (OOMD, calculated by `calc_oomd()`) gives a measure of the distance to the model hyperplane. 
This is available as two metrics, `DModX` and `QRes` [@Eriksson1999]. 

![Figure 1: PCA-modelled data points on the IMD-OOMD plane with 0.95 confidence interval.](./plots/HT2_DModX_example_plot.png)

Finally, the `calc_limit()` function calculates theoretical limits on both IMD and OOMD such that a specified fraction `alpha` of the data lies within these limits. 
The IMD-OOMD plot (see [@Fig1]) is assumed to follow an f-distribution [@Brereton2016]. 

## PLS

Similarly, Partial Least Squares regressions require a `NipalsPLS` object. 
Basic functionality includes the `fit()`, `transform()`, and `fit_transform()` methods:
```python
from open_nipals.nipalsPLS import NipalsPLS

model = NipalsPLS()
transformed_x_data, transformed_y_data = model.fit_transform(data_x, data_y)
```
`NipalsPLS` similarly contains a pseudo-inverse tranform that returns simulated data given a set of PLS scores with `inverse_transform()`, `calc_oomd()` for the out-of-model distance with either `QRes` or `DModX` as implemented metrics, `calc_imd()` for the in-model distance, using the $\mathrm{Hotelling's\, T^2}$ metric. `NipalsPLS` differs primarily from `NipalsPCA` by the inclusion of a `predict()` method to predict a y-matrix from an x-matrix with a previously fitted model, and the calculation of the regression vector with `get_reg_vector()`. 

# Availability

`open_nipals` is available open-source under APACHE 2.0 license from this github repository [@Ochsenbein2025]. We appreciate your feedback and contributions.

# Acknowledgements

We acknowledge support by Johnson & Johnson Innovative Medicine. 
In particular, we would like to express our gratitude to Samuel Tung and Tyler Roussos 
of the Open Source Working Group (OSWG) within J&J, who helped driving the publication process of `open_nipals`.

# References