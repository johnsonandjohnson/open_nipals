"""
Algorithm implemented from Chapter 6 of
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

(C) 2020-2021: Ryan Wall (lead), David Ochsenbein, YBaranwal
revised 2024: Niels Schlusser
"""

from __future__ import (
    annotations,
)  # needed so we can return NipalsPLS class in our type hints
import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.exceptions import NotFittedError
import warnings
from open_nipals.utils import _nan_mult
from typing import Optional, Tuple, Union


class NipalsPLS(_PLS):
    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 10000,
        tol_criteria: float = 10**-10,
        mean_centered: bool = True,
        force_include: bool = False,
    ):
        """Constructor for initialization

        Args:
            n_components (int): The number of components to use in the model
            max_iter (int): The maximum number of iterations to use when fitting
            tol_criteria (float): Tolerance limit to compare when fitting
            mean_centered (boolean): Whether the data is mean centered or not.
                It is HIGHLY suggested to mean center your data.
            force_include (bool, optional): True will force including the data
                which has all nans in y-block. Defaults to False.

        Returns:
            NipalsPLS object
        """

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria
        self.mean_centered = mean_centered
        self.force_include = force_include

        self.fit_data_x = None  # n_rows_x x n_cols_x
        self.fit_data_y = None  # n_rows_y x n_cols_y

        self.loadings_x = None  # n_cols_x x n_components
        self.fit_scores_x = None  # n_rows_x x n_components
        self.weights_x = None  # n_cols_x x n_components

        self.loadings_y = None  # n_cols_y x n_components
        self.fit_scores_y = None  # n_rows_y x n_components

        self.regression_matrix = None  # n_components x n_predictors

        warnings.simplefilter(
            "always", UserWarning
        )  # Ensure showing of warnings

    @property
    def fitted_components(self) -> int:
        """Get total # of LVs in model.
        This may differ from self.n_components which is the number of
        components used by the model.

        Returns:
            int
        """
        if self.loadings_x is not None:
            return self.loadings_x.shape[1]
        else:
            return 0

    def _filter_nan_rows(self, X, y):
        """filter the nan rows and remove them,
        depending on force_include parameter
        """
        # Find rows in which all Y-values are NaN
        y_nan_rows = np.all(np.isnan(y), axis=1)
        if np.any(y_nan_rows):
            # Warn the user
            warnings.warn("Some Y-rows have no data in them!")
            # Depending on force_include, either drop or include
            if self.force_include:
                warnings.warn("Rows still included due to force_include.")
            else:
                warnings.warn(
                    "Rows with all NaNs in Y are dropped, see force_include."
                )
                rows_to_keep = np.invert(y_nan_rows)
                y = y[rows_to_keep, :]
                X = X[rows_to_keep, :]

        # Store the data used to fit
        return X.copy(), y.copy()

    def _add_components(self, n_add: int, verbose: bool = False):
        """Method for adding components to an already-constructed
        model.

        Args:
            n_add (int): number of components to add
            verbose (bool): Whether or not to print out additional
                convergence information. Defaults to False.
        """
        X = self.fit_data_x.copy()
        y = self.fit_data_y.copy()

        # filter and store the data used to fit
        self.fit_data_x, self.fit_data_y = self._filter_nan_rows(X, y)

        # Pull out shapes
        n_rows_x, n_cols_x = self.fit_data_x.shape
        n_rows_y, n_cols_y = self.fit_data_y.shape
        if n_rows_x != n_rows_y:
            raise ValueError("Rows in X do not match rows in Y!")

        # get current number of components
        fitted_components = self.fitted_components

        # if model is not fitted yet, this is the default
        # checks only for X-loadings in agreement with rest of script
        if fitted_components == 0:
            # Range of components to add
            num_lvs = range(n_add)

            # Preallocate space for all vectors
            p = np.zeros((n_cols_x, n_add))  # x loadings
            t = np.zeros((n_rows_x, n_add))  # x scores
            u = np.zeros((n_rows_x, n_add))  # y scores
            w = np.zeros((n_cols_x, n_add))  # weights
            q = np.zeros((n_cols_y, n_add))  # y loadings
            b = np.zeros((n_add, n_add))  # regression coeff for t & u
        else:
            # Range of LVs to add
            num_lvs = range(fitted_components, fitted_components + n_add)

            # There are loadings, so must deflate
            sim_data_x = self.inverse_transform(self.transform(X))
            sim_data_y = self.predict(X, self.fit_scores_x)
            X = X - sim_data_x
            y = y - sim_data_y

            p = np.append(
                self.loadings_x, np.zeros((n_cols_x, n_add)), axis=1
            )  # x loadings
            t = np.append(
                self.fit_scores_x, np.zeros((n_rows_x, n_add)), axis=1
            )  # x scores
            u = np.append(
                self.fit_scores_y, np.zeros((n_rows_x, n_add)), axis=1
            )  # y scores
            w = np.append(
                self.weights_x, np.zeros((n_cols_x, n_add)), axis=1
            )  # weights
            q = np.append(
                self.loadings_y, np.zeros((n_cols_y, n_add)), axis=1
            )  # y loading
            b = np.zeros(
                (fitted_components + n_add, fitted_components + n_add)
            )
            b[:fitted_components, :fitted_components] = (
                self.regression_matrix
            )  # regression coeff for t & u

        # residual matrices for first LV
        # x residuals, initiated as X (will be deflated in subsequent steps)
        x_res = X
        # y residuals, initiated as y (will be deflated in subsequent steps)
        y_res = y

        # nan_mask for X-block and y-block
        nan_mask_x = np.isnan(X)
        nan_mask_y = np.isnan(y)
        nan_flag = np.any(nan_mask_x) or np.any(nan_mask_y)
        # Loop over each LVs
        for ind_lv in num_lvs:
            # Specify column in Y to start.
            # Here we use column of Y with max variance
            std_y = np.nanstd(self.fit_data_y, axis=0)
            start_col = np.argmax(std_y)

            # Scores guesses for LV i is a column vector from the residuals
            ui = y_res[
                :, [start_col]
            ].copy()  # list in second position enforces column shape
            # Replace any nans in ui with zero; it's just a guess after all
            ui[np.isnan(ui)] = 0
            ti = ui.copy()

            iter_count = 0
            converged = False
            if verbose:
                print(
                    f"LV {ind_lv} Started"
                )  # oh wow, I didn't now about f strings, love it. - DRO

            while (not converged) and (iter_count < self.max_iter):
                if verbose:
                    print(f"...Iter {iter_count}")

                # save old t to check convergence
                ti_old = ti.copy()

                if not nan_flag:  # Loop for no NaN
                    wi = x_res.T @ ui  # x weight
                    wi = wi / np.linalg.norm(wi)  # Normalize weights
                    ti = x_res @ wi  # x score
                    qi = y_res.T @ ti  # y weight
                    qi = qi / np.linalg.norm(qi)  # Normalize y-weights
                    ui = y_res @ qi  # y score

                else:  # Loop for handling NaNs,
                    wi = _nan_mult(x_res.T, ui, nan_mask_x.T)
                    wi = wi / np.linalg.norm(wi)
                    ti = _nan_mult(x_res, wi, nan_mask_x, use_denom=True)
                    qi = _nan_mult(y_res.T, ti, nan_mask_y.T)
                    qi = qi / np.linalg.norm(qi)
                    ui = _nan_mult(y_res, qi, nan_mask_y)

                # Check convergence
                diff_norm = np.linalg.norm(ti - ti_old) / np.linalg.norm(ti)

                converged = diff_norm < self.tol_criteria

                iter_count += 1
                # end of while loop

            # Check whether it actually converged
            # or just terminated after max_iter
            if iter_count >= self.max_iter:
                print(f"max_iter Reached on LV {ind_lv}.")

            # x loading
            if not nan_flag:
                pi = (x_res.T @ ti) / (ti.T @ ti)
            else:
                pi = _nan_mult(x_res.T, ti, nan_mask_x.T)

            p_weight = np.linalg.norm(pi)  # Reused to scale ti,wi
            pi = pi / p_weight

            # Scaling x weight
            wi = (wi.T * p_weight).T

            # Scaling x score
            # Recalc score if NaNs
            if not nan_flag:
                ti = ti * p_weight  # Faster if no NaNs
            else:
                ti = _nan_mult(x_res, wi, nan_mask_x, use_denom=True)

            # regression coefficient, is a scalar!
            bi = (ui.T @ ti) / (ti.T @ ti)

            # populate data
            p[:, [ind_lv]] = pi
            t[:, [ind_lv]] = ti
            u[:, [ind_lv]] = ui
            w[:, [ind_lv]] = wi
            q[:, [ind_lv]] = qi
            b[ind_lv, ind_lv] = bi.item()

            # update residual matrices for next LV
            x_res = x_res - ti @ pi.T
            y_res = y_res - bi * ti @ qi.T

            # End of for loop

        self.fit_scores_x = t
        self.loadings_x = p
        self.weights_x = w
        self.fit_scores_y = u
        self.loadings_y = q
        self.regression_matrix = b

    def set_components(self, n_component: int, verbose: bool = False):
        """Method for setting the number of components in an
        already-constructed model. It checks to make sure that loadings
        exist for all of the set components and will fit extras if not.

        Args:
            n_component (int): the desired number of components.
            verbose (bool): Whether or not to print out additional
                convergence information. Defaults to False.

        Raises:
            TypeError: if n_component is not an int
            ValueError: if n_component < 1
        """
        if not isinstance(n_component, int):
            raise TypeError("n_component must be an integer!")
        elif n_component < 1:
            raise ValueError("n_component must be an int > 0")

        max_fit_lvs = self.fitted_components
        # If desired n <= max fit N, simply set the number
        if n_component <= max_fit_lvs:
            self.n_components = n_component
        else:
            n_to_add = n_component - max_fit_lvs
            self._add_components(n_to_add, verbose=verbose)
            self.set_components(n_component)

        return self  # So methods can be chained

    def fit(
        self,
        X: np.array,
        y: np.array,
        verbose: bool = False,
    ) -> NipalsPLS:
        """Function to fit PLS model from X/Y Data.

        Args:
            X (np.array): Input X data.
            y (np.array): Input Y data.
            verbose (bool, optional): Turn verbosity on and off.
                Defaults to False.

        Raises:
            NotFittedError: Model has not yet been fit

        Returns:
            NipalsPLS: A reference to the object.
        """
        if self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model Object has already been fit."
                + " Try set_components() or build a new model object."
            )

        # Check to see if the data is mean_centered; if not raise a warning
        if (not self._check_mean_centered(X)) and (self.mean_centered):
            warnings.warn(
                "X-Block appears to not be mean centered. "
                + "This will cause errors in prediction!"
            )

        if (not self._check_mean_centered(y)) and (
            self.mean_centered
        ):  # see above
            warnings.warn(
                "Y-Block appears to not be mean centered. "
                + "This will cause errors in prediction!"
            )

        self.fit_data_x = np.copy(X)
        self.fit_data_y = np.copy(y)

        self._add_components(n_add=self.n_components, verbose=verbose)

        # Allows model = NipalsPLS().fit(data)
        return self

    def transform(
        self, X: np.array, y: Optional[np.array] = None
    ) -> Union[np.array, Tuple[np.array, np.array]]:
        """Compute scores using model.

        Args:
            X (np.array): X-data.
            y (np.array, optional): Y-data. Defaults to None.

        Raises:
            NotFittedError: If model is not fit.

        Returns:
            Union[np.array, Tuple[np.array, np.array]]: Either the
                scores for X (when no Y-data is provided) or a tuple
                of two scores-arrays.
        """

        # Check whether the model is available or not
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit.")

        if y is None:
            scores_x = self._transform_xy(
                X, self.loadings_x, weights=self.weights_x
            )
            return scores_x
        else:
            scores_x = self._transform_xy(
                X, self.loadings_x, weights=self.weights_x
            )
            scores_y = self._transform_xy(y, self.loadings_y)
            return scores_x, scores_y

    def _transform_xy(
        self,
        input_array: np.array,
        loadings: np.array,
        weights: Optional[np.array] = None,
    ) -> np.array:
        """Copied the naive transform method from nipalsPCA
        This allows for calculating X or Y scores depending on given loads
        Note that for calculating X-scores, weights are used, however loads
        are used to deflate the matrix using the same scores.

        Args:
            input_array (np.array): The input array (X-data).
            loadings (np.array): The loadings.
            weights (np.array, optional): The weights. Defaults to None.

        Returns:
            np.array: The scores.
        """

        # Extract shape of input_array
        n, _ = input_array.shape

        # Extract variables for simplicity
        num_lvs = self.n_components
        if weights is None:
            # if no weights are provided, use loadings
            weights = loadings.copy()

        scores = np.zeros((n, num_lvs))

        nan_mask = np.isnan(input_array)
        resids = input_array.copy()
        # Looping over every loading
        # uses approach described in section 3.1 of McGregor paper
        # (Single component projection algorithm for missing data in PCA/PLS)
        for ind_lv in range(num_lvs):
            scores[:, [ind_lv]] = _nan_mult(
                resids, weights[:, [ind_lv]], nan_mask, use_denom=True
            )
            # deflate input data
            resids = resids - scores[:, [ind_lv]] @ loadings[:, [ind_lv]].T

        return scores

    def _check_mean_centered(self, data: np.array) -> bool:
        """Simple function to check if the data is mean centered along the
        rows within some small tolerance.

        Args:
            data (np.array): Data to check.

        Returns:
            bool: Whether or not the data is mean-centered.
        """
        # catch runtime warning: Mean of empty slice
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="Mean of empty slice"
            )
            # Calculate maximum mean of a given column
            try:
                maxmean = np.nanmax(np.abs(np.nanmean(data, axis=0)))
            except RuntimeWarning:
                maxmean = np.NaN

            # Return boolean
            return maxmean < 10**-10

    def fit_transform(
        self, X: np.array, y: np.array
    ) -> Tuple[np.array, np.array]:
        """Combine fit and transform methods into one command, sklearn style.

        Args:
            X (np.array): The X-data.
            y (np.array): The Y-data.

        Raises:
            ValueError: If attempt to use fit_transform on already fitted data.

        Returns:
            Tuple[np.array, np.array]: A tuple containing the fitted scores
                for X and Y.
        """

        # Check whether the model is available or not
        if not self.__sklearn_is_fitted__():
            # Fit
            self.fit(X, y)
            # Return X/Y Scores
            scores_x = self.fit_scores_x
            scores_y = self.fit_scores_y
            return scores_x, scores_y
        else:
            raise ValueError(
                "Model has already been fit. Try transform instead."
            )

    def calc_imd(
        self,
        input_scores: Optional[np.array] = None,
        input_array: Optional[np.array] = None,
        metric: str = "HotellingT2",
    ):
        """Function copied from nipalsPCA code. This will take in an input
            Array OR scores and return hotelling's T2 value for each row.
            In theory you could expand to include a Y-block in-model distance,
            but the value is limited for the typical use cases.

        Args:
            input_scores (Optional[np.array], optional): Scores array.
                Defaults to None.
            input_array (Optional[np.array], optional): Data array.
                Defaults to None.
            metric (str, optional): In-model-distance to compute.
                Must be one of set {'HotellingT2'}. Defaults to 'HotellingT2'.

        Raises:
            NotFittedError: If model has not been fit.
            ValueError: If neither scores nor data are provided.
            ValueError: If input scores shapes does not match model.
            ValueError: If unknown metric was requested.

        Returns:
            float: The within-model distance(s).
        """

        # Warnings and errors
        if not self.__sklearn_is_fitted__():  # if not fit
            raise NotFittedError(
                "Model has not yet been fit. "
                + "Try fit() or fit_transform() instead."
            )
        elif (input_array is None) and (
            input_scores is None
        ):  # If Nothing Given
            raise ValueError("No values provided.")

        # Calculate scores
        if input_array is not None:  # If we have data, calculate with that
            if (
                input_scores is not None
            ):  # Both inputs = warn and calc with only data
                warnings.warn(
                    "Both Scores and Data are given. Operating on Data alone."
                )
            # Get Scores
            scores = self.transform(X=input_array)
        elif input_scores is not None:  # If scores only, use those
            scores = input_scores

        # Calculate in model distances
        if metric == "HotellingT2":  # Calculate Hotelling's T2
            num_lvs_fit = self.n_components

            if scores.shape[1] != num_lvs_fit:  # Error if incorrect sizes
                raise ValueError(
                    "input_scores have more columns/latent variables than "
                    + "model was fit with."
                )
            else:
                # Calculate fit means/variances
                fit_means = np.mean(self.fit_scores_x[:, :num_lvs_fit], axis=0)
                fit_vars = np.var(
                    self.fit_scores_x[:, :num_lvs_fit], axis=0, ddof=1
                )
                # Calculate imd, the Hotelling's T2
                out_imd = np.sum(
                    (input_scores - fit_means) ** 2 / fit_vars, axis=1
                ).reshape(-1, 1)

        else:  # If incorrect metric given
            raise ValueError(
                "Unknown metric requested (metric = HotellingT2)."
            )

        return out_imd

    def inverse_transform(self, X: np.array) -> np.array:
        """Given a set of scores, return the simulated data.

        Args:
            X (np.array): The scores to transform back.

        Raises:
            NotFittedError: If model has not been fit.
            ValueError: If input scores shapes does not match model.

        Returns:
            np.array: The simulated data.
        """

        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. "
                + "Try fit() or fit_transform() instead."
            )

        # Check size of input_scores:
        _, m = X.shape

        if m != self.n_components:
            raise ValueError(
                "input_scores has number of columns different than "
                + "number of components in model"
            )

        # self.n_components as we may have built loadings to a larger n than
        # the current number of components
        out_data_x = X @ self.loadings_x[:, : self.n_components].T

        return out_data_x

    def calc_oomd(
        self, input_array: np.array, metric: str = "QRes"
    ) -> np.array:
        """Calculate the Out of Model Distance.
        In theory can be used for Y-block, but the value in
        typical use is limited.

        Args:
            input_array (np.array): The input data.
            metric (str, optional): The metric to compute.
            Supported metrics are: {'QRes','DModX'}. Defaults to 'QRes'.

        Raises:
            ValueError: If input metric is unknown.

        Returns:
            np.array: The out-of-model distance(s).
        """

        # Select particular metric
        if metric == "QRes":  # Q-residual
            # Get shape and Preallocate output
            n, _ = input_array.shape
            out_oomd = np.zeros((n, 1))

            # Calculate scores (transform should handle error Handling)
            scores = self.transform(input_array)

            # Calculate Fitted Data
            modeled_data = self.inverse_transform(X=scores)

            # Calculate Residuals
            residual = input_array - modeled_data

            nan_mask = np.isnan(residual)
            not_null = np.invert(nan_mask)
            # Calculate Q_residuals as DMODX is based off of this value
            for row in range(n):
                out_oomd[row, 0] = (
                    residual[row, not_null[row, :]]
                    @ residual[row, not_null[row, :]].T
                )

        elif metric == "DModX":  # DModX
            # Pull out params for weird Simca Factor
            n, _ = self.fit_scores_x.shape  # pylint: disable=unpacking-non-sequence
            num_lvs = self.n_components

            # Calculate the QRes
            out_oomd = self.calc_oomd(input_array, metric="QRes")

            # A0 is 1 if mean mean_centered.
            if self.mean_centered:
                A0 = 1
            else:
                A0 = 0

            K = self.fit_data_x.shape[1]
            factor = np.sqrt(n / ((n - num_lvs - A0) * (K - num_lvs)))
            out_oomd = factor.reshape(-1, 1) * np.sqrt(out_oomd)
        else:
            raise ValueError("Input metric not recognized")

        return out_oomd

    def predict(
        self, X: np.array = None, scores_x: np.array = None
    ) -> np.array:
        """Predict y from data or scores.

        Args:
            X (np.array, optional): Input X data. Defaults to None.
            scores_x (np.array, optional): Input scores. Defaults to None.

        Raises:
            NotFittedError: If model has not been fit.
            ValueError: If neither data nor scores are provided.

        Returns:
            np.array: The predicted y-values.
        """

        # Check whether the model is available or not
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit")

        # Handle different inputs, either scores or raw data
        if (X is None) and (scores_x is None):
            raise ValueError("Provide either data or scores")
        elif scores_x is not None:
            # if scores are present, calculate the result
            num_lvs = self.n_components
            pred_y = (
                scores_x[:, :num_lvs]
                @ self.regression_matrix[:num_lvs, :num_lvs]
                @ self.loadings_y[:, :num_lvs].T
            )
        else:
            # if no scores, but X, calculate scores and re-call func
            scores_x = self.transform(X)
            pred_y = self.predict(scores_x=scores_x)

        return pred_y  # Give the people what they want

    def get_reg_vector(self) -> np.array:
        """Give the user the regression vector for the model.

        Raises:
            NotFittedError: If the model has not been fit.

        Returns:
            np.array: The regression vector.
        """

        # Check whether the model is available or not
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit")

        reg_vects = self.weights_x[:, : self.n_components] @ (
            self.regression_matrix[: self.n_components, :]
            @ self.loadings_y[:, : self.n_components].T
        )

        return reg_vects

    def __sklearn_is_fitted__(self) -> bool:
        """Determine if this is fitted or not

        Returns:
            bool: is fitted or not
        """
        return not (self.fitted_components == 0)
