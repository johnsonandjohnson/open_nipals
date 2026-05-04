"""Code for calculating the PCA Loadings and Scores using NIPALS algorithm.

One of the most concise definitions can be found in this paper on page 7:
Geladi, P.; Kowalski, B. R. Partial Least-Squares Regression: A Tutorial.
Analytica Chimica Acta 1986, 185, 1–17.
https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F. Missing data methods
in PCA and PLS: Score calculations with incomplete observations.
Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.

(c) 2020-2021: Ryan Wall (lead), David Ochsenbein
revised 2024: Niels Schlusser
"""

from __future__ import (
    annotations,
)  # needed so we can return NipalsPCA class in our type hints
import warnings
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.covariance import LedoitWolf
from scipy.stats import f as F_dist
from open_nipals.utils import _nan_mult


class NipalsPCA(BaseEstimator, TransformerMixin):
    """The custom-built class to use PCA using the NIPALS algorithm, i.e.,
    the same algorithm used in SIMCA.

    Attributes:
    --------------------
    n_components : int
        The number of principal components.
    max_iter : int
        The max number of iterations for the fitting step.
    tol_criteria : float
        The convergence tolerance criterion.
    loadings : np.ndarray
        The loadings vectors of the PCA model.
    fit_scores : np.ndarray
        The fitted scores of the PCA model.
    fit_data : np.ndarray
        The data used to fit the model.
    mean_centered : bool
        Whether or not the original data is mean-centered.
    fitted_components : int
        The number of current LVs in the model (0 if not fitted yet.)
    explained_variance_ratio_ : np.ndarray
        The explained variance ratios per fitted component.

    Methods:
    --------------------
    transform
        Transform input data to scores.
    fit
        Fit PCA model on input data.
    fit_transform
        Fit PCA model on input data, then transform said data to scores.
    inverse_transform
        Obtain approximation of input data given fitted model and scores.
    calc_imd
        Calculate within-model distance.
    calc_oomd
        Calculate out-of-model distance.
    calc_limit
        Calculate suitable distance threshold given fitted data.
    set_components
        Change the number of model components.
    get_explained_variance_ratio
        Calculate explained variances as ratio of total explained variance.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 10000,
        tol_criteria: float = 1e-6,
        mean_centered: bool = True,
    ):
        """The constructor for the NipalsPCA class.

        Args:
            n_components (int, optional): The number of principle components.
                Defaults to 2.
            max_iter (int, optional): The maximum number of iterations until
                convergence. Defaults to 10000.
            tol_criteria (float, optional): Relative tolerance limit.
                Defaults to 1e-6.
            mean_centered (bool, optional): Whether or not the data is already
                mean-centered. Defaults to True.

        Returns:
            NipalsPCA object
        """
        # constructor
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria
        self.mean_centered = mean_centered

        self.loadings = None  # m x num_lvs matrix
        self.fit_scores = None  # n x num_lvs matrix

        self.fit_data = None  # n x m training data

        warnings.simplefilter(
            "always", UserWarning
        )  # Ensure showing of warnings

    @property
    def fitted_components(self) -> int:
        """Get total # of LVs in model.
        This may differ from self.n_components which is the number of
        components used by the model.

        Returns:
            int: Number of fitted components
        """
        if self.loadings is None:
            return 0
        else:
            return self.loadings.shape[1]

    def _add_components(self, n_add: int, verbose: bool = False):
        """Method for adding components to an already-constructed
        model.

        Args:
            n_add (int): number of components to add
            verbose (bool): Whether or not to print out additional
                convergence information. Defaults to False.
        """
        data = self.fit_data.copy()
        n, m = data.shape

        fitted_components = self.fitted_components

        # if model is not fitted yet, this is the default
        if fitted_components == 0:
            # Range of components to add
            num_lvs = range(n_add)

            # Preallocate Scores and loadings array
            scores = np.zeros((n, n_add))
            loadings = np.zeros((m, n_add))

        else:
            # There are loadings, so must deflate
            sim_data = self.inverse_transform(self.transform(data))
            data = data - sim_data

            # Range of LVs to add
            num_lvs = range(fitted_components, fitted_components + n_add)

            # Preallocate empty columns in scores/loadings
            scores = np.concatenate(
                [self.fit_scores, np.zeros((n, n_add))], axis=1
            )
            loadings = np.concatenate(
                [self.loadings, np.zeros((m, n_add))], axis=1
            )

        if verbose:
            print("Scores and Loads preallocated")

        # Logical mask of Nan Data
        nan_mask = np.isnan(data)
        nan_flag = np.any(nan_mask)
        if verbose:
            print("nan_mask Generated")

        # Loop for all LVs
        for i in num_lvs:
            # choose a column of input_array
            t_new = data[:, [0]].copy()

            # Replace any nans w/ zero
            t_new[np.isnan(t_new)] = 0
            if verbose:
                print("Score initialized")

            # Allocate variable for a convergence test
            converged = False
            conv_test = np.inf

            # Allocate the interation counter
            num_iter = 0

            # Loop until Converged
            while (not converged) and (num_iter < self.max_iter):
                if verbose:
                    print(f"LV {i} Iteration num_iter Started")
                    print(f"     Conv Test is {conv_test}")
                # Update t_old
                t_old = t_new.copy()

                # Calculate the loadings vector

                if nan_flag:
                    loadings_loc = _nan_mult(data.T, t_old, nan_mask.T)

                    # Normalize the loading
                    loadings_loc = loadings_loc / np.sqrt(
                        loadings_loc.T @ loadings_loc
                    )

                    # Now we compute the scores!
                    t_new = _nan_mult(data, loadings_loc, nan_mask)
                else:
                    loadings_loc = (data.T @ t_old) / (t_old.T @ t_old)
                    loadings_loc = loadings_loc / np.sqrt(
                        loadings_loc.T @ loadings_loc
                    )
                    t_new = data @ loadings_loc
                    t_new = t_new / (loadings_loc.T @ loadings_loc)

                # Perform the convergence test
                score_diff = t_old - t_new

                # guard against zero norm
                den = max(np.sqrt(t_new.T @ t_new), 1e-12)

                conv_test = np.sqrt(score_diff.T @ score_diff) / den
                converged = conv_test < self.tol_criteria

                # Increase num_iter
                num_iter += 1

                # Store scores and loads
                scores[:, i : i + 1] = t_new
                loadings[:, i : i + 1] = loadings_loc

            if num_iter >= self.max_iter:
                warnings.warn(f"max_iter reached on LV {i}")

            if verbose:
                print(f"Iteration finished on LV {i}")
            # Deflate the input matrix
            data = data - t_new @ loadings_loc.T
            if verbose:
                print("Deflation Complete")

        # Store the values in self
        # fit_scores necessary for Hotellings T2 calc
        self.fit_scores = scores
        self.loadings = loadings

    def set_components(self, n_component: int, verbose: bool = False):
        """Method for setting the number of components in an
        already-constructed model. It checks to make sure that loadings
        exist for all of the set components and will fit extras if not.
        Note that in case of decreasing the number of components, previously
        fitted components are internally stored. In case you prefer a clean
        model, create a new model object and fit it with the desired number
        of components.

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

    def transform(self, X: np.ndarray, method: str = "naive") -> np.ndarray:
        """This function takes an input array and projects it based
        on a fitted model.

        Args:
            X (np.ndarray): The nxm input array in the original
                feature space to be projected.
            method (str, optional): The method to use for the projection.
                See reference listed in module docstring.
                Valid options are {'naive','projection','conditional_mean'}
                Defaults to 'naive'.

        Raises:
            NotFittedError: If model has not been fit yet (no loadings).
            ValueError: Method 'conditional_mean' is selected but fit_data is
                not available.

        Returns:
            np.ndarray: The corresponding scores.
        """
        # Check for fit having been done.
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. Consider using fit_transform."
            )

        # Extract shape of X
        n, _ = X.shape

        # Extract variables for simplicity
        num_lvs = self.n_components
        loadings = self.loadings

        scores = np.zeros((n, num_lvs))

        nan_mask = np.isnan(X)
        nan_flag = np.any(nan_mask)

        if (not nan_flag) or (method == "naive"):
            # uses approach described in section 3.1 of McGregor paper (Single
            # component projection algorithm for missing data in PCA and PLS)
            for ind_lv in range(num_lvs):
                if nan_flag:
                    scores[:, [ind_lv]] = _nan_mult(
                        X,
                        loadings[:, [ind_lv]],
                        nan_mask,
                        use_denom=True,
                    )
                else:
                    scores[:, [ind_lv]] = X @ loadings[:, [ind_lv]]
                    scores[:, [ind_lv]] = scores[:, [ind_lv]] / (
                        loadings[:, [ind_lv]].T @ loadings[:, [ind_lv]]
                    )

                # deflate input data
                X = X - scores[:, [ind_lv]] @ loadings[:, [ind_lv]].T

        elif nan_flag and (method == "projection"):
            # uses approach described in section 4 of McGregor paper
            # (Handling missing data in PCA by projection to the model plane)
            for row in range(n):
                not_null = np.invert(nan_mask[row, :])  # just for readability
                denom = np.linalg.pinv(
                    loadings[not_null, :num_lvs].T
                    @ loadings[not_null, :num_lvs]
                )  # denominator in eq. 9
                nom = (
                    loadings[not_null, :num_lvs].T @ X[row, not_null].T
                )  # nominator in eq. 9
                scores[row, :] = (denom @ nom).T

        elif nan_flag and (method == "conditional_mean"):
            # uses approach described in section 5.1 of McGregor paper
            # (Missing data replacement using conditional mean replacement)
            fit_rows, fit_cols = self.fit_data.shape
            if self.fit_data is None:
                raise ValueError(
                    "This transformation method requires calculating the"
                    + " approximate covariance matrix on the original data."
                    + " This data would be stored in property 'fit_data', but"
                    + " is not available in this model."
                )
            elif fit_cols == self.n_components:
                # the number of PCs is equal to the number of variables in
                # the data
                T = self.fit_scores
                P = self.loadings
            elif fit_cols > self.n_components:
                # if more fit_cols required
                # fit them temporarily and
                # go back to lower number
                old_components = self.n_components
                self.set_components(fit_cols)
                T = self.fit_scores
                P = self.loadings
                self.set_components(old_components)

            theta = (T.T @ T) / (fit_rows - 1)
            for row in range(n):
                is_null = nan_mask[row, :]
                not_null = np.invert(is_null)  # just for readability
                if np.any(is_null):
                    S12 = P[is_null, :] @ theta @ P[not_null, :].T
                    S22 = P[not_null, :] @ theta @ P[not_null, :].T

                    # compute an estimate for the missing data
                    z_hash = S12 @ np.linalg.pinv(S22) @ X[row, not_null].T
                    X[row, is_null] = z_hash

                scores[row, :] = (P.T @ X[row, :].T)[0 : self.n_components]

        # Give the people what they want!
        return scores

    def fit(self, X: np.ndarray, verbose: bool = False) -> NipalsPCA:
        """Fits PCA model to input data.

        Args:
            X (np.ndarray): The input data to fit on.
            verbose (bool, optional): Whether or not to print out additional
                convergence information. Defaults to False.

        Returns:
            NipalsPCA: A reference to the object.
        """
        if self.fitted_components > 0:
            raise ValueError(
                "Model Object has already been fit."
                + " Try set_components() or build a new model object."
            )
        # n is the number of observations, m the number of variables/features
        self.fit_data = np.copy(X)

        self._add_components(n_add=self.n_components, verbose=verbose)

        # Allows model = NipalsPCA().fit(data)
        return self

    def fit_transform(
        self, X: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """Fit, then transform input data.
        This function is equivalent to
        >>>> P = NipalsPCA()
        >>>> P.fit(X)
        >>>> T = P.transform(X)

        Args:
            X (np.ndarray): The The input data to fit on and
                to transform.
            verbose (bool, optional): Whether or not to print out additional
                convergence information. Defaults to False.

        Raises:
            ValueError: Model has already been fit.

        Returns:
            np.ndarray: The corresponding scores.
        """
        if not self.__sklearn_is_fitted__():
            self.fit(X, verbose=verbose)

            scores = self.fit_scores.copy()

            return scores
        else:
            raise ValueError(
                "Model has already been fit. Try transform instead."
            )

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Approximate original data from scores.

        Args:
            X (np.ndarray): An array containing the scores.

        Raises:
            NotFittedError: PCA model has not been fit yet.
            ValueError: Shape of provided scores does not match n_components
                in model.

        Returns:
            np.ndarray: The approximation of the original data.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. "
                + "Try fit() or fit_transform() instead."
            )

        # Check size of X:
        _, m = X.shape

        if m != self.n_components:
            raise ValueError(
                "X has number of columns different than number"
                + " of components in model"
            )

        # self.n_components as we may have built loadings to a larger n than
        # the current number of components.
        out_data = X @ self.loadings[:, : self.n_components].T

        return out_data

    def calc_imd(
        self,
        input_scores: Optional[np.ndarray] = None,
        input_array: Optional[np.ndarray] = None,
        metric: str = "HotellingT2",
        covariance: str = "diag",
    ) -> np.ndarray:
        """Calculate in-model distance (IMD) of observations.
        This is the distance from the center of the hyperplane
        to the projected observation.

        Args:
            input_scores (Optional[np.ndarray], optional): The scores from
                which to calculate the distance. Defaults to None.
            input_array (Optional[np.ndarray], optional): The input data in
                original space from which to calculate the distance.
                Defaults to None.
            metric (str, optional): The metric to use. Valid options are
                {'HotellingT2'}. Defaults to 'HotellingT2'.
            covariance (str, optional): Method to compute covariance. Valid
                options are {'diag', 'full', 'ledoit_wolf'}.
                Defaults to 'diag' (quick version).
                'full' uses the entire covariance matrix computed by numpy.
                'ledoit_wolf' uses the full covariance matrix computed
                by Ledoit-Wolf shrinkage.

        Raises:
            NotFittedError: Model has not been fit yet.
            ValueError: Neither scores nor input data provided.
            ValueError: Input scores are inconsistent with n_components of
                model.
            NotImplementedError: Any metric that has not yet been implemented.

        Returns:
            np.ndarray: The calculated within-model distance for each
            observation (row).
        """
        # Warnings and errors
        if not self.__sklearn_is_fitted__():
            # if not fit
            raise NotFittedError(
                "Model has not yet been fit. Try fit() or fit_transform()"
                + " instead."
            )
        elif input_array is None:
            # If Nothing Given
            if input_scores is None:
                raise ValueError("No values provided.")
        else:
            # Both inputs = warn and calc with only data
            if input_scores is not None:
                warnings.warn(
                    "Both Scores and Data are given. Operating on Data alone."
                )

        # Get scores, either calculated or given
        if input_array is not None:
            scores = self.transform(X=input_array)
        else:
            scores = input_scores

        # Calculate in-model distances
        if metric == "HotellingT2":
            # Calculate Hotelling's T2
            num_lvs_fit = self.n_components

            if scores.shape[1] != num_lvs_fit:
                raise ValueError(
                    "Input_scores have different number of columns/latent"
                    + " variables than model n_components."
                )
            else:
                fit_means = np.mean(self.fit_scores[:, :num_lvs_fit], axis=0)

                if covariance == "diag":
                    # assume diagonal covariance matrix
                    fit_vars = np.var(
                        self.fit_scores[:, :num_lvs_fit], axis=0, ddof=1
                    )
                    out_imd = np.sum(
                        (scores - fit_means) ** 2 / fit_vars, axis=1
                    ).reshape(-1, 1)
                elif covariance == "full":
                    # use full covariance matrix
                    out_imd = np.diagonal(
                        (scores - fit_means)
                        @ np.linalg.pinv(
                            np.cov(self.fit_scores[:, :num_lvs_fit].T, ddof=1)
                        )
                        @ (scores - fit_means).T
                    ).reshape(-1, 1)
                elif covariance == "ledoit_wolf":
                    # compute full covariance matrix
                    # with Ledoit-Wolf shrinkage
                    lw_obj = LedoitWolf(
                        assume_centered=self.mean_centered
                    ).fit(self.fit_scores[:, :num_lvs_fit])
                    out_imd = np.diagonal(
                        (scores - fit_means)
                        @ np.linalg.pinv(lw_obj.covariance_)
                        @ (scores - fit_means).T
                    ).reshape(-1, 1)
                else:
                    raise NotImplementedError(
                        f"Covariance method {covariance} not implemented."
                        + "Possible methods are "
                        + "{'diag', 'full', 'ledoit_wolf'}."
                    )

        else:
            # If incorrect metric given
            raise NotImplementedError(
                "This metric has not been implemented. See doc."
            )

        return out_imd

    def calc_oomd(
        self, input_array: np.ndarray, metric: str = "QRes"
    ) -> np.ndarray:
        """Calculate the out-of-model distance (OOMD) of an observations.

        Args:
            input_array (np.ndarray): The data for which to calculate the
                OOMD.
            metric (str, optional): The metric to use. Valid options are
                {'Qres','DModX'}. Defaults to 'QRes'.

        Raises:
            NotImplementedError: Unknown metric.

        Returns:
            np.ndarray: The distances for the provided observations.
        """

        if metric == "QRes":
            # To not perform calc on input_array
            transform_dat = input_array.copy()

            # Preallocate output
            n, _ = input_array.shape
            out_oomd = np.zeros((n, 1))

            # Calculate scores (transform should handle error Handling)
            scores = self.transform(transform_dat)

            # Calculate Fitted Data
            modeled_data = self.inverse_transform(scores)

            # Calculate Residuals
            resids = transform_dat - modeled_data

            nan_mask = np.isnan(resids)
            not_null = ~nan_mask

            # Calculate Q_residuals as DMODX is based off of this value
            for row in range(n):
                out_oomd[row, 0] = (
                    resids[row, not_null[row, :]]
                    @ resids[row, not_null[row, :]].T
                )

        elif metric == "DModX":
            # Pull out params for weird Simca Factor
            fit_n, _ = self.fit_scores.shape
            num_lvs = self.n_components

            # Calculate the QRes
            out_oomd = self.calc_oomd(input_array, metric="QRes")

            # A0 is 1 if mean centered.
            if self.mean_centered:
                A0 = 1
            else:
                A0 = 0

            nan_mask = np.isnan(input_array)
            not_null = ~nan_mask
            K = not_null.sum(axis=1)
            factor = np.sqrt(fit_n / ((fit_n - num_lvs - A0) * (K - num_lvs)))
            out_oomd = factor.reshape(-1, 1) * np.sqrt(out_oomd)

        else:
            raise NotImplementedError("Input metric not recognized. See doc.")

        return out_oomd

    def calc_limit(
        self,
        metric: str = "HotellingT2",
        n: Optional[int] = None,
        num_lvs: Optional[int] = None,
        m: Optional[int] = None,
        alpha: float = 0.95,
    ) -> float:
        """This function calculates the limits for imd and oomd. Assumptions
        on the distribution shape underpin this calculation;  in practice
        limits should be judged by the user.

        Args:
            metric (str, optional): The metric to use. Valid options are
                {'HotellingT2','DModX'} Defaults to 'HotellingT2'.
            n (Optional[int], optional): The number of observations.
                Defaults to None, which results in the n of the fitted scores
                to be used.
            num_lvs (Optional[int], optional): The number of latent variables
                (principal components). Defaults to None, which results in
                the m of the fitted scores to be used.
            m (Optional[int], optional): The number of original features.
                Defaults to None, which results in the number of features in
                the original data/fitted loadings to be used.
            alpha (float, optional): The confidence value to use.
                Defaults to 0.95.

        Returns:
            float: The limit threshold for the given metric and
                confidence value.
        """
        # if not specified otherwise we take the values from the fitted model
        if n is None:
            n = self.fit_scores.shape[0]

        if m is None:
            # Might be best as self.loadings.shape[0]
            # so one can export a "minimal" model (-RMW)
            m = self.fit_data.shape[1]

        if num_lvs is None:
            num_lvs = self.n_components

        if metric == "HotellingT2":
            tsqcl = (
                F_dist.ppf(alpha, num_lvs, n - num_lvs)
                * num_lvs
                * (n - 1)
                / (n - num_lvs)
            )

            # note that the axes of the 'ellipsis of happiness' shown in
            # SIMCA are calculated as
            # ax_i = sqrt(s_i^2 * tsqcl(0.95, 2, n-2))
            # where s_i is the standard deviation of score i

            return tsqcl

        elif metric == "DModX":
            # the following equation were taken from the SIMCA help doc;
            # their origin is not clear.
            # A0 is 1 if mean centered.
            if self.mean_centered:
                A0 = 1
            else:
                A0 = 0

            # degrees of freedom of the model
            dof_mod = np.sqrt((n - A0 - num_lvs) * (m - num_lvs))

            M = np.min([m, 100, dof_mod])
            corr = n / (n - A0 - num_lvs)

            # degrees of freedom of the observations
            if m > dof_mod:
                dof_obs = (M + np.sqrt(m - dof_mod) - num_lvs) / corr
            else:
                dof_obs = (M - num_lvs) / corr

            if np.any(np.array([dof_mod, dof_obs, (n - A0 - num_lvs)]) < 1):
                warnings.warn(
                    "One of the factors in the calculation of the DModX limits"
                    + " was smaller than 1. This shouldn't happen."
                )

            # normalized d_crit value
            d_crit = np.sqrt(F_dist.ppf(alpha, dof_obs, dof_mod))

            return d_crit

    def __sklearn_is_fitted__(self) -> bool:
        """Determine if present model is fitted or not

        Returns:
            bool: is fitted or not
        """
        return self.fitted_components != 0

    def get_explained_variance_ratio(
        self,
        in_data: np.array = None,
    ) -> np.ndarray:
        """calculate the explained variance ratios per fitted component

        Args:
            in_data (np.array, optional):
                Alternative input data. Defaults to None.

        Raises:
            ValueError: if in_data not mean centered.

        Returns:
            np.ndarray: explained variances
        """
        if in_data is not None:
            if self._check_mean_centered(in_data):
                data = in_data
            else:
                raise ValueError("Variance input data is not mean centered.")
        else:
            data = self.fit_data

        orig_n_comp = self.n_components
        ret = np.zeros(orig_n_comp + 1)

        # compute explained variances per component
        # automatically pads a zero at position 0
        for i in range(1, orig_n_comp + 1):
            self.set_components(i)

            # compute data as per model
            sim_data = self.inverse_transform(self.transform(data))

            # compute residual variance
            resid_var = np.nanvar(data - sim_data, axis=0)

            # variance of data scaled to 1
            # average over variables
            ret[i] = np.nanmean(1 - resid_var)

        # go back to original components
        self.set_components(orig_n_comp)

        # subtract previous component
        ret = ret[1:] - ret[:-1]

        return ret

    # a bit hacky, avoid writing explained_variance_ once with arguments
    # as method and once without arguments as property
    explained_variance_ratio_ = property(get_explained_variance_ratio)
