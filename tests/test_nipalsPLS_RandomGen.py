# -*- coding: utf-8 -*-
"""
Script to test the implementation of the NipalsPLS code compared to
PLS Toolbox and SIMCA

@author: Ryan Wall (lead), David Ochsenbein, Niels Schlusser
"""

import unittest
from parameterized import parameterized_class
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from typing import Iterable, Tuple, Optional
from pathlib import Path
import warnings

from open_nipals.nipalsPLS import NipalsPLS


# Load Data
path = Path(__file__).parents[1].joinpath("data")

simca_test_data = pd.read_excel(
    path.joinpath(r"randomGen_SIMCAResults.xlsx"),
    sheet_name=["YesNan_Var", "YesNan_Obs", "NoNan_Var", "NoNan_Obs"],
    header=None,
)
plst_test_data = pd.read_excel(
    path.joinpath(r"randomGen_PLSTResults.xlsx"),
    sheet_name=["YesNan_Var", "YesNan_Obs", "NoNan_Var", "NoNan_Obs"],
    header=None,
)

spec_dat = pd.read_csv(path.joinpath(r"XData.csv")).iloc[:, 1:].to_numpy()
nan_dat = pd.read_csv(path.joinpath(r"nanData.csv")).iloc[:, 1:].to_numpy()
data_Y = pd.read_csv(path.joinpath(r"YData.csv")).iloc[:, [1]].to_numpy()


def nan_conc_coeff(y: np.array, yhat: np.array) -> float:
    """Calculate the Lin's Concordance Coefficient, a
    linearity metric that shows how close to 1:1 a line is"""
    # Note that using the standard numpy var,cov, etc caused some
    # weird errors. could get correlations of 1.001001 etc.
    nan_mask = np.isnan(y)
    nan_mask_yhat = np.isnan(yhat)
    new_y = y[np.invert(nan_mask)].copy()
    new_yhat = yhat[np.invert(nan_mask_yhat)].copy()

    # averages
    ybar = np.mean(new_y)
    yhatbar = np.mean(new_yhat)

    # variances
    sy = np.sum((new_y - ybar) ** 2) / len(new_y)
    syhat = np.sum((new_yhat - yhatbar) ** 2) / len(new_yhat)
    syyhat = np.sum((new_y - ybar) * (new_yhat - yhatbar)) / len(
        new_y
    )  # covariance

    numer = 2 * syyhat
    denom = sy + syhat + (ybar - yhatbar) ** 2
    return numer / denom


def rmse(y: np.array, yhat: np.array) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y - yhat) ** 2))


def read_data(
    in_df: pd.DataFrame,
    sheet_name: str,
    specific_columns: Optional[Iterable[str]] = None,
) -> np.array:
    """Read Data from specific sheets/columns"""
    my_data = in_df[sheet_name].to_numpy()

    if specific_columns is not None:
        my_data = my_data[:, specific_columns]

    if len(my_data.shape) == 1:
        my_data = my_data.reshape(-1, 1)

    return my_data


def init_scaler(dat: np.array) -> Tuple[StandardScaler, np.array]:
    scaler = StandardScaler()
    scaler.fit(dat)
    scaler.scale_ = np.nanstd(
        dat, axis=0, ddof=1
    )  # standardscaler uses biased variance, but we want unbiased estimator

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        transformed_data = scaler.transform(dat)
        # throws unproblematic: "RuntimeWarning: invalid value encountered in divide"

    return scaler, transformed_data


def fitted_model_pass_dat(
    x: np.array, y: np.array
) -> Tuple[NipalsPLS, StandardScaler, StandardScaler]:
    """Return PLS Model given X/Y data"""
    scaler_x, dat_x_scaled = init_scaler(x)
    scaler_y, dat_y_scaled = init_scaler(y)
    pls_model = NipalsPLS(mean_centered=True)  # pylint: disable=not-callable
    pls_model.fit(dat_x_scaled, dat_y_scaled)
    return pls_model, scaler_x, scaler_y


def class_name(cls, num, params_dict: dict) -> str:
    return f"{params_dict['name']}"


class TestSubFuncs(unittest.TestCase):
    """This is a wrap of the quicker to test things that should check for
    bad behavior or improper use cases"""

    def setUp(self) -> None:
        mod, scaler_x, scaler_y = fitted_model_pass_dat(spec_dat, data_Y)
        self.model = mod
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.data_x = spec_dat
        self.data_y = data_Y
        return super().setUp()

    def test_multiFit(self):
        """Test fitting a second time, should throw an error"""
        with self.assertRaises(BaseException) as e:
            self.model.fit(self.data_x, self.data_y)

            with self.subTest():
                self.assertIn(
                    "Model Object has already been fit.", str(e.exception)
                )
            with self.subTest():
                self.assertEqual(self.model.n_components, 2)

    def test_is_fitted(self):
        """Test the __sklearn_is_fitted__() method"""
        model = NipalsPLS()
        with self.subTest():
            self.assertFalse(model.__sklearn_is_fitted__())

        with self.subTest():
            self.assertTrue(self.model.__sklearn_is_fitted__())


@parameterized_class(
    [
        {
            "name": "No NaN, SIMCA RandomGen",
            "X": spec_dat,
            "Y": data_Y,  # first column (0) contains indices, hence ignore it
            "T": read_data(simca_test_data, "NoNan_Obs", [0, 1]),
            "P": read_data(simca_test_data, "NoNan_Var", [0, 1]),
            "b": read_data(simca_test_data, "NoNan_Var", [2]),
            "yhat": read_data(simca_test_data, "NoNan_Obs", [4]),
            "imd": (
                "HotellingT2",
                read_data(simca_test_data, "NoNan_Obs", [2]),
            ),  # tuple to define metric
            "oomd": (
                "DModX",
                read_data(simca_test_data, "NoNan_Obs", [3]),
            ),  # tuple to define metric
            "model": fitted_model_pass_dat(spec_dat, data_Y),
        },
        {
            "name": "Yes NaN, SIMCA RandomGen",
            "X": nan_dat,
            "Y": data_Y,
            "T": read_data(simca_test_data, "YesNan_Obs", [0, 1]),
            "P": read_data(simca_test_data, "YesNan_Var", [0, 1]),
            "b": read_data(simca_test_data, "YesNan_Var", [2]),
            "yhat": read_data(simca_test_data, "YesNan_Obs", [4]),
            "imd": (
                "HotellingT2",
                read_data(simca_test_data, "YesNan_Obs", [2]),
            ),  # tuple to define metric
            "oomd": (
                "DModX",
                read_data(simca_test_data, "YesNan_Obs", [3]),
            ),  # tuple to define metric
            "model": fitted_model_pass_dat(nan_dat, data_Y),
        },
        {
            "name": "No NaN, PLST RandomGen",
            "X": spec_dat,
            "Y": data_Y,
            "T": read_data(plst_test_data, "NoNan_Obs", [0, 1]),
            "P": read_data(plst_test_data, "NoNan_Var", [0, 1]),
            "b": read_data(plst_test_data, "NoNan_Var", [2]),
            "yhat": read_data(plst_test_data, "NoNan_Obs", [4]),
            "imd": (
                "HotellingT2",
                read_data(plst_test_data, "NoNan_Obs", [2]),
            ),  # tuple to define metric
            "oomd": (
                "QRes",
                read_data(plst_test_data, "NoNan_Obs", [3]),
            ),  # tuple to define metric
            "model": fitted_model_pass_dat(spec_dat, data_Y),
        },
        {
            "name": "Yes NaN, PLST RandomGen",
            "X": nan_dat,
            "Y": data_Y,
            "T": read_data(plst_test_data, "YesNan_Obs", [0, 1]),
            "P": read_data(plst_test_data, "YesNan_Var", [0, 1]),
            "b": read_data(plst_test_data, "YesNan_Var", [2]),
            "yhat": read_data(plst_test_data, "YesNan_Obs", [4]),
            "imd": (
                "HotellingT2",
                read_data(plst_test_data, "YesNan_Obs", [2]),
            ),  # tuple to define metric
            "oomd": (
                "QRes",
                read_data(plst_test_data, "YesNan_Obs", [3]),
            ),  # tuple to define metric
            "model": fitted_model_pass_dat(nan_dat, data_Y),
        },
    ],
    class_name_func=class_name,
)
class TestFit(unittest.TestCase):
    def test_fit_scores(
        self,
    ):  # self is an object that contains (among other things) the fields indicates in the dicts passed to parameterized_class
        """Compare fitted scores to scores from package (T)"""
        test_val = rmse(self.T, self.model[0].fit_scores_x)
        lin_val = nan_conc_coeff(self.T, self.model[0].fit_scores_x)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-1
        else:
            err_lim = 1e-2
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_score_method_equivalence(self):
        """Calculate scores and compare to fitted scores.
        Basically ensures that a new dataset coming in would result in the same projection as the original training set. Surprisingly non-obvious."""
        model = self.model[0]
        scaler_x = self.model[1]
        in_data = self.X

        # nan data can throw "invalid value encountered in divide"
        if "NaN" in self.name:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                py_calc_scores = model.transform(scaler_x.transform(in_data))
        else:
            py_calc_scores = model.transform(scaler_x.transform(in_data))

        test_val = rmse(model.fit_scores_x, py_calc_scores)
        lin_val = nan_conc_coeff(model.fit_scores_x, py_calc_scores)

        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-9, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-6, msg=f"linConc = {lin_val}")

    def test_fit_loadings(self):
        """Compare loadings to loadings from package (P)"""
        test_val = rmse(self.P, self.model[0].loadings_x)
        lin_val = nan_conc_coeff(self.P, self.model[0].loadings_x)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-3, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_fit_Y(self):
        """Calculate predictions and compare to predictions from package.
        Prediction is done using the scores from the reference. This way,
        the prediction step is decoupled from the scores calculation step.
        """
        py_y_vals = self.model[0].predict(scores_x=self.T)
        py_y_vals = self.model[2].inverse_transform(
            py_y_vals
        )  # reverse standardscaling
        test_val = rmse(self.yhat, py_y_vals)
        lin_val = nan_conc_coeff(self.yhat, py_y_vals)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-1
        else:
            err_lim = 1e-2
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_fit_reg(self):
        """Calculate regression vector and compare to package"""
        model = self.model[0]  # A reference for convenience
        b = self.b / np.linalg.norm(self.b)  # Normalize to 1

        py_reg_vects = model.get_reg_vector()
        py_reg_vects = py_reg_vects / np.linalg.norm(py_reg_vects)  # Normalize
        test_val = rmse(b, py_reg_vects)
        lin_val = nan_conc_coeff(b, py_reg_vects)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-3, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_imd(self):
        """Test the in-model distance"""
        model = self.model[0]
        metric, known_imd = self.imd
        test_imd = model.calc_imd(
            input_scores=model.fit_scores_x, metric=metric
        )
        test_val = rmse(test_imd, known_imd)
        lin_val = nan_conc_coeff(test_imd, known_imd)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-2
        else:
            err_lim = 1e-5
        with self.subTest():
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_oomd(self):
        """Test the out-of-model distance"""
        model = self.model[0]
        scaler_x = self.model[1]
        in_data = self.X
        metric, known_oomd = self.oomd

        # nan data can throw "invalid value encountered in divide"
        if "NaN" in self.name:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transformed_data = scaler_x.transform(in_data)
        else:
            transformed_data = scaler_x.transform(in_data)

        test_oomd = model.calc_oomd(transformed_data, metric=metric)
        test_val = rmse(test_oomd, known_oomd)
        lin_val = nan_conc_coeff(test_oomd, known_oomd)

        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 0.5
        else:
            err_lim = 1e-2

        with self.subTest():
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            self.assertGreater(lin_val, 1 - 1e-2, msg=f"linConc = {lin_val}")

    # test for set_component
    def test_set_component(self):
        """Test the set_component function"""
        model = self.model[0]
        scaler_x = self.model[1]
        scaler_y = self.model[2]

        # nan data can throw "invalid value encountered in divide"
        if "NaN" in self.name:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transformed_data_x = scaler_x.transform(self.X)
                transformed_data_y = scaler_y.transform(self.Y)
        else:
            transformed_data_x = scaler_x.transform(self.X)
            transformed_data_y = scaler_y.transform(self.Y)

        model_low = NipalsPLS(n_components=1)
        model_low.fit(transformed_data_x, transformed_data_y)

        # Update to new amount of components
        num_lvs = model.n_components
        model_low.set_components(num_lvs)

        # compare X scores
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-1
        else:
            err_lim = 1e-2
        test_val = rmse(model.fit_scores_x, model_low.fit_scores_x)
        lin_val = nan_conc_coeff(model.fit_scores_x, model_low.fit_scores_x)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # compare X loadings
        test_val = rmse(model.loadings_x, model_low.loadings_x)
        lin_val = nan_conc_coeff(model.loadings_x, model_low.loadings_x)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-3, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # compare Y predictions
        py_y_vals = model.predict(scores_x=model.fit_scores_x)
        py_y_vals_low = model_low.predict(scores_x=model_low.fit_scores_x)
        test_val = rmse(py_y_vals, py_y_vals_low)
        lin_val = nan_conc_coeff(py_y_vals, py_y_vals_low)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-1
        else:
            err_lim = 1e-2
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # compare regression vectors
        py_reg_vects = model.get_reg_vector()
        py_reg_vects = py_reg_vects / np.linalg.norm(py_reg_vects)  # Normalize

        py_reg_vects_low = py_reg_vects / np.linalg.norm(
            py_reg_vects
        )  # Normalize

        test_val = rmse(py_reg_vects, py_reg_vects_low)
        lin_val = nan_conc_coeff(py_reg_vects, py_reg_vects_low)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-3, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # Add an extra component, drop back down
        model_low.set_components(num_lvs + 1)
        model_low.set_components(num_lvs)

        # compare X scores
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-1
        else:
            err_lim = 1e-2
        test_val = rmse(self.T, model_low.fit_scores_x[:, :num_lvs])
        lin_val = nan_conc_coeff(self.T, model_low.fit_scores_x[:, :num_lvs])
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # compare X loadings
        test_val = rmse(self.P, model_low.loadings_x[:, :num_lvs])
        lin_val = nan_conc_coeff(self.P, model_low.loadings_x[:, :num_lvs])
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-3, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # compare Y predictions
        py_y_vals = model.predict(scores_x=model.fit_scores_x)
        py_y_vals_low = model_low.predict(
            scores_x=model_low.fit_scores_x[:, :num_lvs]
        )
        test_val = rmse(py_y_vals, py_y_vals_low)
        lin_val = nan_conc_coeff(py_y_vals, py_y_vals_low)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-1
        else:
            err_lim = 1e-2
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # compare regression vectors
        py_reg_vects = model.get_reg_vector()
        py_reg_vects = py_reg_vects / np.linalg.norm(py_reg_vects)  # Normalize

        py_reg_vects_low = py_reg_vects / np.linalg.norm(
            py_reg_vects
        )  # Normalize

        test_val = rmse(py_reg_vects, py_reg_vects_low)
        lin_val = nan_conc_coeff(py_reg_vects, py_reg_vects_low)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-3, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        # Now show that calc_imd/calc_oomd function the same
        metric, known_imd = self.imd
        test_imd = model_low.calc_imd(
            input_scores=model_low.fit_scores_x[:, :num_lvs], metric=metric
        )
        test_val = rmse(test_imd, known_imd)
        lin_val = nan_conc_coeff(test_imd, known_imd)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 1e-2
        else:
            err_lim = 1e-5
        with self.subTest():
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

        metric, known_oomd = self.oomd
        test_oomd = model_low.calc_oomd(transformed_data_x, metric=metric)
        test_val = rmse(known_oomd, test_oomd)
        lin_val = nan_conc_coeff(test_oomd, known_oomd)
        if self.name == "Yes NaN, PLST RandomGen":
            err_lim = 0.5
        else:
            err_lim = 1e-2
        with self.subTest():
            self.assertLess(test_val, err_lim, msg=f"rmse = {test_val}")
        with self.subTest():
            self.assertGreater(lin_val, 1 - 1e-2, msg=f"linConc = {lin_val}")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=3).run(suite)
