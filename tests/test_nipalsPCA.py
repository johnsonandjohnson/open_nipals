# pylint: disable=no-member
import unittest
from parameterized import parameterized_class
import pandas as pd
import numpy as np

from open_nipals.nipalsPCA import NipalsPCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from pathlib import Path

import sys

# Load Data
path = Path(__file__).parents[1].joinpath("data")

pca_dat = pd.read_excel(
    path.joinpath("PCATestData.xlsx"), header=None, engine="openpyxl"
)
input_array = pca_dat.to_numpy()

df_simca_loads = pd.read_excel(
    path.joinpath("SIMCA_ScaledFullDat_Loadings.xlsx"),
    engine="openpyxl",
    usecols=[1, 2],
)  # First column is garbage index
simca_loads = df_simca_loads.to_numpy()

df_simca_sample_dat = pd.read_excel(
    path.joinpath("SIMCA_ScaledFullDat_Scores_T2Range_DMODXAbs.xlsx"),
    engine="openpyxl",
)
simca_scores = df_simca_sample_dat.to_numpy()[
    :, 1:3
]  # First column is garbage index
simca_T2 = df_simca_sample_dat.to_numpy()[:, 3:4]
simca_DmodX_abs = df_simca_sample_dat.to_numpy()[:, 4:5]

# NAN data
nan_dat = pd.read_excel(
    path.joinpath("PCANanData.xlsx"), header=None, engine="openpyxl"
)
input_array_nan = nan_dat.to_numpy()

df_simca_loads = pd.read_excel(
    path.joinpath("SIMCA_ScaledNanDat_Loadings.xlsx"),
    engine="openpyxl",
    usecols=[1, 2],
)  # First column is garbage index
simca_loads_nan = df_simca_loads.to_numpy()

df_simca_sample_dat = pd.read_excel(
    path.joinpath("SIMCA_ScaledNanDat_Scores_T2Range_DMODXAbs.xlsx"),
    engine="openpyxl",
)
simca_scores_nan = df_simca_sample_dat.to_numpy()[
    :, 1:3
]  # First column is garbage index
simca_T2_nan = df_simca_sample_dat.to_numpy()[:, 3:4]
simca_DmodX_abs_nan = df_simca_sample_dat.to_numpy()[:, 4:5]


def nan_conc_coeff(y: np.ndarray, yhat: np.ndarray) -> float:
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


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    y = y.ravel()
    yhat = yhat.ravel()

    return np.sqrt(np.mean((y - yhat) ** 2))


def init_scaler(dat: np.ndarray) -> Tuple[StandardScaler, np.array]:
    scaler = StandardScaler()
    scaler.fit(dat)
    scaler.scale_ = np.nanstd(
        dat, axis=0, ddof=1
    )  # standardscaler uses biased variance, but we want unbiased estimator

    return scaler, scaler.transform(dat)


def fitted_model_pass_dat(x: np.ndarray) -> Tuple[NipalsPCA, StandardScaler]:
    """Return PCA Model given X data"""
    scaler_x, dat_x_scaled = init_scaler(x)

    pca_model = NipalsPCA(mean_centered=True)  # pylint: disable=not-callable
    pca_model.fit(X=dat_x_scaled)
    return pca_model, scaler_x


def class_name(cls, num, params_dict: dict) -> str:
    return f"{params_dict['name']}"


class TestSubFuncs(unittest.TestCase):
    """This is a wrap of the quicker to test things that should check for
    bad behavior or improper use cases"""

    def setUp(self) -> None:
        self.data_raw = input_array
        scalar = StandardScaler()
        scaled_data = scalar.fit_transform(input_array)
        self.scalar = scalar
        self.data_scaled = scaled_data
        return super().setUp()

    def test_multiFit(self):
        """Test fitting twice, should throw an error"""
        model = NipalsPCA().fit(self.data_scaled)

        with self.assertRaises(BaseException) as e:
            model.fit(X=self.data_scaled)

            with self.subTest():
                self.assertIn(
                    "Model Object has already been fit.", str(e.exception)
                )
            with self.subTest():
                self.assertEqual(model.n_components, 2)
    
    def test_is_fitted(self):
        """Test the __sklearn_is_fitted__() method"""
        model = NipalsPCA()
        with self.subTest():
            self.assertFalse(model.__sklearn_is_fitted__())

        model = NipalsPCA().fit(self.data_scaled)
        with self.subTest():
            self.assertTrue(model.__sklearn_is_fitted__())




@parameterized_class(
    [
        {
            "name": "No NaN, SIMCA",
            "X": input_array,
            "T": simca_scores,
            "P": simca_loads,
            "imd": ("HotellingT2", simca_T2),  # tuple to define metric
            "oomd": ("DModX", simca_DmodX_abs),  # tuple to define metric
            "model": fitted_model_pass_dat(input_array),
        },
        {
            "name": "Yes NaN, SIMCA",
            "X": input_array_nan,
            "T": simca_scores_nan,
            "P": simca_loads_nan,
            "imd": ("HotellingT2", simca_T2_nan),  # tuple to define metric
            "oomd": ("DModX", simca_DmodX_abs_nan),  # tuple to define metric
            "model": fitted_model_pass_dat(input_array_nan),
        },
    ],
    class_name_func=class_name,
)
class TestFit(unittest.TestCase):
    # self is an object that contains (among other things) the fields
    # indicated in the dicts passed to parameterized_class
    def test_fit_scores(
        self,
    ):
        """Compare fitted scores to scores from package (T)"""
        test_val = rmse(self.T, self.model[0].fit_scores)
        lin_val = nan_conc_coeff(self.T, self.model[0].fit_scores)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-2, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_score_method_equivalence(self):
        """Calculate scores and compare to fitted scores.
        Basically ensures that a new dataset coming in would result in
        the same projection as the original training set.
        Surprisingly non-obvious."""
        model = self.model[0]
        scaler_x = self.model[1]
        input_data = self.X
        py_calc_scores = model.transform(X=scaler_x.transform(input_data))
        test_val = rmse(model.fit_scores, py_calc_scores)
        lin_val = nan_conc_coeff(model.fit_scores, py_calc_scores)
        with self.subTest():
            # overall rmse is low
            self.assertLess(test_val, 1e-9, msg=f"rmse = {test_val}")
        with self.subTest():
            # correlation is very high
            self.assertGreater(lin_val, 1 - 1e-6, msg=f"linConc = {lin_val}")

    def test_fit_loadings(self):
        """Compare loadings to loadings from package (P)"""
        test_val = rmse(self.P, self.model[0].loadings)
        lin_val = nan_conc_coeff(self.P, self.model[0].loadings)
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
        test_imd = model.calc_imd(input_scores=model.fit_scores, metric=metric)
        test_val = rmse(test_imd, known_imd)
        lin_val = nan_conc_coeff(test_imd, known_imd)

        with self.subTest():
            self.assertLess(test_val, 1e-4, msg=f"rmse = {test_val}")
        with self.subTest():
            self.assertGreater(lin_val, 1 - 1e-5, msg=f"linConc = {lin_val}")

    def test_oomd(self):
        """Test the out-of-model distance"""
        model = self.model[0]
        scaler_x = self.model[1]
        input_data = self.X
        metric, known_oomd = self.oomd
        transformed_data = scaler_x.transform(input_data)
        test_oomd = model.calc_oomd(
            input_array=transformed_data, metric=metric
        )
        test_val = rmse(test_oomd, known_oomd)
        lin_val = nan_conc_coeff(test_oomd, known_oomd)

        with self.subTest():
            self.assertLess(test_val, 1e-2, msg=f"rmse = {test_val}")
        with self.subTest():
            self.assertGreater(lin_val, 1 - 1e-2, msg=f"linConc = {lin_val}")

    def test_set_component(self):
        """Test the set_component function"""
        model = self.model[0]
        scaler_x = self.model[1]
        input_data = self.X
        transformed_data = scaler_x.transform(input_data)
        model_low = NipalsPCA(n_components=1)
        model_low.fit(transformed_data)
        # Update to new amount of components
        num_lvs = model.n_components
        model_low.set_components(n_component=num_lvs)

        # Demonstrate loadings/scores are the same
        max_score_diff = np.max(
            np.abs(model.fit_scores - model_low.fit_scores)
        )
        max_load_diff = np.max(np.abs(model.loadings - model_low.loadings))

        with self.subTest():
            self.assertLess(
                max_score_diff,
                1e-9,
                msg=f"Lower add score diff = {max_score_diff}",
            )
        with self.subTest():
            self.assertLess(
                max_load_diff,
                1e-9,
                msg=f"Lower add load diff = {max_load_diff}",
            )

        # Add an extra component, drop back down
        model_low.set_components(n_component=num_lvs + 1)
        model_low.set_components(n_component=num_lvs)

        # Same loadings/scores test to ensure they didn't change
        max_score_diff = np.max(
            np.abs(model.fit_scores - model_low.fit_scores[:, :num_lvs])
        )
        max_load_diff = np.max(
            np.abs(model.loadings - model_low.loadings[:, :num_lvs])
        )

        with self.subTest():
            self.assertLess(
                max_score_diff,
                1e-9,
                msg=f"Greater add score diff = {max_score_diff}",
            )
        with self.subTest():
            self.assertLess(
                max_load_diff,
                1e-9,
                msg=f"Greater add load diff = {max_load_diff}",
            )

        # Now show that calc_imd/calc_oomd function the same
        known_imd = model.calc_imd(input_array=model.fit_data)
        test_imd = model_low.calc_imd(input_array=model_low.fit_data)
        max_imd_diff = np.max(np.abs(known_imd - test_imd))

        known_oomd = model.calc_oomd(input_array=model.fit_data)
        test_oomd = model_low.calc_oomd(input_array=model_low.fit_data)
        max_oomd_diff = np.max(np.abs(known_oomd - test_oomd))

        with self.subTest():
            self.assertLess(
                max_imd_diff, 1e-9, msg=f"Max imd Diff = {max_imd_diff}"
            )
        with self.subTest():
            self.assertLess(
                max_oomd_diff, 1e-9, msg=f"Max oomd Diff = {max_oomd_diff}"
            )


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=3).run(suite)
