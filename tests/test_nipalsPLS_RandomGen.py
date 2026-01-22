# -*- coding: utf-8 -*-
"""
Script to test the implementation of the NipalsPLS code compared to
PLS Toolbox and SIMCA

@author: Ryan Wall (lead), David Ochsenbein, Niels Schlusser
"""

from typing import Iterable, Tuple, Optional
from pathlib import Path
import warnings
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPLS import NipalsPLS
from conftest import nan_conc_coeff, rmse, init_scaler


@pytest.fixture(scope="module")
def data_path():
    return Path(__file__).parents[1].joinpath("data")


@pytest.fixture(scope="module")
def simca_test_data(data_path):
    return pd.read_excel(
        data_path.joinpath("randomGen_SIMCAResults.xlsx"),
        sheet_name=["YesNan_Var", "YesNan_Obs", "NoNan_Var", "NoNan_Obs"],
        header=None,
    )


@pytest.fixture(scope="module")
def plst_test_data(data_path):
    return pd.read_excel(
        data_path.joinpath("randomGen_PLSTResults.xlsx"),
        sheet_name=["YesNan_Var", "YesNan_Obs", "NoNan_Var", "NoNan_Obs"],
        header=None,
    )


@pytest.fixture(scope="module")
def spec_dat(data_path):
    return pd.read_csv(data_path.joinpath("XData.csv")).iloc[:, 1:].to_numpy()


@pytest.fixture(scope="module")
def nan_dat(data_path):
    return (
        pd.read_csv(data_path.joinpath("nanData.csv")).iloc[:, 1:].to_numpy()
    )


@pytest.fixture(scope="module")
def data_y(data_path):
    return pd.read_csv(data_path.joinpath("YData.csv")).iloc[:, [1]].to_numpy()


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


def fitted_model_pass_dat(
    x: np.array, y: np.array
) -> Tuple[NipalsPLS, StandardScaler, StandardScaler]:
    """Return PLS Model given X/Y data"""
    scaler_x, dat_x_scaled = init_scaler(x)
    scaler_y, dat_y_scaled = init_scaler(y)
    pls_model = NipalsPLS(mean_centered=True)  # pylint: disable=not-callable
    pls_model.fit(X=dat_x_scaled, y=dat_y_scaled)
    return pls_model, scaler_x, scaler_y


# test data fixtures
@pytest.fixture(scope="module")
def test_data_no_nan_simca(simca_test_data, spec_dat, data_y):
    """Fixture for No NaN SIMCA test data."""
    return {
        "name": "No NaN, SIMCA RandomGen",
        "X": spec_dat,
        "Y": data_y,
        "T": read_data(simca_test_data, "NoNan_Obs", [0, 1]),
        "P": read_data(simca_test_data, "NoNan_Var", [0, 1]),
        "b": read_data(simca_test_data, "NoNan_Var", [2]),
        "yhat": read_data(simca_test_data, "NoNan_Obs", [4]),
        "imd": (
            "HotellingT2",
            read_data(simca_test_data, "NoNan_Obs", [2]),
        ),
        "oomd": (
            "DModX",
            read_data(simca_test_data, "NoNan_Obs", [3]),
        ),
        "model": fitted_model_pass_dat(spec_dat, data_y),
    }


@pytest.fixture(scope="module")
def test_data_with_nan_simca(simca_test_data, nan_dat, data_y):
    """Fixture for Yes NaN SIMCA test data."""
    return {
        "name": "Yes NaN, SIMCA RandomGen",
        "X": nan_dat,
        "Y": data_y,
        "T": read_data(simca_test_data, "YesNan_Obs", [0, 1]),
        "P": read_data(simca_test_data, "YesNan_Var", [0, 1]),
        "b": read_data(simca_test_data, "YesNan_Var", [2]),
        "yhat": read_data(simca_test_data, "YesNan_Obs", [4]),
        "imd": (
            "HotellingT2",
            read_data(simca_test_data, "YesNan_Obs", [2]),
        ),
        "oomd": (
            "DModX",
            read_data(simca_test_data, "YesNan_Obs", [3]),
        ),
        "model": fitted_model_pass_dat(nan_dat, data_y),
    }


@pytest.fixture(scope="module")
def test_data_no_nan_plst(plst_test_data, spec_dat, data_y):
    """Fixture for No NaN PLST test data."""
    return {
        "name": "No NaN, PLST RandomGen",
        "X": spec_dat,
        "Y": data_y,
        "T": read_data(plst_test_data, "NoNan_Obs", [0, 1]),
        "P": read_data(plst_test_data, "NoNan_Var", [0, 1]),
        "b": read_data(plst_test_data, "NoNan_Var", [2]),
        "yhat": read_data(plst_test_data, "NoNan_Obs", [4]),
        "imd": (
            "HotellingT2",
            read_data(plst_test_data, "NoNan_Obs", [2]),
        ),
        "oomd": (
            "QRes",
            read_data(plst_test_data, "NoNan_Obs", [3]),
        ),
        "model": fitted_model_pass_dat(spec_dat, data_y),
    }


@pytest.fixture(scope="module")
def test_data_with_nan_plst(plst_test_data, nan_dat, data_y):
    """Fixture for Yes NaN PLST test data."""
    return {
        "name": "Yes NaN, PLST RandomGen",
        "X": nan_dat,
        "Y": data_y,
        "T": read_data(plst_test_data, "YesNan_Obs", [0, 1]),
        "P": read_data(plst_test_data, "YesNan_Var", [0, 1]),
        "b": read_data(plst_test_data, "YesNan_Var", [2]),
        "yhat": read_data(plst_test_data, "YesNan_Obs", [4]),
        "imd": (
            "HotellingT2",
            read_data(plst_test_data, "YesNan_Obs", [2]),
        ),
        "oomd": (
            "QRes",
            read_data(plst_test_data, "YesNan_Obs", [3]),
        ),
        "model": fitted_model_pass_dat(nan_dat, data_y),
    }


@pytest.fixture(scope="module")
def test_sub_funcs_setup(test_data_no_nan_simca):
    """Setup fixture for basic tests."""
    data_raw_x = test_data_no_nan_simca["X"]
    data_raw_y = test_data_no_nan_simca["Y"]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaled_data_x = scaler_x.fit_transform(data_raw_x)
    scaled_data_y = scaler_y.fit_transform(data_raw_y)
    return {
        "data_raw_x": data_raw_x,
        "data_raw_y": data_raw_x,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "scaled_data_x": scaled_data_x,
        "scaled_data_y": scaled_data_y,
    }


def test_multi_fit(test_sub_funcs_setup):
    """Test fitting twice, should throw an error"""
    x_data = test_sub_funcs_setup["scaled_data_x"]
    y_data = test_sub_funcs_setup["scaled_data_y"]

    model = NipalsPLS().fit(x_data, y_data)

    with pytest.raises(Exception) as exc_info:
        model.fit(x_data, y_data)

    assert "Model Object has already been fit." in str(exc_info.value), (
        "Should raise error when fitting twice"
    )
    assert model.n_components == 2, "n_components should be 2"


def test_is_fitted(test_sub_funcs_setup):
    """Test the __sklearn_is_fitted__() method"""
    x_data = test_sub_funcs_setup["scaled_data_x"]
    y_data = test_sub_funcs_setup["scaled_data_y"]

    model = NipalsPLS()
    assert model.__sklearn_is_fitted__() is False, (
        "Unfitted model should return False"
    )

    model = NipalsPLS().fit(x_data, y_data)
    assert model.__sklearn_is_fitted__(), "Fitted model should return True"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
def test_fit_scores(get_data, request):
    """Compare fitted scores to scores from package (T)"""
    test_data_dict = request.getfixturevalue(get_data)

    T = test_data_dict["T"]
    model = test_data_dict["model"][0]

    test_val = rmse(T, model.fit_scores_x)
    lin_val = nan_conc_coeff(T, model.fit_scores_x)

    if get_data == "test_data_with_nan_plst":
        err_lim = 5e-2
    elif get_data == "test_data_no_nan_plst":
        err_lim = 5e-3
    else:
        err_lim = 5e-5

    # overall rmse is low
    assert test_val < err_lim, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning"
)
def test_score_method_equivalence(get_data, request):
    """Calculate scores and compare to fitted scores"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]
    scaler_x = test_data_dict["model"][1]
    in_data = test_data_dict["X"]

    py_calc_scores = model.transform(scaler_x.transform(in_data))

    test_val = rmse(model.fit_scores_x, py_calc_scores)
    lin_val = nan_conc_coeff(model.fit_scores_x, py_calc_scores)

    # overall rmse is low
    assert test_val < 1e-9, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-9, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
def test_fit_loadings(get_data, request):
    """Compare loadings to loadings from package (P)"""
    test_data_dict = request.getfixturevalue(get_data)

    P = test_data_dict["P"]
    model = test_data_dict["model"]

    test_val = rmse(P, model[0].loadings_x)
    lin_val = nan_conc_coeff(P, model[0].loadings_x)

    if get_data == "test_data_with_nan_plst":
        err_lim = 5e-5
    elif get_data == "test_data_no_nan_plst":
        err_lim = 5e-6
    else:
        err_lim = 5e-8

    # overall rmse is low
    assert test_val < err_lim, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
def test_fit_y(get_data, request):
    """Calculate predictions and compare to predictions from package"""
    test_data_dict = request.getfixturevalue(get_data)

    T = test_data_dict["T"]
    yhat = test_data_dict["yhat"]
    model = test_data_dict["model"]

    py_y_vals = model[0].predict(scores_x=T)
    py_y_vals = model[2].inverse_transform(
        py_y_vals
    )  # reverse standardscaling
    test_val = rmse(yhat, py_y_vals)
    lin_val = nan_conc_coeff(yhat, py_y_vals)

    # tolerances per dataset
    if get_data == "test_data_with_nan_plst":
        err_lim = 5e-2
    elif get_data == "test_data_no_nan_plst":
        err_lim = 5e-3
    else:
        err_lim = 5e-4

    # overall rmse is low
    assert test_val < err_lim, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
def test_fit_reg(get_data, request):
    """Calculate regression vector and compare to package"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]  # A reference for convenience
    b = test_data_dict["b"] / np.linalg.norm(
        test_data_dict["b"]
    )  # Normalize to 1

    py_reg_vects = model.get_reg_vector()
    py_reg_vects = py_reg_vects / np.linalg.norm(py_reg_vects)  # Normalize
    test_val = rmse(b, py_reg_vects)
    lin_val = nan_conc_coeff(b, py_reg_vects)

    if get_data == "test_data_with_nan_plst":
        err_lim = 5e-5
    else:
        err_lim = 5e-7

    # overall rmse is low
    assert test_val < err_lim, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
def test_imd(get_data, request):
    """Test the in-model distance"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]
    metric, known_imd = test_data_dict["imd"]

    test_imd = model.calc_imd(input_scores=model.fit_scores_x, metric=metric)
    test_val = rmse(test_imd, known_imd)
    lin_val = nan_conc_coeff(test_imd, known_imd)

    # tolerances per dataset
    if get_data == "test_data_with_nan_plst":
        err_lim = 5e-3
    else:
        err_lim = 5e-6

    assert test_val < err_lim, f"rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning"
)
def test_oomd(get_data, request):
    """Test the out-of-model distance"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]
    scaler_x = test_data_dict["model"][1]
    in_data = test_data_dict["X"]
    metric, known_oomd = test_data_dict["oomd"]

    transformed_data = scaler_x.transform(in_data)

    test_oomd = model.calc_oomd(transformed_data, metric=metric)
    test_val = rmse(test_oomd, known_oomd)
    lin_val = nan_conc_coeff(test_oomd, known_oomd)

    # tolerances per dataset
    if get_data == "test_data_with_nan_plst":
        err_lim = 5e-1
    else:
        err_lim = 5e-3

    assert test_val < err_lim, f"rmse = {test_val}"
    assert lin_val > 1 - 1e-2, f"linConc = {lin_val}"


def _run_set_component_test(test_data):
    """Helper function to run set_component tests"""
    model = test_data["model"][0]
    scaler_x = test_data["model"][1]
    scaler_y = test_data["model"][2]
    X = test_data["X"]
    Y = test_data["Y"]
    name = test_data["name"]
    T = test_data["T"]
    P = test_data["P"]
    imd = test_data["imd"]
    oomd = test_data["oomd"]

    transformed_data_x = scaler_x.transform(X)
    transformed_data_y = scaler_y.transform(Y)

    model_low = NipalsPLS(n_components=1)
    model_low.fit(transformed_data_x, transformed_data_y)

    # Update to new amount of components
    num_lvs = model.n_components
    model_low.set_components(num_lvs)

    # tolerances per dataset
    if name == "Yes NaN, PLST RandomGen":
        err_lim_scores = 5e-2
        err_lim_loadings = 5e-5
        err_lim_predict = 5e-2
        err_lim_imd = 5e-3
        err_lim_oomd = 5e-1
    elif name == "No NaN, PLST RandomGen":
        err_lim_scores = 5e-3
        err_lim_loadings = 5e-6
        err_lim_predict = 5e-3
        err_lim_imd = 5e-6
        err_lim_oomd = 5e-3
    else:
        err_lim_scores = 5e-5
        err_lim_loadings = 5e-8
        err_lim_predict = 5e-4
        err_lim_imd = 5e-6
        err_lim_oomd = 5e-3

    # compare X scores
    test_val = rmse(model.fit_scores_x, model_low.fit_scores_x)
    lin_val = nan_conc_coeff(model.fit_scores_x, model_low.fit_scores_x)

    # overall rmse is low, correlation is very high
    assert test_val < err_lim_scores, f"Inc comps scores rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"Inc comps scores linConc = {lin_val}"

    # compare X loadings
    test_val = rmse(model.loadings_x, model_low.loadings_x)
    lin_val = nan_conc_coeff(model.loadings_x, model_low.loadings_x)

    assert test_val < err_lim_loadings, f"Inc comps load rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"Inc comps load linConc = {lin_val}"

    # compare Y predictions
    py_y_vals = model.predict(scores_x=model.fit_scores_x)
    py_y_vals_low = model_low.predict(scores_x=model_low.fit_scores_x)
    test_val = rmse(py_y_vals, py_y_vals_low)
    lin_val = nan_conc_coeff(py_y_vals, py_y_vals_low)

    assert test_val < err_lim_predict, f"Inc comps preds rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"Inc comps preds linConc = {lin_val}"

    # Add an extra component, drop back down
    model_low.set_components(num_lvs + 1)
    model_low.set_components(num_lvs)

    # compare X scores
    test_val = rmse(T, model_low.fit_scores_x[:, :num_lvs])
    lin_val = nan_conc_coeff(T, model_low.fit_scores_x[:, :num_lvs])

    assert test_val < err_lim_scores, f"Dec comps scores rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"Dec comps scores linConc = {lin_val}"

    # compare X loadings
    test_val = rmse(P, model_low.loadings_x[:, :num_lvs])
    lin_val = nan_conc_coeff(P, model_low.loadings_x[:, :num_lvs])

    assert test_val < err_lim_loadings, f"Dec comps load rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"Dec comps load linConc = {lin_val}"

    # compare Y predictions
    py_y_vals = model.predict(scores_x=model.fit_scores_x)
    py_y_vals_low = model_low.predict(
        scores_x=model_low.fit_scores_x[:, :num_lvs]
    )
    test_val = rmse(py_y_vals, py_y_vals_low)
    lin_val = nan_conc_coeff(py_y_vals, py_y_vals_low)

    assert test_val < err_lim_predict, f"Dec comps preds rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"Dec comps preds linConc = {lin_val}"

    # Now show that calc_imd/calc_oomd function the same
    metric, known_imd = imd
    test_imd = model_low.calc_imd(
        input_scores=model_low.fit_scores_x[:, :num_lvs], metric=metric
    )
    test_val = rmse(test_imd, known_imd)
    lin_val = nan_conc_coeff(test_imd, known_imd)

    assert test_val < err_lim_imd, f"IMD rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"IMD linConc = {lin_val}"

    metric, known_oomd = oomd
    test_oomd = model_low.calc_oomd(transformed_data_x, metric=metric)
    test_val = rmse(known_oomd, test_oomd)
    lin_val = nan_conc_coeff(test_oomd, known_oomd)

    assert test_val < err_lim_oomd, f"OOMD rmse = {test_val}"
    assert lin_val > 1 - 1e-2, f"OOMD linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning"
)
def test_set_component(get_data, request):
    """Test the set_component function"""
    test_data_dict = request.getfixturevalue(get_data)
    _run_set_component_test(test_data_dict)


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan_simca",
        "test_data_with_nan_simca",
        "test_data_no_nan_plst",
        "test_data_with_nan_plst",
    ],
)
def test_explained_variance_ratio(get_data, request):
    """Test explained variance ratio"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ex_var_rat_x, ex_var_rat_y = model.explained_variance_ratio_

    # check if in [0,1]
    assert np.all(ex_var_rat_x >= 0) and np.all(ex_var_rat_x <= 1), (
        "Explained X variance ratio must be in [0,1]"
    )
    assert np.all(ex_var_rat_y >= 0) and np.all(ex_var_rat_y <= 1), (
        "Explained y variance ratio must be in [0,1]"
    )

    # check if normalized
    assert sum(ex_var_rat_x) <= 1, (
        "Explained X variance ratio must sum to >= 1"
    )
    assert sum(ex_var_rat_y) <= 1, (
        "Explained y variance ratio must sum to >= 1"
    )

    # check if strictly decreasing
    diffs_x = np.diff(ex_var_rat_x)
    diffs_y = np.diff(ex_var_rat_y)

    assert np.all(diffs_x < 0), "Explained X variance ratio must be decreasing"
    assert np.all(diffs_y < 0), "Explained y variance ratio must be decreasing"
