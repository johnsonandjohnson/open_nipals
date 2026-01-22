# pylint: disable=no-member
from typing import Tuple
from pathlib import Path
import pytest
import warnings
import pandas as pd
import numpy as np
from open_nipals.nipalsPCA import NipalsPCA
from sklearn.preprocessing import StandardScaler
from conftest import nan_conc_coeff, rmse, init_scaler


@pytest.fixture(scope="module")
def data_path():
    """Fixture for the data directory path."""
    return Path(__file__).parents[1].joinpath("data")


@pytest.fixture(scope="module")
def pca_input(data_path):
    """Fixture for input array from PCA data."""
    df_input = pd.read_excel(
        data_path.joinpath("PCATestData.xlsx"), header=None, engine="openpyxl"
    )
    return df_input.to_numpy()


@pytest.fixture(scope="module")
def simca_loads(data_path):
    """Fixture for SIMCA loadings data."""
    df_simca_loads = pd.read_excel(
        data_path.joinpath("SIMCA_ScaledFullDat_Loadings.xlsx"),
        engine="openpyxl",
        usecols=[1, 2],
    )  # First column is garbage index
    return df_simca_loads.to_numpy()


@pytest.fixture(scope="module")
def simca_scores_data(data_path):
    """Fixture for SIMCA scores, T2, and DModX data."""
    df_simca_sample_dat = pd.read_excel(
        data_path.joinpath("SIMCA_ScaledFullDat_Scores_T2Range_DMODXAbs.xlsx"),
        engine="openpyxl",
    )
    simca_scores = df_simca_sample_dat.to_numpy()[
        :, 1:3
    ]  # First column is garbage index
    simca_t2 = df_simca_sample_dat.to_numpy()[:, 3:4]
    simca_dmodx_abs = df_simca_sample_dat.to_numpy()[:, 4:5]
    return simca_scores, simca_t2, simca_dmodx_abs


@pytest.fixture(scope="module")
def pca_input_nan(data_path):
    """Fixture for NaN test data."""
    df_pca_input_nan = pd.read_excel(
        data_path.joinpath("PCANanData.xlsx"), header=None, engine="openpyxl"
    )
    return df_pca_input_nan.to_numpy()


@pytest.fixture(scope="module")
def simca_loads_nan(data_path):
    """Fixture for SIMCA loadings with NaN data."""
    df_simca_loads = pd.read_excel(
        data_path.joinpath("SIMCA_ScaledNaNDat_Loadings.xlsx"),
        engine="openpyxl",
        usecols=[1, 2],
    )  # First column is garbage index
    return df_simca_loads.to_numpy()


@pytest.fixture(scope="module")
def simca_scores_nan_data(data_path):
    """Fixture for SIMCA scores, T2, and DModX data with NaN."""
    df_simca_sample_dat = pd.read_excel(
        data_path.joinpath("SIMCA_ScaledNaNDat_Scores_T2Range_DMODXAbs.xlsx"),
        engine="openpyxl",
    )
    simca_scores_nan = df_simca_sample_dat.to_numpy()[
        :, 1:3
    ]  # First column is garbage index
    simca_t2_nan = df_simca_sample_dat.to_numpy()[:, 3:4]
    simca_dmodx_abs_nan = df_simca_sample_dat.to_numpy()[:, 4:5]
    return simca_scores_nan, simca_t2_nan, simca_dmodx_abs_nan


def fitted_model_pass_dat(x: np.ndarray) -> Tuple[NipalsPCA, StandardScaler]:
    """Return PCA Model given X data"""
    scaler_x, dat_x_scaled = init_scaler(x)

    pca_model = NipalsPCA(mean_centered=True)  # pylint: disable=not-callable
    pca_model.fit(X=dat_x_scaled)
    return pca_model, scaler_x


@pytest.fixture(scope="module")
def test_sub_funcs_setup(pca_input):
    """Setup fixture for basic tests."""
    data_raw = pca_input
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_input)
    return {"data_raw": data_raw, "scaler": scaler, "data_scaled": scaled_data}


# test data fixtures
@pytest.fixture(scope="module")
def test_data_no_nan(pca_input, simca_scores_data, simca_loads):
    """Fixture for no NaN test data."""
    simca_scores, simca_t2, simca_dmodx_abs = simca_scores_data
    return {
        "name": "No NaN, SIMCA",
        "X": pca_input,
        "T": simca_scores,
        "P": simca_loads,
        "imd": ("HotellingT2", simca_t2),
        "oomd": ("DModX", simca_dmodx_abs),
        "model": fitted_model_pass_dat(pca_input),
    }


@pytest.fixture(scope="module")
def test_data_with_nan(pca_input_nan, simca_scores_nan_data, simca_loads_nan):
    """Fixture for NaN test data."""
    simca_scores_nan, simca_t2_nan, simca_dmodx_abs_nan = simca_scores_nan_data
    return {
        "name": "Yes NaN, SIMCA",
        "X": pca_input_nan,
        "T": simca_scores_nan,
        "P": simca_loads_nan,
        "imd": ("HotellingT2", simca_t2_nan),
        "oomd": ("DModX", simca_dmodx_abs_nan),
        "model": fitted_model_pass_dat(pca_input_nan),
    }


def test_multi_fit(test_sub_funcs_setup):
    """Test fitting twice, should throw an error"""
    model = NipalsPCA().fit(test_sub_funcs_setup["data_scaled"])

    with pytest.raises(Exception) as exc_info:
        model.fit(test_sub_funcs_setup["data_scaled"])

    assert "Model Object has already been fit." in str(exc_info.value), (
        "Should raise error when fitting twice"
    )
    assert model.n_components == 2, "n_components should be 2"


def test_is_fitted(test_sub_funcs_setup):
    """Test the __sklearn_is_fitted__() method"""
    model = NipalsPCA()
    assert model.__sklearn_is_fitted__() is False, (
        "Unfitted model should return False"
    )

    model = NipalsPCA().fit(test_sub_funcs_setup["data_scaled"])
    assert model.__sklearn_is_fitted__(), "Fitted model should return True"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_fit_scores(get_data, request):
    """Compare fitted scores to scores from package (T)"""
    test_data_dict = request.getfixturevalue(get_data)

    T = test_data_dict["T"]
    model = test_data_dict["model"][0]

    test_val = rmse(T, model.fit_scores)
    lin_val = nan_conc_coeff(T, model.fit_scores)

    # overall rmse is low
    assert test_val < 1e-2, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_score_method_equivalence(get_data, request):
    """Calculate scores and compare to fitted scores"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]
    scaler_x = test_data_dict["model"][1]
    input_data = test_data_dict["X"]

    py_calc_scores = model.transform(scaler_x.transform(input_data))
    test_val = rmse(model.fit_scores, py_calc_scores)
    lin_val = nan_conc_coeff(model.fit_scores, py_calc_scores)

    # overall rmse is low
    assert test_val < 1e-9, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-6, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_fit_loadings(get_data, request):
    """Compare loadings to loadings from package (P)"""
    test_data_dict = request.getfixturevalue(get_data)

    P = test_data_dict["P"]
    model = test_data_dict["model"]

    test_val = rmse(P, model[0].loadings)
    lin_val = nan_conc_coeff(P, model[0].loadings)

    # overall rmse is low
    assert test_val < 1e-3, f"rmse = {test_val}"
    # correlation is very high
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_imd(get_data, request):
    """Test the in-model distance"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]
    metric, known_imd = test_data_dict["imd"]

    test_imd = model.calc_imd(input_scores=model.fit_scores, metric=metric)
    test_val = rmse(test_imd, known_imd)
    lin_val = nan_conc_coeff(test_imd, known_imd)

    assert test_val < 1e-4, f"rmse = {test_val}"
    assert lin_val > 1 - 1e-5, f"linConc = {lin_val}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_oomd(get_data, request):
    """Test the out-of-model distance"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]
    scaler_x = test_data_dict["model"][1]
    input_data = test_data_dict["X"]
    metric, known_oomd = test_data_dict["oomd"]

    transformed_data = scaler_x.transform(input_data)
    test_oomd = model.calc_oomd(transformed_data, metric=metric)
    test_val = rmse(test_oomd, known_oomd)
    lin_val = nan_conc_coeff(test_oomd, known_oomd)

    assert test_val < 1e-2, f"rmse = {test_val}"
    assert lin_val > 1 - 1e-2, f"linConc = {lin_val}"


def _run_set_component_test(test_data):
    """Helper function to run set_component tests"""
    model = test_data["model"][0]
    scaler_x = test_data["model"][1]
    input_data = test_data["X"]
    transformed_data = scaler_x.transform(input_data)

    model_low = NipalsPCA(n_components=1)
    model_low.fit(transformed_data)

    # Update to new amount of components
    num_lvs = model.n_components
    model_low.set_components(num_lvs)

    # Demonstrate loadings/scores are the same
    max_score_diff = np.max(np.abs(model.fit_scores - model_low.fit_scores))
    max_load_diff = np.max(np.abs(model.loadings - model_low.loadings))

    assert max_score_diff < 1e-9, f"Inc comps score diff = {max_score_diff}"
    assert max_load_diff < 1e-9, f"Inc comps load diff = {max_load_diff}"

    # Add an extra component, drop back down
    model_low.set_components(num_lvs + 1)
    model_low.set_components(num_lvs)

    # Same loadings/scores test to ensure they didn't change
    max_score_diff = np.max(
        np.abs(model.fit_scores - model_low.fit_scores[:, :num_lvs])
    )
    max_load_diff = np.max(
        np.abs(model.loadings - model_low.loadings[:, :num_lvs])
    )

    assert max_score_diff < 1e-9, f"Decr comps score diff = {max_score_diff}"
    assert max_load_diff < 1e-9, f"Decr comps load diff = {max_load_diff}"

    # Now show that calc_imd/calc_oomd function the same
    known_imd = model.calc_imd(input_array=model.fit_data)
    test_imd = model_low.calc_imd(input_array=model_low.fit_data)
    max_imd_diff = np.max(np.abs(known_imd - test_imd))

    known_oomd = model.calc_oomd(model.fit_data)
    test_oomd = model_low.calc_oomd(model_low.fit_data)
    max_oomd_diff = np.max(np.abs(known_oomd - test_oomd))

    assert max_imd_diff < 1e-9, f"Max IMD diff = {max_imd_diff}"
    assert max_oomd_diff < 1e-9, f"Max OOMD diff = {max_oomd_diff}"


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_set_component(get_data, request):
    """Test the set_component function"""
    test_data_dict = request.getfixturevalue(get_data)
    _run_set_component_test(test_data_dict)


@pytest.mark.parametrize(
    "get_data",
    [
        "test_data_no_nan",
        "test_data_with_nan",
    ],
)
def test_explained_variance_ratio(get_data, request):
    """test explained variance ratio calculation"""
    test_data_dict = request.getfixturevalue(get_data)

    model = test_data_dict["model"][0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ex_var_ratio = model.explained_variance_ratio_

    # check if in [0,1]
    assert np.all(ex_var_ratio >= 0) and np.all(ex_var_ratio <= 1), (
        "Explained variance ratio must be in [0,1]"
    )

    # check if normalized
    assert sum(ex_var_ratio) <= 1.0, (
        "Explained variance ratio must sum to <= 1"
    )

    # check if strictly decreasing
    diffs = np.diff(ex_var_ratio)
    assert np.all(diffs <= 0), "Explained variance ratio must be decreasing"
