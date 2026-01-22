# pylint: disable=no-member
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from open_nipals.utils import ArrangeData


@pytest.fixture(scope="module")
def data_path():
    """Fixture for the data directory path."""
    return Path(__file__).parents[1].joinpath("data")


@pytest.fixture(scope="module")
def pca_data(data_path):
    """Fixture for loading PCA test data."""
    return pd.read_excel(
        data_path.joinpath("PCATestData.xlsx"), header=None, engine="openpyxl"
    )


@pytest.fixture(scope="module")
def pca_array(pca_data):
    """Fixture for PCA data as numpy array."""
    return pca_data.to_numpy()


@pytest.fixture
def arr_data_obj(pca_data):
    """Fixture for fitted ArrangeData object."""
    arr_dat_obj = ArrangeData()
    arr_dat_obj.fit(pca_data)
    return arr_dat_obj


@pytest.fixture
def arrange_data_setup(pca_data, pca_array, arr_data_obj):
    """Setup fixture for ArrangeData tests."""
    return {
        "dataframe": pca_data,
        "array": pca_array,
        "arr_dat_obj": arr_data_obj,
    }


def test_transform(arrange_data_setup):
    """Test that the transform method yields the same
    array as the to_numpy() array of the same original df"""
    test_array = arrange_data_setup["arr_dat_obj"].transform(
        arrange_data_setup["dataframe"]
    )

    assert np.all(test_array == arrange_data_setup["array"])


def test_fit_transform(arrange_data_setup):
    """Test that the fit_transform method yields the same
    array as the to_numpy() array of the same original df"""
    arr_dat_obj = ArrangeData()
    test_array = arr_dat_obj.fit_transform(arrange_data_setup["dataframe"])

    assert np.all(test_array == arrange_data_setup["array"])


def test_rearranged(arrange_data_setup):
    """Test that after fit/transform one can rearrange an array and
    still end up with the identical results so long as column names
    are the same"""
    reverse_df = arrange_data_setup["dataframe"].loc[:, ::-1]
    test_array = arrange_data_setup["arr_dat_obj"].transform(reverse_df)

    assert np.all(test_array == arrange_data_setup["array"])


def test_extra_columns(arrange_data_setup):
    """Test that if extra columns are present, ArrangeData will drop
    and still end up with the identical results"""
    extra_df = arrange_data_setup["dataframe"].copy()
    extra_df["foo"] = np.nan

    # expected to throw warning because variable name dictionaries mismatch
    with pytest.warns(Warning):
        test_array = arrange_data_setup["arr_dat_obj"].transform(extra_df)

    assert np.all(test_array == arrange_data_setup["array"])
