# pylint: disable=no-member
import unittest
import pandas as pd
import numpy as np
import sys

from open_nipals.utils import ArrangeData
from pathlib import Path


class TestArrDatObj(unittest.TestCase):
    """Test the ArrangeData function."""

    def setUp(self):
        # Load Data
        path = Path(__file__).parents[1].joinpath("data")
        pca_dat = pd.read_excel(
            path.joinpath(r"PCATestData.xlsx"), header=None, engine="openpyxl"
        )
        arr_dat_obj = ArrangeData()
        arr_dat_obj.fit(pca_dat)

        self.dataframe = pca_dat
        self.array = pca_dat.to_numpy()
        self.arr_dat_obj = arr_dat_obj

    def test_transform(self):
        """Test that the transform method yields the same
        array as the to_numpy() array of the same original df"""
        test_array = self.arr_dat_obj.transform(self.dataframe)

        self.assertTrue(np.all(test_array == self.array))

    def test_fit_transform(self):
        """Test that the fit_transform method yields the same
        array as the to_numpy() array of the same original df"""
        arr_dat_obj = ArrangeData()
        test_array = arr_dat_obj.fit_transform(self.dataframe)

        self.assertTrue(np.all(test_array == self.array))

    def test_rearranged(self):
        """Test that after fit/transform one can rearrange an array and
        still end up with the identical results so long as column names
        are the same"""
        reverse_df = self.dataframe.loc[:, ::-1]
        test_array = self.arr_dat_obj.transform(reverse_df)

        self.assertTrue(np.all(test_array == self.array))

    def test_extra_columns(self):
        """Test that if extra columns are present, ArrangeData will drop
        and still end up with the identical results"""
        extra_df = self.dataframe.copy()
        extra_df["foo"] = np.nan

        # expected to throw warning because variable name dictionaries mismatch
        with self.assertWarns(Warning):
            test_array = self.arr_dat_obj.transform(extra_df)

        self.assertTrue(np.all(test_array == self.array))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=3).run(suite)
