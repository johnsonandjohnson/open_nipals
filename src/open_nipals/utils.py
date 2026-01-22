"""
Code for pipelining a data arrangement in sklearn. Allows for standarscaler()
followed by other methods

(c) 2020: Ryan Wall, revised 2025: Calvin Ristad, Niels Schlusser
"""

import warnings
from typing import Optional, Union
import numpy as np
from sklearn.base import TransformerMixin
import pandas as pd


def _nan_mult(
    x: np.array,
    y: np.array,
    nan_mask: Optional[np.array] = None,
    use_denom: bool = True,
) -> np.ndarray:
    """Matrix multiplication for when the left matrix (X) contains NaNs.

    Args:
        x (np.array): The left matrix in the multiplication.
        y (np.array): The right matrix in the multiplication.
        nan_mask (np.array, optional): The nan_mask for the left matrix.
            Defaults to None.
        use_denom (bool, optional): Flag for using a normalization method.
            Defaults to True. Scales the resultant vector by 1/(y.T @ y)
            with appropriate nulls

    Raises:
        ValueError: If nan_mask is passed but has the wrong shape.

    Returns:
        np.array: The product of the matrix multiplication.
    """
    N, M = x.shape

    if nan_mask is None:
        nan_mask = np.isnan(x)
    else:
        # Catch error for if nan_mask isn't rotated correctly
        rows_nan, _ = nan_mask.shape
        if N != rows_nan:
            raise ValueError("nan_mask not rotated correctly")

    y_was_1d = y.ndim == 1  # if y was already 1 dim

    if y_was_1d:
        y2d = y.reshape(M, 1)  # (M,1)
    else:
        y2d = y

    x_nomiss = x.copy()
    x_nomiss[nan_mask] = 0.0

    numer = x_nomiss @ y2d  # (N,M) x (M,1) => (N,1)

    if not use_denom:
        out = numer
    else:
        obs = (~nan_mask).astype(x.dtype)  # (N,M)
        y_sq = (y2d**2).astype(x.dtype)  # (M,1)

        denom = obs @ y_sq  # (N,M) x (M,1) => (N,1)

        out = np.divide(
            numer, denom, out=np.zeros_like(numer), where=(denom != 0)
        )  # (N,1)

    if y_was_1d:
        out = out.ravel()

    return out


class ArrangeData(TransformerMixin):
    """ArrangeData class
    creates a sklearn-style transformer object that orders the columns
    in dataframe correctly

    Attributes:
    --------------------
    var_dict : dict
        A dictionary indicating which column should be in which position.

    Methods:
    --------------------
    var_dict_from_df
        Infer var_dict from template dataframe.
    fit_transform
        concatenation of fit and transform
    fit
        Applies var_dict_from_df in a consistent manner with sklearn
        nomenclature
    transform
        Takes a new dataframe or np.array + variable dictionary
        and arranges to be consistent with stored var_dict
    """

    def __init__(self, var_dict: Union[dict, pd.DataFrame] = None):
        """The constructor for the ArrangeData object.

        Args:
            var_dict (Union[dict, pd.DataFrame], optional): The variable
                dictionary to use. Can be inferred from a template
                dataframe on construction or by  Defaults to None.
        """
        if isinstance(var_dict, pd.DataFrame):
            self.var_dict = self.var_dict_from_df(var_dict)
        else:
            self.var_dict = var_dict

    def var_dict_from_df(self, input_df: pd.DataFrame) -> dict:
        """Infer the var_dict from a template dataframe.

        Args:
            input_df (pd.DataFrame): A dataframe that has the desired
                order of columns.

        Returns:
            dict: The corresponding var_dict. {column Name: column index}
        """
        # Takes an input dataframe and turns column headers into a
        # variable dictionary
        var_dict = {c: i for i, c in enumerate(input_df.columns)}
        return var_dict

    def fit_transform(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        input_var_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Fit a data/column model and then transform data.

        Args:
            input_data (Union[pd.DataFrame, np.ndarray]): The input data
                to transform. Note that if this is a dataframe it _will_
                be used to fit var_dict. This might not be what you want.
            input_var_dict (Optional[dict], optional): The var_dict to be used
                to fit. Defaults to None.

        Returns:
            pd.DataFrame: transformed input frame
        """
        self.fit(input_data=input_data, input_var_dict=input_var_dict)
        out_data = self.transform(input_data, input_var_dict=self.var_dict)
        return out_data

    def fit(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        input_var_dict: Optional[dict] = None,
    ):
        """The function which will fit the var_dict object.

        Args:
            input_data (Union[pd.DataFrame, np.ndarray]): The input data to
                use for fitting.
            input_var_dict (Optional[dict], optional): Required if input_data
                is an array, this dictionary contains the headers as
                {column Name: column index}.

        Raises:
            ValueError: input_data is a numpy array and input_var_dict is missing.
            ValueError: input_data is a numpy array and its shape does not
                match input_var_dict.
            ValueError: An unknown error occurred.
        """
        if isinstance(input_data, pd.DataFrame):
            if self.var_dict is not None:
                warnings.warn(
                    "Variable Dictionary in Dataframe is overwriting"
                    + " that specified in constructor."
                )

            # Returns the Variable Dictionary from a Dataframe
            self.var_dict = self.var_dict_from_df(input_data)

        elif (isinstance(input_data, np.ndarray)) and (input_var_dict is None):
            raise ValueError(
                "Input dataframe is an array but no variable Dictionary has"
                " been provided"
            )

        elif isinstance(input_data, np.ndarray) and isinstance(
            input_var_dict, dict
        ):
            # input_data is an array and input_var_dict has been provided.
            if len(input_var_dict) != input_data.shape[1]:
                raise ValueError(
                    "input_var_dict incorrect size for Data Array."
                )
            else:
                # simply override with input_var_dict
                self.var_dict = input_var_dict

        else:
            # catch if, e.g., input_data is None or wrong type.
            raise ValueError("An error has occurred in fit() of ArrangeData")

    def transform(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        input_var_dict: Optional[dict] = None,
    ) -> np.ndarray:
        """Transform input data based on stored data model.

        Args:
            input_data (Union[pd.DataFrame, np.ndarray]): The input_data to
                transform.
            input_var_dict (Optional[dict], optional): Required if input_data
                is an array, this dictionary contains the headers as
                {column Name: column index}.. Defaults to None.

        Raises:
            ValueError: ArrangeData object has not yet been fit.
            ValueError: input_data is a numpy array and no input_var_dict has
                been provided.
            ValueError: input_data is a numpy array and its shape does not
                match input_var_dict

        Returns:
            np.ndarray: The transformed (column-rearranged) data.
        """
        # Check for fit having been done.
        if self.var_dict is None:
            raise ValueError("ArrangeData() has not been fit")

        # If dataframe input, convert to Numpy Array and get var_dict from
        # col headers
        if isinstance(input_data, pd.DataFrame):
            input_var_dict = self.var_dict_from_df(input_data)
            input_data = input_data.to_numpy()
        elif isinstance(input_data, np.ndarray):
            if input_var_dict is None:
                raise ValueError(
                    "input_data is a numpy array and no input_var_dict given."
                )

            if input_data.shape[1] != len(input_var_dict):
                raise ValueError(
                    "Given Variable Dictionary not the same length as number"
                    + " of variables"
                )

        # Extract shape of input_data
        # n is the number of observations, m the number of variables/features
        (
            n,
            _,
        ) = input_data.shape

        # Extract m of fit variables
        fit_m = len(self.var_dict)

        diff_set = set(input_var_dict.keys()) - set(self.var_dict.keys())
        if len(diff_set) > 0:
            warnings.warn(
                f"Given Variable Dictionary contains {len(diff_set)} variable"
                + f" names that are not known to the model: {diff_set}"
            )

        # Preallocate out_data filled w/ NaNs
        out_data = np.ndarray(shape=(n, fit_m))
        out_data.fill(np.nan)

        # Loop over each input_data Column and put it in out_data Array
        for var_name, var_index in input_var_dict.items():
            # If variable is in the stored dictionary
            if var_name in self.var_dict:
                col_index = self.var_dict[var_name]  # Slightly easier to read
                out_data[:, col_index] = input_data[:, var_index]

        return out_data
