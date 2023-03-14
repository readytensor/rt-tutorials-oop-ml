import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import label_binarize


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects or drops specified columns."""
    def __init__(self, columns, selector_type='keep'):
        """
        Initializes a new instance of the `ColumnSelector` class.

        Args:
            columns : list of str
                List of column names to select or drop.
            selector_type : str, optional (default='keep')
                Type of selection. Must be either 'keep' or 'drop'.
        """
        self.columns = columns
        assert selector_type in ["keep", "drop"]
        self.selector_type = selector_type

    def fit(self, X, y=None):
        """
        Fits the transformer.

        Args:
            X: pandas DataFrame - the input data
            y: unused

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Applies the column selection.

        Args:
            X : pandas.DataFrame - The input data.

        Returns:
            pandas.DataFrame: The transformed data.
        """
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]
            X = X.drop(dropped_cols, axis=1)
        return X


class ValueClipper(BaseEstimator, TransformerMixin):
    """Clips the values of the specified fields to a specified range."""
    def __init__(self, fields_to_clip, min_val, max_val) -> None:
        """
        Initializes a new instance of the `ValueClipper` class.

        Args:
            fields_to_clip : list of str
                List of field names to clip.
            min_val : float or None, optional (default=None)
                Minimum value of the range. If None, the values are not clipped from the lower end.
            max_val : float or None, optional (default=None)
                Maximum value of the range. If None, the values are not clipped from the upper end.

        """
        super().__init__()
        self.fields_to_clip = fields_to_clip
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data):
        """
        Fit function that does nothing.

        Args:
            data : pandas.DataFrame
                The input data.

        Returns:
            ValueClipper - This instance of the ValueClipper class.
        """
        return self

    def transform(self, data):
        """
        Clips the values of the specified fields to the specified range.
        
        Args:
            data: pandas.DataFrame - The input data.

        Returns:
            pandas.DataFrame
                The transformed data.

        """
        for field in self.fields_to_clip:
            if self.min_val is not None:
                data[field] = data[field].clip(lower=self.min_val)
            if self.max_val is not None:
                data[field] = data[field].clip(upper=self.max_val)
        return data


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using the most frequently observed class for categorical features when missing values are rare (under 10% of samples). """
    def __init__(self, cat_vars, threshold):
        """
        Initializes a new instance of the `MostFrequentImputer` class.
        
        Args:
            cat_vars : list of str
                List of the categorical features to impute.
            threshold : float, optional (default=1)
                The minimum proportion of the samples that must contain a missing value for the imputation to be performed.

        """
        self.cat_vars = cat_vars
        self.threshold = threshold
        self.fill_vals = {}

    def fit(self, X, y=None):
        """
        Fits the transformer.

        Args:
            X: pandas DataFrame - the input data
            y: unused
        Returns:
            self
        """
        self.fitted_cat_vars = [
            var for var in self.cat_vars
            if var in X.columns and X[var].isnull().mean() <  self.threshold ]

        for col in self.fitted_cat_vars:
            self.fill_vals[col] = X[col].value_counts().index[0]
        return self

    def transform(self, X, y=None):
        """
        Transform the data by imputing the most frequent class for the fitted categorical features.

        Args:
            X: pandas DataFrame - The data to transform.
            y: unused
        Returns:
            pandas DataFrame - The transformed data with the most frequent class imputed for the fitted categorical features.
        """
        for col in self.fill_vals:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_vals[col])
        return X


class OneHotEncoderMultipleCols(BaseEstimator, TransformerMixin):
    """ Encodes categorical features using one-hot encoding. """
    def __init__(self, ohe_columns, max_num_categories=10):
        """
        Initializes a new instance of the `OneHotEncoderMultipleCols` class.

        Args:
            ohe_columns : list of str
                List of the categorical features to one-hot encode.
            max_num_categories : int, optional (default=10)
                Maximum number of categories to include for each feature.
        """
        super().__init__()
        self.ohe_columns = ohe_columns
        self.max_num_categories = max_num_categories
        self.top_cat_by_ohe_col = {}

    def fit(self, X, y=None):
        """
        Learns the values to be used for one-hot encoding from the input data X.

        Args:
            X : pandas DataFrame
                Data to learn one-hot encoding from.
            y : unused

        Returns:
            self : OneHotEncoderMultipleCols
        """
        for col in self.ohe_columns:
            if col in X.columns:
                self.top_cat_by_ohe_col[col] = [
                    cat for cat in X[col].value_counts()\
                        .sort_values(ascending = False).head(self.max_num_categories).index
                    ]
        return self

    def transform(self, data):
        """
        Encodes the input data using the learned values.

        Args:
            data : pandas DataFrame - Data to one-hot encode.

        Returns:
            transformed_data : pandas DataFrame - One-hot encoded data.
        """
        data.reset_index(inplace=True, drop=True)
        df_list = [data]
        cols_list = list(data.columns)
        for col in self.ohe_columns:
            if len(self.top_cat_by_ohe_col[col]) > 0:
                if col in data.columns:
                    for cat in self.top_cat_by_ohe_col[col]:
                        col_name = col + '_' + cat
                        vals = np.where(data[col] == cat, 1, 0)
                        df = pd.DataFrame(vals, columns=[col_name])
                        df_list.append(df)

                        cols_list.append(col_name)
                else:
                    raise Exception(f'''
                        Error: Fitted one-hot-encoded column {col}
                        does not exist in dataframe given for transformation.
                        This will result in a shape mismatch for train/prediction job.
                        ''')
        transformed_data = pd.concat(df_list, axis=1, ignore_index=True)
        transformed_data.columns =  cols_list
        return transformed_data


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    """ Binarizes the target variable to 0/1 values. """
    def __init__(self, target_field, target_class) -> None:
        """
        Initializes a new instance of the `CustomLabelBinarizer` class.

        Parameters:
        -----------
        :target_field: str
            Name of the target field.
        :target_class: str
            Name of the target class.
        """
        super().__init__()
        self.target_field = target_field
        self.target_class = target_class
        self.given_classes = None

    def fit(self, data):
        """
        Fit the transformer.

        Args:
            data: pandas DataFrame - input data
        Returns:
            self
        """
        # grab the two classes
        given_classes = data[self.target_field].drop_duplicates().tolist()
        # sort so that the target class is last
        given_classes.sort(key = lambda k: k == self.target_class)
        # save for transformation
        self.given_classes = given_classes
        return self

    def transform(self, data):
        """
        Transform the data.

        Args:
            data: pandas DataFrame - data to transform
        Returns:
            pandas DataFrame - transformed data
        """
        if self.target_field in data.columns:
            data[self.target_field] = label_binarize(
                data[self.target_field],
                classes = self.given_classes
            ).flatten()
        return data