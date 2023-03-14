import numpy as np, pandas as pd

import data_management.pipeline as pipeline
from algorithm.classifier import Classifier


class ModelServer:
    """
    Class for making batch or online predictions using a trained classifier.

    """

    def __init__(self, model_path, data_schema):
        """
        Initializes a new instance of the `ModelServer` class
        
        Args:
            model_path (str): The path to the directory containing the trained model artifacts.
            data_schema (BinaryClassificationSchema): An instance of the BinaryClassificationSchema class that defines the dataset schema.
        """

        self.model_path = model_path
        self.preprocessor = None
        self.label_encoder = None
        self.model = None
        self.data_schema = data_schema

    def _get_preprocessor_and_lbl_encoder(self):
        """
        Internal function to load the preprocessor and label encoder from disk and return them.

        Returns:
            preprocessor (sklearn.Pipeline): The loaded preprocessing pipeline.
            label_encoder (sklearn.LabelEncoder): The loaded label encoder.
        """
        if self.preprocessor is None:
            self.preprocessor, self.label_encoder = pipeline.load_preprocessor_and_lbl_encoder(self.model_path)

        return self.preprocessor, self.label_encoder

    def _get_model(self):
        """
        Internal function to load the trained classifier model from disk and return it.

        Returns:
            model (Classifier): The loaded classifier model.
        """
        if self.model is None:
            self.model = Classifier.load(self.model_path)
        return self.model

    def _get_predictions(self, data):
        """
        Internal function to make batch predictions on the input data.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.

        Returns:
            preds (numpy.ndarray): The predicted class probabilities.
        """
        preprocessor, _ = self._get_preprocessor_and_lbl_encoder()
        model = self._get_model()

        # transform data
        transformed_data = preprocessor.transform(data)

        # Grab input features for prediction
        feature_cols = [c for c in transformed_data.columns if c not in [self.data_schema.id_field, self.data_schema.target_field]]

        # make predictions
        preds = model.predict_proba(transformed_data[feature_cols])
        return preds

    def predict_proba(self, data):
        """
        Function to make batch predictions on the input data and return predicted class probabilities.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.

        Returns:
            preds_df (pandas.DataFrame): A pandas DataFrame with the input data ids and the predicted class probabilities.
        """
        preds = self._get_predictions(data)
        class_names = pipeline.get_class_names(self.label_encoder)
        preds_df = data[[self.data_schema.id_field]].copy()
        preds_df[class_names] = np.round(preds, 5)
        return preds_df

    def predict(self, data):
        """
        Function to make batch predictions on the input data and return predicted classes.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.

        Returns:
            preds_df (pandas.DataFrame): A pandas DataFrame with the input data ids and the predicted classes.
        """
        class_names = pipeline.get_class_names(self.label_encoder)
        preds_df = data[[self.data_schema.id_field]].copy()
        preds_df["prediction"] = pd.DataFrame(
            self.predict_proba(data), columns=class_names
        ).idxmax(axis=1)
        return preds_df