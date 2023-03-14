import os
import warnings
import joblib
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


class Classifier:
    """A wrapper class for the Random Forest binary classifier.

    This class provides a consistent interface that can be used with other classifier models.
    The Random Forest binary classifier is encapsulated inside this class.
    """

    model_name = "random_forest_binary_classifier"
    model_fname = "random_forest_binary_classifier.save"

    def __init__(self, n_estimators=100, min_samples_split=2, min_samples_leaf=1, **kwargs):
        """Construct a new Random Forest binary classifier.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
            min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
        """
        self.n_estimators = int(n_estimators)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.model = self.build_model()

    def build_model(self):
        """Build a new Random Forest binary classifier."""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        return model

    def fit(self, train_X, train_y):
        """Fit the Random Forest binary classifier to the training data.

        Args:
            train_X (pandas.DataFrame): The features of the training data.
            train_y (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_X, train_y)

    def predict(self, X):
        """Predict class labels for the given data.

        Args:
            X (pandas.DataFrame): The input data.

        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for the given data.

        Args:
            X (pandas.DataFrame): The input data.

        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def evaluate(self, x_test, y_test):
        """Evaluate the Random Forest binary classifier and return the accuracy.

        Args:
            x_test (pandas.DataFrame): The features of the test data.
            y_test (pandas.Series): The labels of the test data.

        Returns:
            float: The accuracy of the Random Forest binary classifier.
        """
        if self.model is not None:
            return self.model.score(x_test, y_test)

    def save(self, model_path):
        """Save the Random Forest binary classifier to disk.

        Args:
            model_path (str): The path to save the model to.
        """
        joblib.dump(self, os.path.join(model_path, Classifier.model_fname))

    @classmethod
    def load(cls, model_path):
        """Load the Random Forest binary classifier from disk.

        Args:
            model_path (str): The path to the saved model.

        Returns:
            Classifier: A new instance of the loaded Random Forest binary classifier.
        """
        model = joblib.load(os.path.join(model_path, Classifier.model_fname))
        return model
