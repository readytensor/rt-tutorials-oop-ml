from typing import List


class BinaryClassificationSchema:
    """
    A class for loading a binary classification schema file and providing methods to access
    the fields defined in the schema.
    """

    def __init__(self, schema_dict: dict) -> None:
        """
        Initializes a new instance of the `BinaryClassificationSchema` class
        and loads the schema file.

        Args:
            schema_dict (dict): The python dictionary of schema.
        """
        self.schema = schema_dict
        self._numeric_features = self._get_features_of_type("NUMERIC", "INT", "REAL")
        self._categorical_features = self._get_features_of_type("CATEGORICAL")

    def _get_features_of_type(self, *types) -> List[str]:
        """
        Returns the feature names of the specified data type.

        Args:
            *types (str): The data types of the features.

        Returns:
            List(str): The list of feature names.
        """
        fields = self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["predictorFields"]
        return [f["fieldName"] for f in fields if f["dataType"] in types]

    @property
    def id_field(self) -> str:
        """
        Gets the name of the ID field.

        Returns:
            str: The name of the ID field.
        """
        return self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]

    @property
    def target_field(self) -> str:
        """
        Gets the name of the target field.

        Returns:
            str: The name of the target field.
        """
        return self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["targetField"]

    @property
    def target_class(self) -> str:
        """
        Gets the name of the target class.

        Returns:
            str: The name of the target class.
        """
        return self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["targetClass"]

    @property
    def numeric_features(self) -> List[str]:
        """
        Gets the names of the numeric features.

        Returns:
            List[str]: The list of numeric feature names.
        """
        return self._numeric_features

    @property
    def categorical_features(self) -> List[str]:
        """
        Gets the names of the categorical features.

        Returns:
            List[str]: The list of categorical feature names.
        """
        return self._categorical_features

    @property
    def features(self) -> List[str]:
        """
        Gets the names of all the features.

        Returns:
            List[str]: The list of all feature names.
        """
        return self.numeric_features + self.categorical_features

    @property
    def all_fields(self) -> List[str]:
        """
        Gets the names of all the fields.

        Returns:
            List[str]: The list of all field names.
        """
        return [self.id_field, self.target_field] + self.features