import os
import json
import pandas as pd, numpy as np
from data_management.schema_provider import BinaryClassificationSchema

def read_json_in_directory(directory_path: str) -> dict:
    """
    Reads a JSON file in the given directory path as a dictionary and returns the dictionary.
    
    Args:
    - directory_path (str): The path to the directory containing the JSON file.
    
    Returns:
    - dict: The contents of the JSON file as a dictionary.
    
    Raises:
    - ValueError: If no JSON file is found in the directory or if multiple JSON files are found in the directory.
    """
    json_files = [file for file in os.listdir(directory_path) if file.endswith('.json')]
    
    if not json_files:
        raise ValueError('No JSON file found in directory')
    
    if len(json_files) > 1:
        raise ValueError('Multiple JSON files found in directory')
    
    json_file_path = os.path.join(directory_path, json_files[0])
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data


def read_csv_in_directory(directory_path: str) -> pd.DataFrame:
    """
    Reads a CSV file in the given directory path as a pandas dataframe and returns the dataframe.
    
    Args:
    - directory_path (str): The path to the directory containing the CSV file.
    
    Returns:
    - pd.DataFrame: The pandas dataframe containing the data from the CSV file.
    
    Raises:
    - ValueError: If no CSV file is found in the directory or if multiple CSV files are found in the directory.
    """
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    
    if not csv_files:
        raise ValueError('No CSV file found in directory')
    
    if len(csv_files) > 1:
        raise ValueError('Multiple CSV files found in directory')
    
    csv_file_path = os.path.join(directory_path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    return df


def read_data(data_dirpath: str, data_schema: BinaryClassificationSchema) -> pd.DataFrame:
    """
    Reads data and casts fields to the expected type as per the provided schema.

    Args:
        data_dirpath (str): The directory path to the training data.
        data_schema (BinaryClassificationSchema): The schema provider object.

    Returns:
        pd.DataFrame: The training data as a pandas DataFrame with casted field types.
    """

    data = read_csv_in_directory(data_dirpath)

    # Cast id field to string
    data[data_schema.id_field] = data[data_schema.id_field].astype(str)
    
    # Cast target field to string
    if data_schema.target_field in data.columns:
        data[data_schema.target_field] = data[data_schema.target_field].astype(str)
    
    # Cast categorical features to string
    for c in data_schema.categorical_features:
        if c in data.columns:
            data[c] = data[c].astype(str)
    
    # Cast numeric features to float
    for c in data_schema.numeric_features:
        if c in data.columns:
            data[c] = data[c].astype(np.float32)
    
    return data