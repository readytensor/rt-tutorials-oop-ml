import os
import numpy as np
import random 

from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, save_preprocessor_and_lbl_encoder, get_label_encoder
from data_management.data_utils import read_json_in_directory, read_data
from algorithm.classifier import Classifier
import paths

def set_seeds(seed_value=42):
    if type(seed_value) == int or type(seed_value) == float:          
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
    else: 
        print(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def run_training():

    # set seeds 
    set_seeds(seed_value=0)     

    # loads the json file schema into a dictionary and use it to instantiate the schema provider
    schema_dict = read_json_in_directory(paths.SCHEMA_DIR)
    data_schema = BinaryClassificationSchema(schema_dict)

    # load train data
    train_data = read_data(data_dirpath=paths.TRAIN_DIR, data_schema=data_schema)

    # preprocessing
    preprocess_pipeline = get_preprocess_pipeline(data_schema)
    transformed_data = preprocess_pipeline.fit_transform(train_data[data_schema.features])     
    label_encoder = get_label_encoder(data_schema)
    transformed_labels = label_encoder.fit_transform(train_data[[data_schema.target_field]])

    # instantiate and train classifier model 
    classifier = Classifier()
    feature_cols = [c for c in transformed_data.columns if c not in [data_schema.id_field, data_schema.target_field]]
    classifier.fit(
        train_X=transformed_data[feature_cols], 
        train_y=transformed_labels
    )

    # save preprocessor
    save_preprocessor_and_lbl_encoder(preprocess_pipeline, label_encoder, paths.MODEL_ARTIFACTS_PATH)

    # save model
    classifier.save(model_path=paths.MODEL_ARTIFACTS_PATH)

    print("Training completed successfully")


if __name__ == "__main__": 
    run_training()