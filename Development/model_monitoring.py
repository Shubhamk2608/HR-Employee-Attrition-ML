import os
import pandas as pd


def save_prediction(
    input_data,
    prediction,  ## model result
    probability,
    model_version,
    log_file  ###### prediction saved in this file
):

    log_data = input_data.copy()      ######original data copy

    log_data["Prediction"] = prediction
    log_data["Probability"] = probability
    log_data["Model_Version"] = model_version  ###model name
    log_data["Actual_Label"] = None
    file_exists = os.path.exists(log_file)  ####file checking

    log_data.to_csv(
        log_file,
        mode="a",
        header=not file_exists,
        index=False
    )

def add_actual_labels(log_file, actual_labels):

    df = pd.read_csv(log_file)

    df["Actual_Label"] = actual_labels

    df.to_csv(
        log_file,
        index=False
    )
