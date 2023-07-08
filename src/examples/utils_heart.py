from typing import Tuple

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def binarize_sex(sex: str) -> int:
    if sex.lower() == 'f':
        return 1
    else:
        return 0

def binarize_exercise_angina(exercise_angina: str) -> int:
    if exercise_angina.lower() == 'y':
        return 1
    else:
        return 0

TRANSFORMERS = {
        'Age': MinMaxScaler(),
        'RestingBP': MinMaxScaler(),
        'Cholesterol': MinMaxScaler(),
        'MaxHR': MinMaxScaler(),
        'Oldpeak': MinMaxScaler(),
        'ChestPainType': OneHotEncoder(),
        'RestingECG': OneHotEncoder(),
        'ST_Slope': OneHotEncoder(),
        'Sex': binarize_sex,
        'ExerciseAngina': binarize_exercise_angina
}

def preprocess_heart_data(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]: 
    single_values_columns = []
    multiple_values_columns = []
    for column_name, preprocessor in TRANSFORMERS.items():
        if isinstance(preprocessor, MinMaxScaler):
            df[column_name] = preprocessor.fit_transform(df[column_name].to_numpy().reshape(-1, 1))
            single_values_columns.append(column_name)
        elif isinstance(preprocessor, OneHotEncoder):
            df[column_name] = preprocessor.fit_transform(df[column_name].to_numpy().reshape(-1, 1)).toarray().tolist()
            multiple_values_columns.append(column_name)
        else:
            df[column_name] = df[column_name].apply(lambda val: preprocessor(val))
            single_values_columns.append(column_name)
    
    labels = df['HeartDisease'].to_numpy()

    inputs_rows = []
    columns_order = []
    for _, row in df.drop(columns='HeartDisease').iterrows():
        arr_row = [row[coln] for coln in single_values_columns]
        [arr_row.extend(row[coln]) for coln in multiple_values_columns]

        columns_order = single_values_columns + multiple_values_columns

        inputs_rows.append(arr_row)

    labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    inputs = torch.tensor(inputs_rows)

    return inputs, labels