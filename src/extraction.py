import pandas as pd

def get_dataframe(file_name):
    df = pd.read_csv(f"data/{file_name}.csv")
    return df