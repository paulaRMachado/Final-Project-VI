import pandas as pd

def get_dataframe(file_name):
    """
    This function imports a CSV file form the data folder into a variable
    :arg:
    file_name: The name of the file to be imported
    """
    df = pd.read_csv(f"data/{file_name}.csv")
    return df