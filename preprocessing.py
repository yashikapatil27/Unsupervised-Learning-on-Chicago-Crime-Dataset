def reset_index(df):
    return df.reset_index(drop=True)

def convert_column_types(df, columns, dtype):
    for column in columns:
        df = df.astype({column: dtype})
    return df

def check_missing_values(df):
    return df.isnull().sum()

def check_unique_values(df):
    return df.nunique()