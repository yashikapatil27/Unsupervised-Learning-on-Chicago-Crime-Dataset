from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.75, random_state=50):
    X = df.drop(target, axis=1)
    Y = df[target]
    data_train, data_test, target_train, target_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return data_train, data_test, target_train, target_test

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)
