import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_data():
    train_data = pd.read_csv('job_change_train.csv')
    test_data = pd.read_csv('job_change_test.csv')

    train_data = train_data.ffill()
    test_data = test_data.ffill()

    X = train_data.drop(columns=['id', 'willing_to_change_job'])
    y = train_data['willing_to_change_job']

    categorical_cols = X.select_dtypes(include=['object']).columns

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cols = pd.DataFrame(
        encoder.fit_transform(X[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X.index
    )

    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, encoded_cols], axis=1)

    encoded_test_cols = pd.DataFrame(
        encoder.transform(test_data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=test_data.index
    )
    test_data = test_data.drop(columns=categorical_cols)
    test_data = pd.concat([test_data, encoded_test_cols], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, test_data

X_train, X_val, y_train, y_val, test_data = load_and_preprocess_data()
