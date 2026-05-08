import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ✅ ADD THIS FUNCTION (you are missing this)
def load_data(train_path, test_path):
    columns = [f"f{i}" for i in range(41)] + ["label"]

    train = pd.read_csv(train_path, names=columns)
    test = pd.read_csv(test_path, names=columns)

    return train, test


def preprocess(train, test):

    # Convert label
    train['label'] = train['label'].apply(lambda x: 0 if x == "normal" else 1)
    test['label'] = test['label'].apply(lambda x: 0 if x == "normal" else 1)

    # Combine for encoding (fix unseen label error)
    combined = pd.concat([train, test])

    for col in combined.columns:
        if combined[col].dtype == 'object':
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col])

    # Split back
    train = combined.iloc[:len(train)]
    test = combined.iloc[len(train):]

    X_train = train.drop("label", axis=1)
    y_train = train["label"]

    X_test = test.drop("label", axis=1)
    y_test = test["label"]

    return X_train, X_test, y_train, y_test