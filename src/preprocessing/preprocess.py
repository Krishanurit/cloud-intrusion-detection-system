import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# NSL-KDD column names
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]


def load_data(train_path, test_path):
    train = pd.read_csv(train_path, names=columns)
    test = pd.read_csv(test_path, names=columns)

    return train, test


def preprocess(train, test):

    # Remove difficulty column
    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty", axis=1, inplace=True)

    # Convert labels
    train["label"] = train["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    test["label"] = test["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # Encode categorical columns
    categorical = ["protocol_type", "service", "flag"]

    os.makedirs("models/preprocessors", exist_ok=True)

    encoders = {}

    for col in categorical:
        encoder = LabelEncoder()

        encoder.fit(
            pd.concat([train[col], test[col]], axis=0)
        )

        train[col] = encoder.transform(train[col])
        test[col] = encoder.transform(test[col])

        encoders[col] = encoder

    # Save encoders
    with open("models/preprocessors/encoder.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Split features and labels
    X_train = train.drop("label", axis=1)
    y_train = train["label"]

    X_test = test.drop("label", axis=1)
    y_test = test["label"]

    # Scale data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    with open("models/preprocessors/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test