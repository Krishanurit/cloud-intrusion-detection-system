from src.preprocessing.preprocess import load_data, preprocess
from src.detection.predictor import predict
from src.detection.alert import alert

def main():
    print("📥 Loading data...")
    train, test = load_data("data/raw/train.txt", "data/raw/test.txt")

    print("⚙️ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess(train, test)

    # Take one sample from test data
    sample = X_test.iloc[0].values

    print("🔍 Running detection...")
    result = predict(sample)

    alert(result)


if __name__ == "__main__":
    main()