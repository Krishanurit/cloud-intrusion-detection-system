import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.preprocess import load_data, preprocess

# Step 1: Create folder if not exists
os.makedirs("models/saved",  exist_ok=True)

# Step 2: Load dataset
print("📥 Loading data...")
train, test = load_data("data/raw/train.txt", "data/raw/test.txt")

# Step 3: Preprocess data
print("⚙️ Preprocessing data...")
X_train, X_test, y_train, y_test = preprocess(train, test)

print("Training data shape:", X_train.shape)

# Step 4: Train model
print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 5: Save model
model_path = "models/saved/ids_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at:", model_path)