from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (8x8 images of handwritten digits 0–9)
digits = load_digits()
X, y = digits.data, digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.2f}")

# Save model
joblib.dump(model, "C:/Users/INDUSTRY 4.0/Desktop/sample/aircrafts/digits_model.joblib")
