import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("C:/Users/INDUSTRY 4.0/Desktop/sample/aircrafts/digits_model.joblib")

# Example: predict on first test sample
from sklearn.datasets import load_digits
digits = load_digits()
sample = digits.data[0].reshape(1, -1)
prediction = model.predict(sample)

# Show result
print("Predicted digit:", prediction[0])
plt.imshow(digits.images[0], cmap="gray")
plt.title(f"Predicted: {prediction[0]}")
plt.show()
