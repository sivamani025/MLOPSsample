# You can add helper functions here later, e.g., data preprocessing
def normalize(X):
    """Normalize input data (0–1 scaling)."""
    return X / 16.0  # since pixel values range from 0–16
