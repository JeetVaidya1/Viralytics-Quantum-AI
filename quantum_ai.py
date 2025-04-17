import os
import subprocess
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd
import pennylane as qml
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

# Disable GPU and force TensorFlow to use CPU mode (for macOS compatibility)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.experimental.set_visible_devices([], 'GPU')

# Step 1: Define Quantum Device (PennyLane)
dev = qml.device("default.qubit", wires=6)  # Increased to 6 qubits for more expressiveness

# Step 2: Enhanced Quantum Feature Map
@qml.qnode(dev)
def quantum_feature_map(x):
    qml.Hadamard(wires=0)
    qml.RX(x[0], wires=1)
    qml.RY(x[1], wires=2)
    qml.RZ(x[0] * x[1], wires=3)
    
    # Deeper Entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])

    qml.RX(x[1] + x[0], wires=4)
    qml.RY(x[0] - x[1], wires=5)

    return qml.probs(wires=[0, 1, 2, 3, 4, 5])  # 64-dimensional output

# Step 3: Load & Improve Cybersecurity Dataset
def load_cybersecurity_data():
    file_path = "data/processed_cybersecurity_data.csv"
    df = pd.read_csv(file_path)
    
    X, y = df.drop(columns=["label"]).to_numpy(), df["label"].to_numpy().reshape(-1, 1)

    # Fix Synthetic Data
    if len(np.unique(y)) == 1:
        print("⚠️ Warning: Only one class found! Generating more realistic synthetic samples.")

        num_samples = X.shape[0]
        num_features = X.shape[1]

        synthetic_X = np.random.randn(num_samples, num_features) * 0.3 + np.mean(X, axis=0) * 0.7
        synthetic_y = np.array([0 if y[0][0] == 1 else 1] * num_samples).reshape(-1, 1)

        X = np.vstack((X, synthetic_X))
        y = np.vstack((y, synthetic_y))

        X, y = shuffle(X, y, random_state=42)

    return X, y

# Step 4: Apply Quantum Feature Mapping
def apply_quantum_features(X, batch_size=50):
    print("🔄 Processing quantum features in batches...")
    X_trimmed = X[:, :2]

    quantum_transformed = []
    for i in range(0, len(X_trimmed), batch_size):
        batch = X_trimmed[i : i + batch_size]
        transformed_batch = [quantum_feature_map(x) for x in batch]
        quantum_transformed.extend(transformed_batch)

        if i % 200 == 0:
            print(f"✅ Processed {i}/{len(X_trimmed)} samples")

    return np.array(quantum_transformed)

# Load dataset
X, y = load_cybersecurity_data()

# Increase Training Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_q_train = apply_quantum_features(X_train[:5000])  # More training data
X_q_test = apply_quantum_features(X_test[:1000])

# Step 5: Optimized Neural Network with L2 Regularization
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_q_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Stronger Optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=2500, decay_rate=0.85, staircase=True
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

# Train with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_q_train, y_train[:5000], epochs=30, batch_size=128, validation_data=(X_q_test, y_test[:1000]), callbacks=[early_stopping])

# Evaluate Performance
loss, accuracy = model.evaluate(X_q_test, y_test[:1000])
print(f"Test Accuracy on New Cyberattacks: {accuracy * 100:.2f}%")

# Step 6: Generate Quantum Data for Real-Time Threat Detection
def generate_quantum_features():
    test_input = np.array([0.5, 0.2])  # Example input
    return quantum_feature_map(test_input).reshape(1, -1)

quantum_features = generate_quantum_features()

# Make Prediction
prediction = model.predict(quantum_features)

# Print Result
print(f"Quantum AI Threat Prediction: {'⚠️ THREAT DETECTED' if prediction[0][0] > 0.5 else '✅ SAFE'}")

model.save("model.keras")

