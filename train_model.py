"""
Train a CNN model on the MNIST dataset for hand-written digit recognition.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_model():
    """Create a CNN model for digit recognition."""
    model = keras.Sequential([
        # Input layer - expecting 28x28 grayscale images
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    return model

def train():
    """Load data, train model, and save it."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Create and compile the model
    print("\nCreating model...")
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Display model architecture
    model.summary()

    # Train the model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate the model
    print("\nEvaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save the model
    print("\nSaving model...")
    model.save('mnist_model.h5')
    print("Model saved as 'mnist_model.h5'")

    return model, history

if __name__ == "__main__":
    train()
