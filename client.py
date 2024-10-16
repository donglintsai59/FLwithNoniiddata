import flwr as fl
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers
import numpy as np 

import argparse 
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''
 Build Local Model 
'''
# Hyperparameter超參數
num_classes = 10
input_shape = (28, 28, 1)

# Build Model
def CNN_Model(input_shape, number_classes):
    # define Input layer
    input_tensor = Input(shape=input_shape)  # Input: convert normal numpy to Tensor (float32)

    # define layer connection without L2 regularization
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(number_classes, activation="softmax")(x)

    # define model
    model = Model(inputs=input_tensor, outputs=outputs, name="simple_mnist_model")
    return model
'''
Load local dataset
'''

def load_partition(idx: int, num_clients: int = 10, alpha: float = 0.5, samples_per_client: list = [300,300,300]):
    """Load a fixed number of samples for each client to simulate a partition using Dirichlet distribution."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Train 60000-5000, Test 10000

    # Data preprocessinga
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Get the number of samples for the current client
    if idx < len(samples_per_client):
        num_samples = samples_per_client[idx]
    else:
        raise ValueError("Client index out of range for samples_per_client list")

    # Use Dirichlet distribution to create non-IID partitions
    num_classes = 10
    class_size = len(y_train) // num_classes
    class_indices = [np.where(np.argmax(y_train, axis=1) == i)[0] for i in range(num_classes)]
    
    # Generate Dirichlet distribution samples
    proportions = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
    
    # Allocate data to each client
    client_indices = []
    for i in range(num_classes):
        class_idx = class_indices[i]
        np.random.shuffle(class_idx)
        split_indices = np.split(class_idx, (proportions[i].cumsum()[:-1] * len(class_idx)).astype(int))
        client_indices.append(split_indices[idx])

    # Flatten list of indices for the current client
    client_indices = np.concatenate(client_indices)
    
    # Shuffle indices if needed
    np.random.shuffle(client_indices)
    
    client_indices = client_indices[:num_samples]  # Limit to num_samples

    return (
        x_train[client_indices],
        y_train[client_indices],
    ), (
        x_test[:num_samples // num_clients],  # Equally divide the test set
        y_test[:num_samples // num_clients],
    )
'''waaaaa
Define Flower client 
'''
class MnistClient(fl.client.NumPyClient):
    # Class初始化: local model、dataset
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

'''
 Create an instance of our flower client and add one line to actually run this client.
'''
def main() -> None:

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = CNN_Model(input_shape=(28, 28, 1), number_classes=10)
    #model.summary()
    model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = MnistClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client) # windows

if __name__ == "__main__":
    main()
