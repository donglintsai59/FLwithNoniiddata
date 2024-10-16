'''
引入套件
'''
import flwr as fl
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, models # 建立CNN架構
import numpy as np # 資料前處理

import argparse # CmmandLine 輸入控制參數
import os # 更改tensorflow的Log訊息的顯示模式

# Make TensorFlow logs less verbose (減少不必要的訊息顯示)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''
Step 1. Build Local Model (建立本地模型)
'''
# Hyperparameter超參數
num_classes = 10
input_shape = (28, 28, 1)

# Build Model
def CNN_Model(input_shape, number_classes):
  # define Input layer
  input_tensor = Input(shape=input_shape) # Input: convert normal numpy to Tensor (float32)

  # define layer connection
  x = layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu")(input_tensor)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(number_classes, activation="softmax")(x)

  # define model
  model = Model(inputs=input_tensor, outputs=outputs, name="mnist_model")
  return model

'''
Step 2. Load local dataset (引入本地端資料集)，對Dataset進行切割
'''
def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    #assert idx in range(10) # limit it can't more than 10...
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Train 60000-5000, Test 10000

    # Data preprocessing
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    sample_size = 60000 #x_train.shape[0]
    sample_size -= 5000 # Server-side 預留做驗證
    training_size = int(sample_size/2)
    testing_size = int(10000/2)

    return ( 
        x_train[idx * training_size : (idx + 1) * training_size],
        y_train[idx * training_size : (idx + 1) * training_size],
    ), (
        x_test[idx * testing_size : (idx + 1) * testing_size],
        y_test[idx * testing_size : (idx + 1) * testing_size],
    )

'''
Step 3. Define Flower client (定義client的相關設定: 接收Server-side的global model weight、hyperparameters)
'''
class MnistClient(fl.client.NumPyClient):
    # Class初始化: local model、dataset
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    # 此時已無作用，原用來取得 local model 的 ini-weight，
    # 目前初始權重值是來自 Server-side 而非 client 自己
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

        # Train the model using hyperparameters from config
        # (依 Server-side 的 hyperparameters 進行訓練)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        # 將訓練後的權重、資料集筆數、正確率/loss值等，回傳至server-side
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
Step 4. Create an instance of our flower client and add one line to actually run this client. (建立Client-to-Server的連線)
'''
def main() -> None:
    # Parse command line argument `partition`
    # 從 CommandLine 輸入 Client 編號，對 Dataset進行切割
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = CNN_Model(input_shape=(28, 28, 1), number_classes=10)
    #model.summary()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = MnistClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client) # windows

if __name__ == "__main__":
    main()