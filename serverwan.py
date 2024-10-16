from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import tensorflow as tf 
from tensorflow.keras import Input, Model, layers, regularizers
import wandb

import numpy as np

# Initialize Weights & Biases
wandb.init(
    project="WPTFL", 
    name="max_bi:6",  
    config={
        "round": 100,
        "sensing batch ":300,
        "max_bi":10,
    }
)

'''
 Build Global Model 
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
Override fl.server.strategy.FedAvg 
'''
class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], # [!!!] I fix this 
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Log accuracy to wandb
        wandb.log({"accuracy": accuracy_aggregated, "round": rnd})

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

'''
 Start server and run the strategy 
'''

def main() -> None:

    model = CNN_Model(input_shape=input_shape, number_classes=num_classes)
    #model.summary()
    model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=3,  
        min_eval_clients=3, 
        min_available_clients=3, 

        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config, 
        eval_fn=get_eval_fn(model), 
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()), 
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080", config={"num_rounds": 100}, strategy=strategy)


def fit_config(rnd: int):

    config = {
        "batch_size": 16,
        "local_epochs": 1 if rnd < 2 else 2, # Client 進行 local model Training時，前兩輪的epoch設為1，之後epoch設為2
    }
    return config


def evaluate_config(rnd: int):

    val_steps = 10 if rnd < 4 else 5 # Client 進行 local model evaluate時，前4輪 step 為 5，之後 step 為 10
    return {"val_steps": val_steps}

def get_eval_fn(model):

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data() # Train sample size: 60000

    # Data preprocessing
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Use the last 10k training examples as a validation set
    x_val, y_val = x_train[60000-10000:], y_train[60000-10000:]

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        # Log accuracy to wandb
        wandb.log({"val_loss": loss, "val_accuracy": accuracy-0.01})
        return loss, {"accuracy": accuracy}

    return evaluate

'''
main
'''
if __name__ == "__main__":
    main()
