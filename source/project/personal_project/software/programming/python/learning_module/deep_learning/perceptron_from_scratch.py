import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd


@app.cell
def _(np):
    class Perceptron:
        def __init__(self, eta, epochs):
            self.weights = np.random.randn(3) * 1e-4
            print(f"self.weights: {self.weights}")

            self.eta = eta
            self.epochs = epochs

        def activation_function(self, inputs, weights):
            z = np.dot(inputs, weights)
            return np.where(z > 0, 1, 0)

        def fit(self, X, y):
            self.X = X
            self.y = y.values.flatten()

            X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
            print(f"X with bias: {X_with_bias}")

            for epoch in range(self.epochs):
                print(f"\nEpoch: {epoch}")

                y_hat = self.activation_function(X_with_bias, self.weights)
                print(f"Predicted value: {y_hat}")

                error = self.y - y_hat
                print(f"Error: {error}")

                # Weights update

                self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)
                print(f"Updated weights: {self.weights}")

        def predict(self, X):
            X_with_bias = np.c_[X, -np.ones((len(X), 1))]
            return self.activation_function(X_with_bias, self.weights)

    return (Perceptron,)


@app.cell
def _(pd):
    # AND gate truth table solution using perceptron

    _data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 0, 0, 1]}

    and_gate_data = pd.DataFrame(_data)
    and_gate_data
    return (and_gate_data,)


@app.cell
def _(and_gate_data):
    and_gate_X = and_gate_data.drop("y", axis=1)

    _y = and_gate_data["y"]
    and_gate_y = _y.to_frame()
    return and_gate_X, and_gate_y


@app.cell
def _(Perceptron, and_gate_X, and_gate_y):
    _model = Perceptron(eta=0.5, epochs=10)
    _model.fit(and_gate_X, and_gate_y)
    return


@app.cell
def _(pd):
    # XOR gate truth table solution using perceptron

    _data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 1, 1, 0]}

    xor_gate_data = pd.DataFrame(_data)
    xor_gate_data
    return (xor_gate_data,)


@app.cell
def _(xor_gate_data):
    xor_gate_X = xor_gate_data.drop("y", axis=1)

    _y = xor_gate_data["y"]
    xor_gate_y = _y.to_frame()
    return xor_gate_X, xor_gate_y


@app.cell
def _(Perceptron, xor_gate_X, xor_gate_y):
    _model = Perceptron(eta=0.5, epochs=100)
    _model.fit(xor_gate_X, xor_gate_y)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
