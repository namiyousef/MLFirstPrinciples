from functools import partial
import time
from first_principles.kernel_functions import polynomial_kernel
import torch

class KernelPerceptron:

    def __init__(self, verbose=0, epsilon=0.001, epochs=1, kernel=polynomial_kernel, **kernel_kw):

        self.kernel = partial(kernel, **kernel_kw)
        self.epsilon = epsilon
        self.verbose = verbose
        self.epochs = epochs

    def fit(self, X, y):

        assert len(y.shape) == 1

        if not hasattr(self, 'X_train'):
            self.X_train = X
        if not hasattr(self, 'K'):
            self.K = self.kernel(X, X).float()

        m = X.shape[0]
        self.weights = torch.zeros((m))

        for epoch in range(self.epochs):
            s = time.time()
            tot_err = 0
            for i in range(m):
                y_pred = torch.sign(self.K[i, :] @ self.weights + self.epsilon)

                # TODO maybe self.weights should be += is_err * y[i]?
                is_err = y_pred != y[i]
                tot_err += is_err  # TODO need to check: how to correctly calculate the errors? Currently this is wrong for agg_vec=True
                self.weights[i] = is_err * y[i]

            e = time.time()
            if self.verbose:
                print(
                    f'Epoch {epoch + 1}/{self.epochs} complete. Running Loss: {tot_err / (i + 1):.3g}.  Time taken: {e - s:.3g}')

        return self

    def predict(self, X, weights=None, return_labels=True):
        X = self.transform_data_for_prediction(X)
        if weights is not None:
            y_pred = X @ weights
        else:
            y_pred = X @ self.weights

        if return_labels:
            y_pred = self.predict_labels(y_pred)
        return y_pred

    def predict_labels(self, y):
        y = torch.sign(y + self.epsilon)
        return y

    def transform_data_for_prediction(self, X):
        """Transforms data to a format that is ready to be multiplied by weights for prediction
        (output_size x dimension_size) * (dimension_size)
        """
        return self.kernel(X, self.X_train).float()


class OneVsOne:

    def __init__(self, classes, model, agg_vec=False, **model_params):
        """
        NOTE: requires classes to be arranged as 0,1, ... k-1
        """

        self.agg_vec = agg_vec
        self.classes = classes
        self.k = len(self.classes)

        if agg_vec:
            self.model = model(**model_params)
        else:
            self.model = [model(**model_params) for i in range(self.k * (self.k - 1) // 2)]

    def fit(self, X, y):
        combs = torch.combinations(self.classes, 2)
        mask_val = float(self.k)  # make the mask value the number that corresponds to the last class.

        if self.agg_vec:
            y = [torch.where((y != i) & (y != j), mask_val, y) for (i, j) in combs]
            y = [torch.where(y_ == j, -1., y_) for (_, j), y_ in zip(combs, y)]
            y = [torch.where(y_ == i, 1., y_) for (i, _), y_ in zip(combs, y)]
            y = [torch.where(y_ == mask_val, 0., y_) for y_ in y]
            self.weights = torch.cat([self.model.fit(X, y_).weights for y_ in y]).reshape(self.k * (self.k - 1) // 2,
                                                                                          -1).T

        else:
            indices = [(y == i) | (y == j) for i, j in combs]
            y = [torch.where(y[ids] == i, 1, -1) for (i, j), ids in zip(combs, indices)]
            X = [X[ids] for ids in indices]

            self.model = [model.fit(X_, y_) for X_, y_, model in zip(X, y, self.model)]

        return self

    def predict(self, X, return_labels=True, weighted_vote=False):

        if self.agg_vec:
            y_pred = self.model.predict(X, weights=self.weights, return_labels=not weighted_vote).T
        else:
            y_pred = torch.cat([
                model.predict(X, weights=model.weights, return_labels=not weighted_vote).reshape(1, -1) for model in
                self.model
            ])

        y_pred = torch.cat([
            # add positives
            (torch.where(
                y_pred[(2 * self.k - i - 1) * i // 2:(2 * self.k - i - 2) * (i + 1) // 2] > 0,
                y_pred[(2 * self.k - i - 1) * i // 2:(2 * self.k - i - 2) * (i + 1) // 2],
                torch.zeros(1)
            ).sum(dim=0) + \
             # add negatives
             (torch.where(
                 torch.cat([y_pred[(i - 1) + j * (2 * self.k - j - 3) // 2].reshape(1, -1) for j in range(i)]) < 0,
                 -torch.cat([y_pred[(i - 1) + j * (2 * self.k - j - 3) // 2].reshape(1, -1) for j in range(i)]),
                 torch.zeros(1)
             ).sum(dim=0) \
                  if i else torch.zeros(X.shape[0]))).reshape(1, -1) \
            for i in range(self.k)
        ]).T

        if return_labels:
            y_pred = self.predict_labels(y_pred)

        return y_pred

    def predict_labels(self, y):
        y = torch.argmax(y, dim=1)
        return y


class OneVsAll:
    def __init__(self, classes, model, **model_params):
        self.model = model(**model_params)
        self.classes = classes
        self.k = len(self.classes)

    def fit(self, X, y):
        y = [torch.where(y == i, 1, -1) for i in self.classes]
        self.weights = torch.cat([self.model.fit(X, y_).weights for y_ in y]).reshape(self.k, -1).T

        return self

    def predict(self, X, return_labels=True):
        y_pred = self.model.predict(X, weights=self.weights, return_labels=False)

        if return_labels:
            y_pred = self.predict_labels(y_pred)

        return y_pred

    def predict_labels(self, y):
        y = torch.argmax(y, dim=1)
        return y