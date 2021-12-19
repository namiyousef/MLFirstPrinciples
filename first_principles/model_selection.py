import torch
import numpy as np

def kfold(data, splits=3, shuffle=False):
    D = len(data)

    ids = torch.randperm(D) if shuffle else torch.arange(D)
    k = D // splits

    for split in range(splits):
        mask = torch.ones(D, dtype=torch.bool)
        mask[ids[k * split:k * (split + 1)]] = False
        yield ids[mask], ids[~mask]

def tts(X, y, test_size=1/3, random_seed = None): # TODO this is the numpy version, not pytorch!
    """
    Splits the data into train and test samples, without replacement
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    m = X.shape[0]
    assert m == y.shape[0]
    l = round(m * test_size)

    indices = np.arange(m)
    np.random.shuffle(indices)

    X_test, y_test = X[indices[:l]], y[indices[:l]]
    X_train, y_train = X[indices[l:]], y[indices[l:]]
    return X_train, X_test, y_train, y_test


def cross_validation(X, y, partial_model, cross_val_params, k_split_ids, error, **prediction_params):
    """Performs cross validation on a model and returns the parameters with the least error

    :param cross_val_params: keys are params for cross validation, values are iterables
    :type corss_val_params: dict
    """
    # needs an option for prediction parameters!
    param_names = cross_val_params.keys()
    param_ranges = cross_val_params.values()
    param_combs = product(*param_ranges)
    min_err = float('inf')
    for comb in param_combs:
        model_params = {param: value for param, value in zip(param_names, comb)}
        cv_err = 0
        for i, (train_ids, val_ids) in enumerate(k_split_ids):
            # TODO need to double check that shapes are correct here, remember model currently supports
            # taking .reshape(-1,1)!
            X_train, y_train = X[train_ids], y[train_ids]
            X_val, y_val = X[val_ids], y[val_ids]

            model = partial_model(**model_params)  # TODO this is wrong because it does not create a fresh model...

            model.fit(X_train, y_train)
            y_hat = model.predict(X_val, **prediction_params)
            cv_err += error(y_hat, y_val)

        cv_err /= (i + 1)
        if cv_err < min_err:
            min_err = cv_err
            best_params = model_params

    return best_params, min_err