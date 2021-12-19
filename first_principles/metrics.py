def accuracy(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    return float((y_hat == y_true).sum() / len(y_true))

def mis_class_err(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    return float((y_hat != y_true).sum() / len(y_true))
