import numpy as np
from scipy.stats import t # Make sure scipy.stats is imported

SCORING = {
    'linear': lambda y, yh: y - yh,
    'square': lambda y, yh: np.square(y - yh),
    'absolute': lambda y, yh: np.abs(y - yh),
    'exponential': lambda y, yh: 1 - np.exp(-np.abs(y - yh)),
    'poisson': lambda y, yh: yh.clip(1e-6) - y * np.log(yh.clip(1e-6)),
    'hamming': lambda y, yh, classes: (y != yh).astype(int),
    'entropy': lambda y, yh, classes: np.sum(list(map(
        lambda c: -(y == c[1]).astype(int) * np.log(yh[:, c[0]]),
        enumerate(classes))), axis=0)
}


def _normalize_score(scores, weights=None):
    """Normalize scores according to weights"""

    if weights is None:
        return scores.mean()
    else:
        return np.mean(np.dot(scores.T, weights) / weights.sum())


def mse(model, X, y, weights=None, **largs):
    """Mean Squared Error"""

    pred = model.predict(X)
    scores = SCORING['square'](y, pred)

    return _normalize_score(scores, weights)


def rmse(model, X, y, weights=None, **largs):
    """Root Mean Squared Error"""

    return np.sqrt(mse(model, X, y, weights, **largs))


def mae(model, X, y, weights=None, **largs):
    """Mean Absolute Error"""

    pred = model.predict(X)
    scores = SCORING['absolute'](y, pred)

    return _normalize_score(scores, weights)


def poisson(model, X, y, weights=None, **largs):
    """Poisson Loss"""

    if np.any(y < 0):
        raise ValueError("Some value(s) of y are negative which is"
                         " not allowed for Poisson regression.")

    pred = model.predict(X)
    scores = SCORING['poisson'](y, pred)

    return _normalize_score(scores, weights)


def hamming(model, X, y, weights=None, **largs):
    """Hamming Loss"""

    pred = model.predict(X)
    scores = SCORING['hamming'](y, pred, None)

    return _normalize_score(scores, weights)


def crossentropy(model, X, y, classes, weights=None, **largs):
    """Cross Entropy Loss"""

    pred = model.predict_proba(X).clip(1e-5, 1 - 1e-5)
    scores = SCORING['entropy'](y, pred, classes)

    return _normalize_score(scores, weights)

def mae_ci90(model, X, y_true, weights=None, **kwargs):
    """
    Calculates the upper bound of the 90% confidence interval for the
    mean of the absolute errors (residuals) of the model's predictions
    for the samples within the node. It leverages the 'absolute' scoring
    function from the SCORING dictionary.

    Parameters
    ----------
    model : fitted estimator
        The model used to make predictions within the node.
    X : array-like of shape (n_samples, n_features)
        The input samples for the node.
    y_true : array-like of shape (n_samples,)
        The true target values for the node.
    weights : array-like of shape (n_samples,), optional
        Sample weights. Note: This implementation calculates an unweighted
        mean and standard deviation for the confidence interval itself.
        The `weights` argument is passed for signature consistency, but its
        direct effect on the CI calculation within this function is not implemented.
        Node-level aggregation (in `_parallel_binning_fit`) still uses these weights.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    upper_bound_ci : float
        The upper bound of the 90% confidence interval for the mean
        of the absolute errors. Returns infinity if n_samples is too small
        to compute a valid standard deviation (e.g., n <= 1).
    """
    n_samples = len(y_true)

    # Need at least 2 samples to calculate standard deviation (ddof=1)
    # and to have degrees of freedom (n-1) for the t-distribution
    if n_samples <= 1:
        return np.inf # Return a very high loss to discourage splitting into tiny nodes

    pred = model.predict(X)
    # Use SCORING['absolute'] to get the individual absolute errors (scores)
    abs_errors = SCORING['absolute'](y_true, pred)

    # Calculate the sample mean of these absolute errors
    # We do NOT use _normalize_score here because we need the raw scores
    # array for the standard deviation calculation, not just its mean.
    mean_abs_errors = np.mean(abs_errors)

    # Calculate the sample standard deviation of the absolute errors
    # ddof=1 for unbiased sample standard deviation (divides by n-1)
    std_abs_errors = np.std(abs_errors, ddof=1)

    # If the standard deviation is zero (all errors are identical),
    # the confidence interval width is zero, so the upper bound is just the mean.
    if std_abs_errors == 0:
        return mean_abs_errors

    # Calculate the Standard Error of the Mean (SEM) for absolute errors
    sem_abs_errors = std_abs_errors / np.sqrt(n_samples)

    # Determine the t-score for the upper bound of a 90% confidence interval.
    # For a 90% two-sided confidence interval, alpha = 0.10.
    # The upper bound uses the t-value for the (1 - alpha/2) = 0.95 quantile.
    degrees_freedom = n_samples - 1
    t_critical = t.ppf(0.95, degrees_freedom)

    # Calculate the upper bound of the 90% CI for the mean absolute error
    upper_ci_bound = mean_abs_errors + t_critical * sem_abs_errors

    return upper_ci_bound
