"""
Utility functions for advertising data analysis
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_descriptive_stats(data):
    """
    Calculate descriptive statistics for a dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
        
    Returns:
    --------
    pandas.DataFrame
        Statistical summary
    """
    stats_dict = {
        'Mean': data.mean(),
        'Median': data.median(),
        'Variance': data.var(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Q1': data.quantile(0.25),
        'Q3': data.quantile(0.75)
    }
    return pd.DataFrame(stats_dict).T


def perform_svd(X, normalize=True):
    """
    Perform Singular Value Decomposition.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input matrix
    normalize : bool
        Whether to normalize data before SVD
        
    Returns:
    --------
    tuple
        U, S, Vt matrices and variance explained
    """
    if normalize:
        X_proc = (X - X.mean(axis=0)) / X.std(axis=0)
    else:
        X_proc = X.copy()
    
    U, S, Vt = np.linalg.svd(X_proc, full_matrices=False)
    
    # Calculate variance explained
    variance = (S ** 2) / (len(X_proc) - 1)
    variance_ratio = variance / variance.sum()
    
    return U, S, Vt, variance_ratio


def fit_linear_regression(X, y, method='svd'):
    """
    Fit linear regression model using matrix operations.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (without intercept)
    y : numpy.ndarray
        Target vector
    method : str
        Method to use: 'inverse' or 'svd'
        
    Returns:
    --------
    numpy.ndarray
        Coefficient vector (including intercept)
    """
    # Add intercept
    X_int = np.column_stack([np.ones(len(X)), X])
    
    if method == 'inverse':
        # Normal equation: β = (X'X)^-1 X'y
        XtX = X_int.T @ X_int
        Xty = X_int.T @ y
        beta = np.linalg.inv(XtX) @ Xty
    elif method == 'svd':
        # Using SVD for numerical stability
        U, S, Vt = np.linalg.svd(X_int, full_matrices=False)
        beta = Vt.T @ np.diag(1/S) @ U.T @ y
    else:
        raise ValueError("Method must be 'inverse' or 'svd'")
    
    return beta


def regression_statistics(X, y, beta):
    """
    Calculate regression statistics.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (without intercept)
    y : numpy.ndarray
        Target vector
    beta : numpy.ndarray
        Coefficient vector (including intercept)
        
    Returns:
    --------
    dict
        Dictionary containing various statistics
    """
    # Add intercept
    X_int = np.column_stack([np.ones(len(X)), X])
    
    # Predictions and residuals
    y_pred = X_int @ beta
    residuals = y - y_pred
    
    # Basic metrics
    n = len(y)
    p = X.shape[1]
    
    SST = np.sum((y - y.mean()) ** 2)
    SSR = np.sum((y_pred - y.mean()) ** 2)
    SSE = np.sum(residuals ** 2)
    
    r2 = SSR / SST
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = SSE / (n - p - 1)
    rmse = np.sqrt(mse)
    
    # Standard errors and t-statistics
    var_beta = mse * np.linalg.inv(X_int.T @ X_int)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'mse': mse,
        'rmse': rmse,
        'se_beta': se_beta,
        't_stats': t_stats,
        'p_values': p_values,
        'residuals': residuals,
        'predictions': y_pred
    }


def bayesian_update_beta(successes, failures, alpha_prior=1, beta_prior=1):
    """
    Perform Bayesian update with Beta distribution.
    
    Parameters:
    -----------
    successes : int
        Number of successes
    failures : int
        Number of failures
    alpha_prior : float
        Prior alpha parameter
    beta_prior : float
        Prior beta parameter
        
    Returns:
    --------
    dict
        Dictionary containing posterior parameters and statistics
    """
    from scipy.stats import beta
    
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    
    posterior_mean = alpha_post / (alpha_post + beta_post)
    credible_interval = beta.interval(0.95, alpha_post, beta_post)
    
    return {
        'alpha_post': alpha_post,
        'beta_post': beta_post,
        'posterior_mean': posterior_mean,
        'credible_interval': credible_interval
    }
