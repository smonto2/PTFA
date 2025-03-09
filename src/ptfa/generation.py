# Collection of functions to generate data according to a variety of DGPs
# All DGPs are carefully described in the "example.ipybn" notebook file of PTFA
import numpy as np
from scipy.linalg import toeplitz

# Generating from simplest univariate Gaussian data with factors
def generate_data_simple(T, p, q, k, sigma_x, sigma_y, seed_value=None):
    """
    Input: 
        T          - Sample size
        p          - Number of features
        q          - Number of targets
        k          - Number of factors (components)
        sigma_x    - Scale of features
        sigma_y    - Scale of targets
        seed_value - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X - Features (T x p matrix)
        Y - Targets  (T x q matrix)
        F - Scores   (T x k matrix)
        P - Loadings of feature equation (p x k matrix)
        Q - Loadings of target equation (q x k matrix)
    """
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [T, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [T, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [T, q])
    
    return X, Y, F, P, Q

# Generating from targeted factor model with Toeplitz covariance matrices
def generate_data_system(T, p, q, k, sigma_x, sigma_y, seed_value=None):
    """
    Input: 
        T          - Sample size
        p          - Number of features
        q          - Number of targets
        k          - Number of factors (components)
        sigma_x    - Vector with components for Toeplitz structure in covariance matrix of features (size T)
        sigma_y    - Vector with components for Toeplitz structure in covariance matrix of targets (size T)
        seed_value - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X       - Features (T x p matrix)
        Y       - Targets  (T x q matrix)
        F       - Scores   (T x k matrix)
        P       - Loadings of feature equation (p x k matrix)
        Q       - Loadings of target equation (q x k matrix)
        Sigma_x - Covariance matrix of features
        Sigma_y - Covariance matrix of targets
    """
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [T, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Construct covariance matrices using Toeplitz structure
    Sigma_x = toeplitz(sigma_x)
    Sigma_y = toeplitz(sigma_y)

    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.multivariate_normal(mean = np.zeros(p), cov = Sigma_x, size = T)
    Y = F @ Q.T + rng.multivariate_normal(mean = np.zeros(q), cov = Sigma_y, size = T)
    
    return X, Y, F, P, Q, Sigma_x, Sigma_y

# Generating from targeted factor model with predetermined general errors
def generate_data_generalerrors(T, p, q, k, E_X, E_Y, seed_value=None):
    """
    Input: 
        T          - Sample size
        p          - Number of features
        q          - Number of targets
        k          - Number of factors (components)
        E_X        - Matrix of errors for features (T x p)
        E_Y        - Matrix of errors for targets (T x q)
        seed_value - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X       - Features (T x p matrix)
        Y       - Targets  (T x q matrix)
        F       - Scores   (T x k matrix)
        P       - Loadings of feature equation (p x k matrix)
        Q       - Loadings of target equation (q x k matrix)
    """
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [T, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + E_X
    Y = F @ Q.T + E_Y
    
    return X, Y, F, P, Q

# Generate data from the simplest model with missing-at-random entries
def generate_data_missingatrandom(T, p, q, k, sigma_x, sigma_y,
                                  proportion_x=0.1, proportion_y=None, seed_value=None, return_nan=True):
    """
    Input: 
        T            - Sample size
        p            - Number of features
        q            - Number of targets
        k            - Number of factors (components)
        sigma_x      - Scale of features
        sigma_y      - Scale of targets
        proportion_x - Percentage of missing-at-random observations in feature matrix
        proportion_y - Percentage of missing-at-random observations in target matrix (= proportion_x if None)
        seed_value   - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X - Features (T x p Numpy masked array with fill_value=0.0)
        Y - Targets  (T x q Numpy masked array with fill_value=0.0)
        F - Scores   (T x k matrix)
        P - Loadings of feature equation (p x k matrix)
        Q - Loadings of target equation (q x k matrix)
    """
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [T, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [T, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [T, q])
    
    # Select indices to turn into missing observations
    if proportion_y is None:
        proportion_y = proportion_x
    X_size = T * p
    Y_size = T * q
    num_missing_X = int(proportion_x * X_size)
    num_missing_Y = int(proportion_y * Y_size)
    missing_indices_X = rng.choice(X_size, num_missing_X, replace=False, shuffle=False)
    missing_indices_Y = rng.choice(Y_size, num_missing_Y, replace=False, shuffle=False)

    # Transform indices of missing observations to a mask over the data matrices
    missing_indices_X = np.unravel_index(missing_indices_X, [T, p])
    missing_indices_Y = np.unravel_index(missing_indices_Y, [T, q])
    X_missing_mask = np.zeros_like(X, dtype="bool")
    Y_missing_mask = np.zeros_like(Y, dtype="bool")
    X_missing_mask[missing_indices_X] = True
    Y_missing_mask[missing_indices_Y] = True

    # Return the data as Numpy masked objects for easy handling of missing data and imputation
    X = np.ma.MaskedArray(data=X, mask=X_missing_mask, fill_value=0.0)
    Y = np.ma.MaskedArray(data=Y, mask=Y_missing_mask, fill_value=0.0)
    
    return X, Y, F, P, Q

# Generate from a mixed-frequency version of the model
def generate_data_mixedfrequency(high_frequency_T, low_frequency_T, periods, p, q, k, sigma_x, sigma_y, seed_value=None):
    """
    Input: 
        high_frequency_T - Number of high-frequency observations (size of features)
        low_frequency_T  - Number of low-frequency observations (size of targets)
        periods          - Number of high-frequency periods per low-frequency interval
        p                - Number of features
        q                - Number of targets
        k                - Number of factors (components)
        sigma_x          - Scale of features
        sigma_y          - Scale of targets
        seed_value       - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X        - High-frequency features (high_frequency_T x p matrix)
        Y        - Low-frequency targets  (low_frequency_T x q matrix)
        F        - Scores   (high_frequency_T x k matrix)
        P        - Loadings of feature equation (p x k matrix)
        Q        - Loadings of target equation (q x k matrix)
        Y_latent - Simulated high-frequency targets (high_frequency_T x q matrix)
    """
    # Error checking for time periods: (low_frequency_T - 1) * periods < high_frequency_T <= low_frequency_T * periods
    if (low_frequency_T - 1) * periods >= high_frequency_T or high_frequency_T > low_frequency_T * periods:
        raise AssertionError("Provide arguments such that (low_frequency_T - 1) * periods < high_frequency_T <= low_frequency_T * periods")

    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [high_frequency_T, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise (at the higher frequency)
    X = F @ P.T + rng.normal(scale = sigma_x, size = [high_frequency_T, p])
    Y_latent = F @ Q.T + rng.normal(scale = sigma_y, size = [high_frequency_T, q])

    # Transform latent targets to observable targets (at lower frequency)
    last_T = periods * (low_frequency_T - 1)
    remainder_T = high_frequency_T - last_T
    Y = np.zeros([low_frequency_T, q])
    for t in range(low_frequency_T - 1):
        row_index = range(t * periods, (t+1) * periods)
        Y[t] = np.sum(Y_latent[row_index], axis = 0)
    row_index = range(last_T, (low_frequency_T - 1) * periods + remainder_T)
    Y[low_frequency_T - 1] = np.sum(Y_latent[row_index], axis = 0)
    
    return X, Y, F, P, Q, Y_latent

# Add univariate stochastic volatility through EWMA processes
def generate_data_stochasticvolatility(T, p, q, k, sigma2_x0=0.5, sigma2_y0=0.5, lambda_x=0.9, lambda_y=0.9, seed_value=None):
    """
    Input: 
        T          - Sample size
        p          - Number of features
        q          - Number of targets
        k          - Number of factors (components)
        sigma2_x0  - Initial variance of features and variance of moving average component of feature volatility
        sigma2_y0  - Initial variance of targets and variance of moving average component of target volatility
        lambda_x   - Smoothing parameter for generating stochastic volatility in features according to EWMA
        lambda_y   - Smoothing parameter for generating stochastic volatility in targets according to EWMA
        seed_value - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X        - Features (T x p matrix, masked array with fill_value=0.0 if return_nan = False)
        Y        - Targets  (T x q matrix, masked array with fill_value=0.0 if return_nan = False)
        F        - Scores   (T x k matrix)
        P        - Loadings of feature equation (p x k matrix)
        Q        - Loadings of target equation (q x k matrix)
        sigma2_x - Vector of variances for features (T-dimensional vector)
        sigma2_y - Vector of variances for targets (T-dimensional vector)
    """
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [T, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate volatilities according to a Exponentially Weighted Moving Average (EWMA)
    sigma2_x = np.zeros(T)
    sigma2_y = np.zeros(T)
    sigma2_x[0], sigma2_y[0] = sigma2_x0, sigma2_y0  # Initial volatilities
    eta_x_squared = rng.normal(scale = np.sqrt(sigma2_x0), size = T-1) ** 2
    eta_y_squared = rng.normal(scale = np.sqrt(sigma2_y0), size = T-1) ** 2
    for t in range(1, T):
        sigma2_x[t] = (1 - lambda_x) * eta_x_squared[t-1] + lambda_x * sigma2_x[t-1]
        sigma2_y[t] = (1 - lambda_y) * eta_y_squared[t-1] + lambda_y * sigma2_y[t-1]
    
    # Generate synthetic X and Y data with time-varying volatility
    X = F @ P.T + rng.normal(scale=np.sqrt(sigma2_x)[:, np.newaxis], size=[T, p])
    Y = F @ Q.T + rng.normal(scale=np.sqrt(sigma2_y)[:, np.newaxis], size=[T, q])

    return X, Y, F, P, Q, sigma2_x, sigma2_y

# Add a dynamic VAR(1) factor equation to our standard specification
def generate_data_dynamicfactors(T, p, q, k, sigma_x, sigma_y, A, f0, seed_value=None):
    """
    Input: 
        T          - Sample size
        p          - Number of features
        q          - Number of targets
        k          - Number of factors (components)
        sigma_x    - Scale of features
        sigma_y    - Scale of targets
        A          - Coefficients used to generate dynamics (k x k matrix)
        f0         - Initial condition used to generate dynamics (length k vector)
        seed_value - If None, random seed used every time. Otherwise, seed passed to RNG
    Output:
        X - Features (T x p matrix)
        Y - Targets  (T x q matrix)
        F - Scores   (T x k matrix)
        P - Loadings of feature equation (p x k matrix)
        Q - Loadings of target equation (q x k matrix)
    """
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = np.zeros([T, k])
    v = rng.normal(size = [T, k])
    F[0] = A @ f0 + v[0]
    for t in range(1, T):
        F[t] = A @ F[t-1] + v[t]
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [T, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [T, q])
    
    return X, Y, F, P, Q
