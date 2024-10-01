import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def generate_synthetic_data(n, p, q, k, sigma_x, sigma_y):
    # Generate latent variables
    rng = np.random.default_rng()
    F = rng.normal(size = [n, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [n, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [n, q])
    
    return X, Y, F, P, Q

def generate_synthetic_data_system(n, p, q, k, Sigma):
    # Generate latent variables
    rng = np.random.default_rng()
    F = rng.normal(size = [n, k])
    
    # Generate loadings
    d = p + q
    L = rng.normal(size = [d, k])
    
    # Generate predictor and response variables with added noise
    epsilon = rng.multivariate_normal(mean = np.zeros(d), cov = Sigma, size = n)
    Z = F @ L.T + epsilon
    
    # Return data, factors and loadings
    X = Z[:, :p]
    Y = Z[:, p:]
    P = L[:p]
    Q = L[p:]
    return X, Y, F, P, Q

def generate_synthetic_missing_data(n, p, q, k, sigma_x, sigma_y, proportion, return_nan=True):
    # Generate latent variables
    rng = np.random.default_rng()
    F = rng.normal(size = [n, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [n, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [n, q])
    
    # Select indices to turn into missing observations
    X_size = n * p
    Y_size = n * q
    num_missing_X = int(proportion * X_size)
    num_missing_Y = int(proportion * Y_size)
    missing_indices_X = rng.choice(X_size, num_missing_X, replace=False, shuffle=False)
    missing_indices_Y = rng.choice(Y_size, num_missing_Y, replace=False, shuffle=False)

    # Transform indices of missing observations to a mask over the data matrices
    missing_indices_X = np.unravel_index(missing_indices_X, [n, p])
    missing_indices_Y = np.unravel_index(missing_indices_Y, [n, q])
    X_missing_mask = np.zeros_like(X, dtype="bool")
    Y_missing_mask = np.zeros_like(Y, dtype="bool")
    X_missing_mask[missing_indices_X] = True
    Y_missing_mask[missing_indices_Y] = True

    # Return type changes depending on 'return_nan'
    if return_nan:
        # Set appropriate entries to missing and return data matrices with missing observations
        X[X_missing_mask] = np.nan
        Y[Y_missing_mask] = np.nan
        return X, Y, F, P, Q
    else:
        # Return the data as a masked object
        X = np.ma.MaskedArray(data=X, mask=X_missing_mask, fill_value=0.0)
        Y = np.ma.MaskedArray(data=Y, mask=Y_missing_mask, fill_value=0.0)
        return X, Y, F, P, Q

def PPLS_stacked(X, Y, k, tolerance = 1e-6,  max_iter = 1000, V_prior = None):
    # Obtain sizes
    # X is n x p; Y is n x q; Factors assumed as n x k
    n, p = X.shape
    _, q = Y.shape
    d = p + q

    # Initial values for the parameters
    rng = np.random.default_rng()
    if V_prior is None:
        V_prior = np.eye(k)
        V_prior_inv = np.eye(k)
    else:
        V_prior_inv = np.linalg.inv(V_prior)
    L0 = rng.multivariate_normal(mean = np.zeros(k), cov = V_prior, size = d)
    sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
    sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

    # Center and scale X and Y separately before stacking
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
    Z = np.hstack([X, Y])

    # Start EM algorithm main loop
    for _ in range(max_iter):
        # Expectation step: Update posterior paramater for factors
        L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
        Sigma = np.linalg.inv(V_prior_inv + L0.T @ L_scaled)
        M = Z @ L_scaled @ Sigma
        V = n*Sigma + M.T @ M
        
        # Maximization step: Update factor loadings and variances
        L1 = np.linalg.solve(V, M.T @ Z).T
        sigma2_x1 = (1/(n * p)) * (np.trace(X.T @ X) - np.trace(P.T @ P @ V))
        sigma2_y1 = (1/(n * q)) * (np.trace(Y.T @ Y) - np.trace(Q.T @ Q @ V))

        # Calculate distance between iterates
        L_distance = np.linalg.norm(L1 - L0, "fro")
        sigma_distance = np.abs(sigma2_x1 - sigma2_x0) + np.abs(sigma2_y1 - sigma2_y0)
        theta_distance = L_distance + sigma_distance
        # r2_distance = np.abs(r2_old - r2_new)
        
        # Check convergence condition: Break if achieved, otherwise update for a new pass of EM
        converged = (theta_distance <= tolerance)#  or (r2_distance <= tolerance)
        if converged:
            break
        else:
            # Prepare values for next iteration
            L0 = L1
            sigma2_x0 = sigma2_x1
            sigma2_y0 = sigma2_y1
            # r2_old = r2_new

    # At convergence, return final estimate of factors and prediction
    return L1, sigma2_x1, sigma2_y1

def PPLS_Stacked_Predict(X, Y, L, sigma2_x, sigma2_y, V_prior = None):
    _, p = X.shape
    _, k = L.shape
    if V_prior is None:
        V_prior_inv = np.eye(k)
    else:
        V_prior_inv = np.linalg.inv(V_prior)
    Z = np.hstack([X, Y])
    L_scaled = np.vstack([L[:p] / sigma2_x, L[p:] / sigma2_y])
    M = np.linalg.solve(V_prior_inv + L.T @ L_scaled, L_scaled.T @ Z.T).T
    return(M @ Q.T)

def PPLS_stacked_system(X, Y, k, tolerance = 1e-6, max_iter = 1000, V_prior = None):
    # Obtain sizes
    # X is n x p; Y is n x q; Factors assumed as n x k
    n, p = X.shape
    _, q = Y.shape
    d = p + q

    # Stack, center and scale variables before processing
    Z = np.hstack([X, Y])
    # Z = (Z - Z.mean(axis = 0)) / Z.std(axis = 0)
    S = (1/n) * (Z.T @ Z)    # Sample covariance matrix

    # Initial values for the parameters
    rng = np.random.default_rng()
    if V_prior is None:
        V_prior = np.eye(k)
        V_prior_inv = np.eye(k)
    else:
        V_prior_inv = np.linalg.inv(V_prior)
    L0 = rng.multivariate_normal(mean = np.zeros(k), cov = V_prior, size = d)
    Sigma0 = S

    # Start EM algorithm main loop
    r2_old = 0.0
    r2_list = []      # List of achieved R-squared values
    for _ in range(max_iter):
        # Expectation step: Update posterior parameters for factors
        L_scaled = np.linalg.solve(Sigma0, L0)
        Omega = np.linalg.inv(V_prior_inv + L0.T @ L_scaled)
        C = L_scaled @ Omega
        V = Omega + C.T @ S @ C

        # Maximization step: Update factor loadings and variances
        L1 = np.transpose(np.linalg.solve(V, C.T @ S))
        Sigma1 = S - S @ C @ L1.T
        L_scaled = np.linalg.solve(Sigma1, L1)

        # Predict using factors and obtain R-squared
        M = np.transpose(np.linalg.solve(V_prior_inv + L1.T @ L_scaled, L_scaled.T @ Z.T))
        Y_hat = M @ Q.T
        r2_values = [r2_score(Y[:, j], Y_hat[:, j]) for j in range(q)]
        r2_new = np.mean(r2_values)
        r2_list.append(r2_new)
        
        # Test convergence using distance between iterates and R2 increase
        L_difference = np.linalg.norm(L1 - L0, 2)
        Sigma_difference = np.linalg.norm(Sigma1 - Sigma0, 2)
        theta_difference = L_difference + Sigma_difference
        r2_distance = np.abs(r2_old - r2_new)
        converged = (theta_difference <= tolerance) or (r2_distance <= tolerance)
        if converged:
            break
        else:
            # Prepare values for next iteration
            L0 = L1
            Sigma0 = Sigma1
            r2_old = r2_new

    # At convergence, return final estimate of factors and prediction
    return M, Y_hat, r2_list, L1, Sigma1

def PPLS_stacked_missing(X, Y, k, tolerance = 1e-6,  max_iter = 1000, V_prior = None):
    # X is n x p; Y is n x q; Factors assumed as n x k
    n, p = X.shape
    _, q = Y.shape
    d = p + q
    
    # Stack variables and first round of imputing missing values (with mean)
    Z = np.hstack([X, Y])
    # Z = (Z - Z.mean(axis = 0)) / Z.std(axis = 0)
    Z_missing_idx = np.isnan(Z)
    Z_means = np.mean(Z, axis=0, where = np.logical_not(Z_missing_idx))
    Z_vars = np.var(Z, axis=0, where = np.logical_not(Z_missing_idx))
    for j in range(d):
        Z[Z_missing_idx[:, j], j] = Z_means[j]

    # Initial values for the parameters
    rng = np.random.default_rng()
    if V_prior is None:
        V_prior = np.eye(k)
        V_prior_inv = np.eye(k)
    else:
        V_prior_inv = np.linalg.inv(V_prior)
    L0 = rng.multivariate_normal(mean = np.zeros(k), cov = V_prior, size = d)
    sigma2_x0 = np.mean(Z_vars[:p])         # Mean variance across features
    sigma2_y0 = np.mean(Z_vars[p:])         # Mean variance across targets

    # Start EM algorithm main loop
    r2_old = 0.0
    r2_list = []      # List of achieved R-squared values
    for _ in range(max_iter):
        # Expectation step: Update posterior paramater for factors
        P = L0[:p]
        Q = L0[p:]
        L_scaled = np.vstack([P / sigma2_x0, Q / sigma2_y0])
        Omega = np.linalg.inv(V_prior_inv + L0.T @ L_scaled)
        M = Z @ L_scaled @ Omega
        V = Omega + M.T @ M
        
        # Impute missing data using current predicted values
        Z_hat = M @ L0.T
        Z[Z_missing_idx] = Z_hat[Z_missing_idx]
        X = Z[:, :p]
        Y = Z[:, p:]

        # Maximization step: Update factor loadings and variances
        L1 = np.transpose(np.linalg.solve(V, M.T @ Z))
        P = L1[:p]
        Q = L1[p:]
        # sigma2_x1 = (1/p) * np.trace(S[:p, :p] - S[:p] @ C @ P.T)
        # sigma2_y1 = (1/q) * np.trace(S[p:, p:] - S[p:] @ C @ Q.T)
        sigma2_x1 = (1/(n * p)) * (np.linalg.norm(X - M @ P.T, 'fro')**2 + (1/p) * np.trace(P.T @ P @ Omega))
        sigma2_y1 = (1/(n * q)) * (np.linalg.norm(Y - M @ Q.T, 'fro')**2 + (1/p) * np.trace(Q.T @ Q @ Omega))
        L_scaled = np.vstack([P / sigma2_x1, Q / sigma2_y1])
        
        # Predict using factors and obtain R-squared
        M = np.transpose(np.linalg.solve(V_prior_inv + L1.T @ L_scaled, L_scaled.T @ Z.T))
        Y_hat = M @ Q.T
        r2_values = [r2_score(Y[:, j], Y_hat[:, j]) for j in range(q)]
        r2_new = np.mean(r2_values)
        r2_list.append(r2_new)

        # Calculate distance between iterates
        L_distance = np.linalg.norm(L1 - L0, "fro")
        sigma_distance = np.abs(sigma2_x1 - sigma2_x0) + np.abs(sigma2_y1 - sigma2_y0)
        theta_distance = L_distance + sigma_distance
        r2_distance = np.abs(r2_old - r2_new)
        
        # Check convergence condition: Break if achieved, otherwise update for a new pass of EM
        converged = (theta_distance <= tolerance) or (r2_distance <= tolerance)
        if converged:
            break
        else:
            # Prepare values for next iteration
            L0 = L1
            sigma2_x0 = sigma2_x1
            sigma2_y0 = sigma2_y1
            r2_old = r2_new

    # At convergence, return final estimate of factors and prediction
    return M, Y_hat, r2_list, L1, sigma2_x1, sigma2_y1

# Run a test of the commands with homoskedastic data
n = 1000
p = 5
q = 2
k = 3
sigma_x = 1.0
sigma_y = 1.0
X, Y, F, P, Q = generate_synthetic_data(n, p, q, k, sigma_x, sigma_y)
F_hat, Y_hat, r2_list, L_hat, sigma2_x_hat, sigma2_y_hat = PPLS_stacked(X, Y, k)
print("Homoskedastic data, Homoskedastic PPLS")
print(np.array([sigma2_x_hat, sigma2_y_hat]))
print(r2_list)

plt.figure()
plt.plot(r2_list, "s-b")
plt.title("R-squared as EM progresses: Homoskedastic data, Homoskedastic PPLS")
plt.xlabel("Iterations")
plt.ylabel("$R^2$")
plt.show()

F_hat, Y_hat, r2_list, L_hat, Sigma_hat = PPLS_stacked_system(X, Y, k)
print("Homoskedastic data, System PPLS")
print(Sigma_hat)
print(r2_list)

plt.figure()
plt.plot(r2_list, "s-b")
plt.title("R-squared as EM progresses: Homoskedastic data, System PPLS")
plt.xlabel("Iterations")
plt.ylabel("$R^2$")
plt.show()

# Run a test of the commands with system correlated data
Sigma = Sigma_hat
X, Y, F, P, Q = generate_synthetic_data_system(n, p, q, k, Sigma)
F_hat, Y_hat, r2_list, L_hat, sigma2_x_hat, sigma2_y_hat = PPLS_stacked(X, Y, k)
print("Correlated data, Homoskedastic PPLS")
print(np.array([sigma2_x_hat, sigma2_y_hat]))
print(r2_list)

plt.figure()
plt.plot(r2_list, "s-b")
plt.title("R-squared as EM progresses: Correlated data, Homoskedastic PPLS")
plt.xlabel("Iterations")
plt.ylabel("$R^2$")
plt.show()

F_hat, Y_hat, r2_list, L_hat, Sigma_hat = PPLS_stacked_system(X, Y, k)
print("Correlated data, System PPLS")
print(Sigma_hat)
print(r2_list)

plt.figure()
plt.plot(r2_list, "s-b")
plt.title("R-squared as EM progresses: Correlated data, System PPLS")
plt.xlabel("Iterations")
plt.ylabel("$R^2$")
plt.show()

# Test the command with missing data
proportion = 0.2
X, Y, F, P, Q = generate_synthetic_missing_data(n, p, q, k, sigma_x, sigma_y, proportion, return_nan=False)
F_hat, Y_hat, r2_list, L_hat, sigma2_x_hat, sigma2_y_hat = PPLS_stacked(X.filled(), Y.filled(), k)
print("PPLS on Imputed Data")
print(np.array([sigma2_x_hat, sigma2_y_hat]))
print(r2_list)

X.data[X.mask] = np.nan
Y.data[Y.mask] = np.nan
F_hat, Y_hat, r2_list, L_hat, sigma2_x_hat, sigma2_y_hat = PPLS_stacked_missing(X.data, Y.data, k)
print("PPLS EM Algorithm for Missing Data")
print(np.array([sigma2_x_hat, sigma2_y_hat]))
print(r2_list)
