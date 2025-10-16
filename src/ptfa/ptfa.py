import numpy as np
from scipy import linalg, sparse
from sklearn.metrics import r2_score

# Class of methods introduced in the paper
class ProbabilisticTFA:
    """
    Probabilistic Targeted Factor Analysis (PTFA) class for fitting and predicting using a probabilistic model.
    Attributes:
        n_components (int):              Number of components (factors) to estimate.
        P (np.ndarray):                  Estimated loadings for predictors.
        Q (np.ndarray):                  Estimated loadings for targets.
        sigma2_x (float):                Estimated variance for predictors.
        sigma2_y (float):                Estimated variance for targets.
        factors (np.ndarray):            Predicted factors.
        V_prior (np.ndarray):            Prior covariance matrix for factors.
        max_iter (int):                  Maximum number of iterations for the EM algorithm.
        tolerance (float):               Convergence tolerance for the EM algorithm.
        r2_array (np.ndarray):           Array of R-squared values across iterations if tracking is enabled.
    Methods:
        __init__(self, n_components):
            Initializes the PTFA model with the specified number of components.
        fit(self, X, Y, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, max_iter=1000,
            r2_stop=True, r2_iters=100, rng=None):
            Fits the PTFA model to the given data using the EM algorithm.
            Parameters:
                X (np.ndarray):            Predictor data matrix of shape (T, p).
                Y (np.ndarray):            Target data matrix of shape (T, q).
                standardize (bool):        Whether to standardize the data before fitting.
                V_prior (np.ndarray):      Prior covariance matrix for factors.
                track_r2 (bool):           track R-squared values across iterations.
                tolerance (float):         Convergence tolerance for the EM algorithm.
                max_iter (int):            Maximum number of iterations for the EM algorithm.
                r2_stop (bool):            Whether to stop based on R-squared convergence.
                r2_iters (int):            Number of iterations to consider for R-squared convergence.
                rng (np.random.Generator): Random number generator for reproducibility.
        fitted(self, standardize=True, compute_variance=False):
            Computes the fitted values and optionally the prediction variance.
            Parameters:
                compute_variance (bool): Whether to compute the prediction variance.
            Returns:
                np.ndarray:              Predicted target values.
                np.ndarray (optional):   Prediction variance matrix if compute_variance is True.
        predict(self, X, standardize=True, compute_variance=False):
            Predicts target values using the fitted PTFA model.
            Parameters:
                X (np.ndarray):          Predictor data matrix of shape (T, p).
                standardize (bool):      Whether to standardize the data before predicting.
                compute_variance (bool): Whether to compute the prediction variance.
            Returns:
                np.ndarray: Predicted target values.
                np.ndarray (optional):  Prediction variance matrix if compute_variance is True.
    """
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
                
        # Pre-allocate memory for estimates
        self.P = None
        self.sigma2 = None
        self.factors = None

    def fit(self, X, Y, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25, rng = None):
        # Obtain sizes
        # X is T x p; Y is T x q; Factors assumed as T x k
        T, p = X.shape
        _, q = Y.shape
        d = p + q
        k = self.n_components

        # Main error check on dimensions: small-dimensional problem requires T > d + k and d > k
        if T <= d + k:
            raise ValueError("PTFA requires enough observations and features:\nsample_size > n_predictors + n_targets + n_components and n_predictors + n_targets > n_components.")
        
        # Fill in components of the class controlling algorithm
        self.max_iter = max_iter
        self.tolerance = tolerance
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else linalg.inv(V_prior)
        
        # Obtain indices of missing observations to create masked objects and initial imputation step
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X_missing_flag = np.any(X_missing_index)
            if X_missing_flag:
                X[X_missing_index] = 0.0
        else:
            X_missing_index = X.mask
            X_missing_flag = True
            X = X.filled(fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y_missing_flag = np.any(Y_missing_index)
            if Y_missing_flag:
                Y[Y_missing_index] = 0.0
        else:
            Y_missing_index = Y.mask
            Y_missing_flag = True
            Y = Y.filled(fill_value=0.0)

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - np.mean(Y, axis = 0, where = np.logical_not(Y_missing_index))) / np.std(Y, axis = 0, where = np.logical_not(Y_missing_index))
        Z = np.hstack([X, Y])
        
        # Initial values for the parameters
        if rng is None:
            rng = np.random.default_rng()
        L0 = rng.normal(size = [d, k])
        sigma2_x0 = X.var(axis = 0).mean()    # Mean variance across features
        sigma2_y0 = Y.var(axis = 0).mean()    # Mean variance across targets

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
        
        # Start EM algorithm main loop
        for it in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            Omega = linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Omega

            # If any missing data, update imputation step using current EM fit
            if X_missing_flag:
                X_hat = M @ L0[:p].T
                X[X_missing_index] = X_hat[X_missing_index]
                Z[:, :p] = X
            if Y_missing_flag:
                Y_hat = M @ L0[p:].T
                Y[Y_missing_index] = Y_hat[Y_missing_index]
                Z[:, p:] = Y

            # Maximization step: Update factor loadings and variances
            V = T * Omega + M.T @ M
            L1 = linalg.solve(V, M.T @ Z).T
            P1 = L1[:p]
            Q1 = L1[p:]
            sigma2_x1 = (1/(T * p)) * (np.sum(X**2) - np.trace(P1.T @ P1 @ V))
            sigma2_y1 = (1/(T * q)) * (np.sum(Y**2) - np.trace(Q1.T @ Q1 @ V))

            # Compute distance between iterates
            P_distance = linalg.norm(P1 - L0[:p], "fro")
            Q_distance = linalg.norm(Q1 - L0[p:], "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = max([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                # Save current value of R-squared
                Y_hat = M @ Q1.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

                # # Save parameter values with the best R-squared so far
                # if it == 0 or np.mean(r2_values) >= np.mean(r2_list[-2]):
                #     best_P, best_Q, best_sigma2_x, best_sigma2_y = (P1, Q1, sigma2_x1, sigma2_y1)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and it >= r2_iters:
                # Add stopping condition based on history of R-squared across iterations
                r2_convergence = np.allclose(np.mean(r2_list[-r2_iters:]),    # Average of previous r_iters values
                                             np.mean(r2_list[-1]),            # Current value
                                             self.tolerance)
                convergence = convergence or r2_convergence
            if convergence:
                # Break if either distance between each estimate or R-squared decrease is small
                break
            else:
                # Prepare values for next iteration if convergence not reached
                L0 = L1
                sigma2_x0 = sigma2_x1
                sigma2_y0 = sigma2_y1
        
        # Update values of the class with results from EM algorithm
        self.P = P1
        self.Q = Q1
        self.sigma2_x = sigma2_x1
        self.sigma2_y = sigma2_y1
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.factors = M
        
    def fitted(self, compute_variance = False):
        # PTFA prediction in-sample:
        Y_hat = self.factors @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            Omega_inverse = self.V_prior_inv + self.P.T @ (self.P / self.sigma2_x) + self.Q.T @ (self.Q / self.sigma2_y)
            Y_hat_variance = self.Q @ linalg.solve(Omega_inverse, self.Q.T)
            return Y_hat, Y_hat_variance

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations and impute using EM fit
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X_hat = self.factors @ self.P.T
                X[X_missing_index] = X_hat[X_missing_index]
        else:
            X_missing_index = X.mask
            if np.any(X_missing_index):
                X_hat = self.factors @ self.P.T
                X = X.filled(X_hat)            
        
        # Center and scale predictors if required
        h = X.shape[0]
        if standardize and h > 1:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))

        # Obtains predicted factors using X only: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x
        Omega_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        F_predicted = linalg.solve(Omega_X_inverse, P_scaled.T @ X.T).T

        # PTFA out-of-sample prediction:
        Y_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            Y_hat_variance = self.sigma2_y * np.eye(q) + self.Q @ linalg.solve(Omega_X_inverse, self.Q.T)
            return Y_hat, Y_hat_variance

class ProbabilisticTFA_StochasticVolatility:
    """
    Probabilistic Targeted Factor Analysis (PTFA) with Stochastic Volatility class for
    fitting and predicting using time-varying volatilities.
    
    This class extends the basic PTFA model to handle time-varying volatilities using separate
    EWMA (Exponentially Weighted Moving Average) smoothing processes for the predictor and target equations.
    
    Attributes:
        n_components (int):     Number of components (factors) to estimate.
        P (np.ndarray):         Estimated loadings for predictors.
        Q (np.ndarray):         Estimated loadings for targets.
        sigma2_x (np.ndarray):  Time-varying volatilities for predictors equations of shape (T,).
        sigma2_y (np.ndarray):  Time-varying volatilities for targets equations of shape (T,).
        factors (np.ndarray):   Predicted factors.
        V_prior (np.ndarray):   Prior covariance matrix for factors.
        max_iter (int):         Maximum number of iterations for the EM algorithm.
        tolerance (float):      Convergence tolerance for the EM algorithm.
        ewma_lambda_x (float):  EWMA smoothing parameter for predictors (default: 0.94).
        ewma_lambda_y (float):  EWMA smoothing parameter for targets.
        r2_array (np.ndarray):  Array of R-squared values across iterations if tracking is enabled.
    
    Methods:
        __init__(self, n_components):
            Initializes the PTFA with Stochastic Volatility model with the specified number of components.
        fit(self, X, Y, standardize=True, ewma_lambda_x=0.94, ewma_lambda_y=None, V_prior=None, 
            track_r2=True, tolerance=1e-6, max_iter=1000, r2_stop=True, r2_iters=100, rng=None):
            Fits the PTFA model with time-varying volatilities to the given data using an EM algorithm.
            Parameters:
                X (np.ndarray):            Predictor data matrix of shape (T, p).
                Y (np.ndarray):            Target data matrix of shape (T, q).
                standardize (bool):        Whether to standardize the data before fitting.
                ewma_lambda_x (float):     EWMA smoothing parameter for predictors volatility.
                ewma_lambda_y (float):     EWMA smoothing parameter for targets volatility (defaults to ewma_lambda_x).
                V_prior (np.ndarray):      Prior covariance matrix for factors.
                track_r2 (bool):           Track R-squared values across iterations.
                tolerance (float):         Convergence tolerance for the EM algorithm.
                max_iter (int):            Maximum number of iterations for the EM algorithm.
                r2_stop (bool):            Whether to stop based on R-squared convergence.
                r2_iters (int):            Number of iterations to consider for R-squared convergence.
                rng (np.random.Generator): Random number generator for reproducibility.
        fitted(self, compute_variance=False):
            Computes the fitted values and optionally the prediction variance.
            Parameters:
                compute_variance (bool):   Whether to compute the prediction variance.
            Returns:
                np.ndarray:                Predicted target values.
                np.ndarray (optional):     Time-varying prediction variance tensor if compute_variance is True.
        predict(self, X, standardize=True, compute_variance=False):
            Predicts target values using the fitted PTFA model with stochastic volatility.
            Parameters:
                X (np.ndarray):            Predictor data matrix of shape (T, p).
                standardize (bool):        Whether to standardize the data before predicting.
                compute_variance (bool):   Whether to compute the prediction variance.
            Returns:
                np.ndarray:                Predicted target values.
                np.ndarray (optional):     Prediction variance matrix if compute_variance is True.
    """
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.factors = None
        self.Omega = None

    def fit(self, X, Y, standardize = True, ewma_lambda_x = 0.94, ewma_lambda_y = None, V_prior = None,
            track_r2 = True, tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25, rng = None):
        # Obtain sizes
        # X is T x p; Y is T x q; Factors assumed as T x k
        T, p = X.shape
        _, q = Y.shape
        d = p + q
        k = self.n_components

        # Main error check on dimensions: small-dimensional problem requires T > d + k and d > k
        if T <= d + k:
            raise ValueError("PTFA requires enough observations and features:\nsample_size > n_predictors + n_targets + n_components and n_predictors + n_targets > n_components.")

        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance          # EM stopping tolerance
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else linalg.inv(V_prior)

        # Specific for stochastic volatility: EWMA smoothing parameter for feature process and targets
        self.ewma_lambda_x = ewma_lambda_x 
        self.ewma_lambda_y = ewma_lambda_y if ewma_lambda_y is not None else ewma_lambda_x
        
       # Obtain indices of missing observations to create masked objects and initial imputation step
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X_missing_flag = np.any(X_missing_index)
            if X_missing_flag:
                X[X_missing_index] = 0.0
        else:
            X_missing_index = X.mask
            X_missing_flag = True
            X = X.filled(fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y_missing_flag = np.any(Y_missing_index)
            if Y_missing_flag:
                Y[Y_missing_index] = 0.0
        else:
            Y_missing_index = Y.mask
            Y_missing_flag = True
            Y = Y.filled(fill_value=0.0)

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - np.mean(Y, axis = 0, where = np.logical_not(Y_missing_index))) / np.std(Y, axis = 0, where = np.logical_not(Y_missing_index))
        Z = np.hstack([X, Y])

        # Initial values for the parameters (time-varying volatilities start constant)
        if rng is None:
            rng = np.random.default_rng()
        L0 = rng.normal(size = [d, k])
        sigma2_x_initial = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y_initial = np.var(Y, axis = 0).mean()    # Mean variance across targets
        sigma2_x = np.full(T, sigma2_x_initial)
        sigma2_y = np.full(T, sigma2_y_initial)

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
        
        # Start EM algorithm main loop
        M = np.zeros([T, k])
        Omega = np.zeros([T, k, k])
        V = np.zeros([T, k, k])
        for it in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            for t in range(T):
                P0 = L0[:p]
                Q0 = L0[p:]
                L_scaled_t = np.vstack([P0 / sigma2_x[t], Q0 / sigma2_y[t]])
                Omega[t] = linalg.inv(self.V_prior_inv + L0.T @ L_scaled_t)
                M[t] = Z[t] @ L_scaled_t @ Omega[t]
                V[t] = Omega[t] + np.outer(M[t], M[t])

            # If any missing data, update imputation step using current EM fit
            if X_missing_flag:
                X_hat = M @ L0[:p].T
                X[X_missing_index] = X_hat[X_missing_index]
                Z[:, :p] = X
            if Y_missing_flag:
                Y_hat = M @ L0[p:].T
                Y[Y_missing_index] = Y_hat[Y_missing_index]
                Z[:, p:] = Y

            # Maximization step
            # Factor loadings update
            V_x = np.einsum('t,tij->ij', 1/sigma2_x, V)
            V_y = np.einsum('t,tij->ij', 1/sigma2_y, V)
            M_x = (M / sigma2_x[:, np.newaxis]).T @ X
            M_y = (M / sigma2_y[:, np.newaxis]).T @ Y
            P1 = linalg.solve(V_x, M_x).T
            Q1 = linalg.solve(V_y, M_y).T
            L1 = np.vstack([P1, Q1])

            # Update volatilities using EWMA
            Z_hat = M @ L1.T
            residuals_Z = Z - Z_hat
            sigma2_x[0] = (1/p) * (np.sum(residuals_Z[0, :p]**2) + np.trace(P1.T @ P1 @ Omega[0]))
            sigma2_y[0] = (1/q) * (np.sum(residuals_Z[0, p:]**2) + np.trace(Q1.T @ Q1 @ Omega[0]))
            for t in range(1, T):
                hat_sigma2_x_t = (1/p) * (np.sum(residuals_Z[t, :p]**2) + np.trace(P1.T @ P1 @ Omega[t]))
                hat_sigma2_y_t = (1/q) * (np.sum(residuals_Z[t, p:]**2) + np.trace(Q1.T @ Q1 @ Omega[t]))
                sigma2_x[t] = self.ewma_lambda_x * sigma2_x[t-1] + (1 - self.ewma_lambda_x) * hat_sigma2_x_t
                sigma2_y[t] = self.ewma_lambda_y * sigma2_y[t-1] + (1 - self.ewma_lambda_y) * hat_sigma2_y_t
            
            # Compute distance between iterates
            P_distance = linalg.norm(P1 - P0, "fro")
            Q_distance = linalg.norm(Q1 - Q0, "fro")
            theta_distance = max([P_distance, Q_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                r2_values = r2_score(Y, Z_hat[:, p:], multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and it >= r2_iters:
                # Add stopping condition based on history of R-squared across iterations
                r2_convergence = np.allclose(np.mean(r2_list[-r2_iters:]),    # Average of previous r_iters values
                                             np.mean(r2_list[-1]),            # Current value
                                             self.tolerance)
                convergence = convergence or r2_convergence
            if convergence:
                # Break if distance between each estimate is less than a tolerance
                break
            else:
                # Prepare values for next iteration if convergence not reached
                L0 = L1
                
        # Update final values of the class with results from EM algorithm
        self.P = P1
        self.Q = Q1
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.factors = M
        self.Omega = Omega
        
    def fitted(self, compute_variance = False):
        # PTFA prediction in-sample:
        Y_hat = self.factors @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            T = self.factors.shape[0]
            Y_hat_variance = np.zeros([T, q, q])
            for t in range(T):
                Y_hat_variance[t] = self.Q @ self.Omega[t] @ self.Q.T
            return Y_hat, np.squeeze(Y_hat_variance)

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations and impute using EM fit
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X_hat = self.factors @ self.P.T
                X[X_missing_index] = X_hat[X_missing_index]
        else:
            X_missing_index = X.mask
            if np.any(X_missing_index):
                X_hat = self.factors @ self.P.T
                X = X.filled(X_hat)            
        
        # Center and scale predictors if required
        h = X.shape[0]
        if standardize and h > 1:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
        
        # Obtains predicted factors and volatilities using X only: F_predicted = M (Posterior mean of factors)
        k = self.n_components
        F_predicted = np.zeros([h, k])
        
        # Initialize volatility arrays
        hat_sigma2_x_t = self.ewma_lambda_x * self.sigma2_x[-1] + (1 - self.ewma_lambda_x) * np.mean(self.sigma2_x)
        hat_sigma2_y_t = self.ewma_lambda_y * self.sigma2_y[-1] + (1 - self.ewma_lambda_y) * np.mean(self.sigma2_y)

        # Update factors and volatilities using common estimate
        P_scaled_t = self.P / hat_sigma2_x_t
        Omega_X_t = linalg.inv(self.V_prior_inv + self.P.T @ P_scaled_t)
        F_predicted = X @ P_scaled_t @ Omega_X_t

        # PTFA out-of-sample prediction:
        Y_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            # Update variance calculation
            q = self.Q.shape[0]
            Y_hat_variance = hat_sigma2_y_t * np.eye(q) + self.Q @ Omega_X_t @ self.Q.T
            return Y_hat, np.squeeze(Y_hat_variance)

class ProbabilisticTFA_MixedFrequency:
    """
    Probabilistic Targeted Factor Analysis (PTFA) with Mixed Frequency class for handling data at different temporal frequencies.
    
    This class extends the basic PTFA model to handle mixed frequency data where predictors are observed at high frequency
    and targets are observed at low frequency. Supports both regular (constant periods) and irregular (variable periods) sampling.
    
    Attributes:
        n_components (int):              Number of components (factors) to estimate.
        P (np.ndarray):                  Estimated loadings for high-frequency predictors.
        Q (np.ndarray):                  Estimated loadings for low-frequency targets.
        sigma2_x (float):                Variance of high-frequency predictors.
        sigma2_y (float):                Variance of low-frequency targets.
        factors (np.ndarray):            Estimated factors.
        V_prior (np.ndarray):            Prior covariance matrix for factors.
        max_iter (int):                  Maximum number of iterations for the EM algorithm.
        tolerance (float):               Convergence tolerance for the EM algorithm.
        periods (int or np.ndarray):     Frequency ratio(s) between high and low frequency data.
        r2_array (np.ndarray):           Array of R-squared values across iterations if tracking is enabled.
    
    Methods:
        __init__(self, n_components):
            Initializes the PTFA Mixed Frequency model with the specified number of components.
        fit(self, X, Y, periods, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, 
            max_iter=1000, r2_stop=True, r2_iters=100, rng=None):
            Fits the PTFA model to mixed frequency data using the EM algorithm.
            Parameters:
                X (np.ndarray):            High-frequency predictor data matrix of shape (high_frequency_T, p).
                Y (np.ndarray):            Low-frequency target data matrix of shape (low_frequency_T, q).
                periods (int or list):     Frequency ratio (e.g., 3 for quarterly) or list of periods for irregular sampling.
                standardize (bool):        Whether to standardize the data before fitting.
                V_prior (np.ndarray):      Prior covariance matrix for factors.
                track_r2 (bool):           Track R-squared values across iterations.
                tolerance (float):         Convergence tolerance for the EM algorithm.
                max_iter (int):            Maximum number of iterations for the EM algorithm.
                r2_stop (bool):            Whether to stop based on R-squared convergence.
                r2_iters (int):            Number of iterations to consider for R-squared convergence.
                rng (np.random.Generator): Random number generator for reproducibility.
        fitted(self, compute_variance=False):
            Computes the fitted values for low-frequency targets and optionally the prediction variance.
            Parameters:
                compute_variance (bool):   Whether to compute the prediction variance.
            Returns:
                np.ndarray:                Predicted low-frequency target values.
                np.ndarray (optional):     Prediction variance tensor if compute_variance is True.
        predict(self, X, standardize=True, compute_variance=False):
            Predicts low-frequency target values using high-frequency predictors.
            Parameters:
                X (np.ndarray):            High-frequency predictor data matrix of shape (high_frequency_T, p).
                standardize (bool):        Whether to standardize the data before predicting.
                compute_variance (bool):   Whether to compute the prediction variance.
            Returns:
                np.ndarray:                Predicted low-frequency target values.
                np.ndarray (optional):     Prediction variance tensor if compute_variance is True.
    """
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.factors = None
        self.periods = None
        
    def fit(self, X, Y, periods, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25, rng = None):
        # Obtain sizes
        # X is assumed inputed as high_frequency_T x p; Y is low_frequency_T x q
        # periods can be either an integer (e.g. 3 for quarterly) or a list of integers
        high_frequency_T, p = X.shape
        low_frequency_T, q = Y.shape
        k = self.n_components
        d = p + q
        single_period = isinstance(periods, int)

        # Main error check on dimensions: small-dimensional problem requires T > d + k and d > k
        if low_frequency_T <= q + k or high_frequency_T <= p + k:
            raise ValueError("PTFA requires enough observations and features:\nsample_size > n_predictors + n_targets + n_components and n_predictors + n_targets > n_components.")

        # Mixed-frequency edge case: high_frequency_T must be equal to sum(periods) if periods is a list
        if not single_period and high_frequency_T != sum(periods):
                raise ValueError("If periods is a list of integers, high_frequency_T must equal sum(periods).")

        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.periods = periods if single_period else np.array(periods)
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else linalg.inv(V_prior)
        
        # Obtain indices of missing observations to create masked objects and initial imputation step
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X_missing_flag = np.any(X_missing_index)
            if X_missing_flag:
                X[X_missing_index] = 0.0
        else:
            X_missing_index = X.mask
            X_missing_flag = True
            X = X.filled(fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y_missing_flag = np.any(Y_missing_index)
            if Y_missing_flag:
                Y[Y_missing_index] = 0.0
        else:
            Y_missing_index = Y.mask
            Y_missing_flag = True
            Y = Y.filled(fill_value=0.0)

        # Center and scale predictors and targets separately
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - np.mean(Y, axis = 0, where = np.logical_not(Y_missing_index))) / np.std(Y, axis = 0, where = np.logical_not(Y_missing_index))
        
        # Initial values for the parameters
        if rng is None:
            rng = np.random.default_rng()
        P0 = rng.normal(size = [p, k])
        Q0 = rng.normal(size = [q, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
    
        # Start EM algorithm main loop
        # Simplify calculations in the case of same frequency ratio for all targets
        period_indices = range(0, high_frequency_T, self.periods) if single_period else np.cumsum([0] + self.periods[:-1])
        for it in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            # Posterior mean and covariance components
            P_scaled = P0 / sigma2_x0
            Q_scaled = Q0 / sigma2_y0
            V_P_inv = linalg.inv(self.V_prior_inv + P0.T @ P_scaled) 
            V_inv = linalg.inv(self.V_prior_inv + P0.T @ P_scaled + Q0.T @ Q_scaled)
            
            # Posterior mean matrix M = E[F | X, Y]
            X_bar = np.add.reduceat(X, period_indices, axis=0) / (self.periods if single_period else self.periods[:, np.newaxis])
            X_tilde = X - np.repeat(X_bar, self.periods, axis=0)
            M_low_frequency = (X_bar @ P_scaled + Y @ Q_scaled) @ V_inv
            M = X_tilde @ P_scaled @ V_P_inv + np.repeat(M_low_frequency, self.periods, axis=0)

            # Sums over posterior covariance block components
            M_sum = np.add.reduceat(M, period_indices, axis=0)
            M_bar = M_sum / (self.periods if single_period else self.periods[:, np.newaxis])
            V_diagsum = (high_frequency_T - low_frequency_T) * V_P_inv + low_frequency_T * V_inv + M.T @ M
            V_allsum = low_frequency_T * V_inv + M_sum.T @ M_bar

            # Update missing data using current EM fitted values
            if X_missing_flag:
                X_hat = M @ P0.T
                X[X_missing_index] = X_hat[X_missing_index]
            if Y_missing_flag:
                Y_hat = M_bar @ Q0.T
                Y[Y_missing_index] = Y_hat[Y_missing_index]
            
            # Maximization step: Update factor loadings and variances
            P1 = linalg.solve(V_diagsum, M.T @ X).T
            Q1 = linalg.solve(V_allsum, M_sum.T @ Y).T
            Y_periods = Y * (self.periods if single_period else self.periods[:, np.newaxis])
            sigma2_x1 = (1/(high_frequency_T * p)) * (np.sum(X**2) - np.trace(P1.T @ P1 @ V_diagsum))
            sigma2_y1 = (1/(low_frequency_T * q)) * (np.sum(Y * Y_periods) - np.trace(Q1.T @ Q1 @ V_allsum))
            
            # Compute distance between iterates
            P_distance = linalg.norm(P1 - P0, "fro")
            Q_distance = linalg.norm(Q1 - Q0, "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = max([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                Y_hat = M_bar @ Q1.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and it >= r2_iters:
                # Add stopping condition based on history of R-squared across iterations
                r2_convergence = np.allclose(np.mean(r2_list[-r2_iters:]),    # Average of previous r_iters values
                                             np.mean(r2_list[-1]),            # Current value
                                             self.tolerance)
                convergence = convergence or r2_convergence
            if convergence:
                # Break if distance between each estimate is less than a tolerance
                break
            else:
                # Prepare values for next iteration if convergence not reached
                P0 = P1
                Q0 = Q1
                sigma2_x0 = sigma2_x1
                sigma2_y0 = sigma2_y1
        
        # Update values of the class with results from EM algorithm
        self.P = P1
        self.Q = Q1
        self.sigma2_x = sigma2_x1
        self.sigma2_y = sigma2_y1
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.high_frequency_factors = M
        self.low_frequency_factors = M_bar

    def fitted(self, compute_variance = False):
        # Variance returned as np.squeeze of final variance object to remove unnecessary dimensions in case of single targets
        # PTFA prediction in-sample:
        Y_hat = self.low_frequency_factors @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            # Obtain sizes and period flag
            q = self.Q.shape[0]
            low_frequency_T = self.low_frequency_factors.shape[0]
            single_period = isinstance(self.periods, int)

            # Variance of each low-frequency prediction
            V_PQ = self.V_prior_inv + self.P.T @ (self.P / self.sigma2_x) + self.Q.T @ (self.Q / self.sigma2_y)
            Y_latent_variance = self.Q @ linalg.solve(V_PQ, self.Q.T)
            if single_period:
                Y_hat_variance = Y_latent_variance / self.periods
            else:
                Y_hat_variance = np.tile(Y_latent_variance, [low_frequency_T, q, q]) / self.periods[:, np.newaxis, np.newaxis]
            return Y_hat, np.squeeze(Y_hat_variance)
    
    def predict(self, X, periods, standardize = True, compute_variance = False):
        # Obtain necessary sizes and period indices
        single_period = isinstance(periods, int)
        if not single_period:
            periods = np.array(periods)
        high_frequency_h = X.shape[0]
        period_indices = range(0, high_frequency_h, periods) if single_period else np.cumsum([0] + periods[:-1])

        # Obtain indices of missing observations and impute using EM fit
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X_hat = self.high_frequency_factors @ self.P.T
                X[X_missing_index] = X_hat[X_missing_index]
        else:
            X_missing_index = X.mask
            if np.any(X_missing_index):
                X_hat = self.high_frequency_factors @ self.P.T
                X = X.filled(X_hat)  
        
        # Center and scale predictors if required
        h = X.shape[0]
        if standardize and h > 1:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))

        # Obtains predicted factors using X only: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x
        V_P = self.V_prior_inv + self.P.T @ P_scaled
        F_predicted = linalg.solve(V_P, P_scaled.T @ X.T).T

        # PTFA prediction out of sample:
        F_predicted_sum = np.add.reduceat(F_predicted, period_indices, axis=0)
        F_predicted_bar = F_predicted_sum / (periods if single_period else periods[:, np.newaxis])
        Y_hat = F_predicted_bar @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            # Obtain sizes and period flag
            q = self.Q.shape[0]
            low_frequency_h = Y_hat.shape[0]

            # Variance of each low-frequency prediction
            Y_latent_variance = self.Q @ linalg.solve(V_P, self.Q.T) + self.sigma2_y * np.eye(q)
            if single_period:
                Y_hat_variance = Y_latent_variance / periods
            else:
                Y_hat_variance = np.tile(Y_latent_variance, [low_frequency_h, q, q]) / periods[:, np.newaxis, np.newaxis]
            return Y_hat, np.squeeze(Y_hat_variance)

class ProbabilisticTFA_DynamicFactors:
    """
    Probabilistic Targeted Factor Analysis (PTFA) with Dynamic Factors class for modeling time-dependent latent factors.
    
    This class extends the basic PTFA model to include autoregressive dynamics in the latent factors, allowing for
    temporal dependencies and improved forecasting capabilities. Uses efficient banded matrix operations for scalability.
    
    Attributes:
        n_components (int):              Number of components (factors) to estimate.
        P (np.ndarray):                  Estimated loadings for predictors.
        Q (np.ndarray):                  Estimated loadings for targets.
        sigma2_x (float):                Estimated variance for predictors.
        sigma2_y (float):                Estimated variance for targets.
        A (np.ndarray):                  Autoregressive coefficient matrix for factor dynamics.
        f0 (np.ndarray):                 Initial condition for the factors.
        factors (np.ndarray):            Predicted factors with temporal dynamics.
        V_prior (np.ndarray):            Prior covariance matrix for factors.
        V_prior_inv (np.ndarray):        Inverse of the prior covariance matrix.
        max_iter (int):                  Maximum number of iterations for the EM algorithm.
        tolerance (float):               Convergence tolerance for the EM algorithm.
        r2_array (np.ndarray):           Array of R-squared values across iterations if tracking is enabled.
    
    Methods:
        __init__(self, n_components):
            Initializes the PTFA Dynamic Factors model with the specified number of components.
        bands_cholesky(self, cholesky_banded, desired_bands=0):
            Computes the inverse elements of a banded matrix using its Cholesky decomposition.
            Parameters:
                cholesky_banded (np.ndarray): Banded Cholesky decomposition matrix.
                desired_bands (int):          Number of bands to compute (0 for diagonal only).
            Returns:
                np.ndarray:                   Banded matrix with inverse elements.
        fit(self, X, Y, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, 
            max_iter=1000, r2_stop=True, r2_iters=100, rng=None):
            Fits the PTFA model with dynamic factors to the given data using the EM algorithm.
            Parameters:
                X (np.ndarray):            Predictor data matrix of shape (T, p).
                Y (np.ndarray):            Target data matrix of shape (T, q).
                standardize (bool):        Whether to standardize the data before fitting.
                V_prior (np.ndarray):      Prior covariance matrix for factors.
                track_r2 (bool):           Track R-squared values across iterations.
                tolerance (float):         Convergence tolerance for the EM algorithm.
                max_iter (int):            Maximum number of iterations for the EM algorithm.
                r2_stop (bool):            Whether to stop based on R-squared convergence.
                r2_iters (int):            Number of iterations to consider for R-squared convergence.
                rng (np.random.Generator): Random number generator for reproducibility.
        fitted(self, compute_variance=False):
            Computes the fitted values and optionally the prediction variance.
            Parameters:
                compute_variance (bool):   Whether to compute the prediction variance.
            Returns:
                np.ndarray:                Predicted target values.
                np.ndarray (optional):     Prediction variance matrix if compute_variance is True.
        predict(self, X, standardize=True, compute_variance=False):
            Predicts target values using the fitted PTFA model with dynamic factors.
            Parameters:
                X (np.ndarray):            Predictor data matrix of shape (T, p).
                standardize (bool):        Whether to standardize the data before predicting.
                compute_variance (bool):   Whether to compute the prediction variance.
            Returns:
                np.ndarray:                Predicted target values.
                np.ndarray (optional):     Approximated prediction variance matrix if compute_variance is True.
    """

    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.A = None
        self.f0 = None
        self.factors = None
        self.Omega = None

    def bands_cholesky(self, cholesky_banded, desired_bands=0):
        """
        Input:
            cholesky_banded - (l * k x T * k) banded matrix with the Cholesky decomposition of Omega_inverse
                Cholesky decomposition of Omega^(-1) (T * k x T * k), where k = n_components
                Omega_inverse is assumed banded with lower bandwidth = l * k - 1 or total_bandwidth = l * k - 1
            desired_bands   - Number of total bands to calculate (if set to 0, only calculates main diagonal)
        Output:
            Omega - (desired_bands * k, T * k) banded matrix with inverse elements
                As Omega is symmetric, only need to store the lower-diagonal elements in the end
                Element (i, j) of Omega is located in [desired_bands + i - j, l*k + r] for l, r in {0, ..., k-1} 
        """
        # Pre-allocate final objective using both the upper and lower bands for now
        half_bandwidth, Tk = cholesky_banded.shape # Half bandwidth includes main diagonal (l + 1)
        total_rows = 2 * desired_bands + 1
        Omega = np.zeros([total_rows, Tk])

        # Transform Cholesky decomposition to LDL' decomposition
        cholesky_diagonal = cholesky_banded[0]                 # Given banded structure, diagonal is first row
        cholesky_banded = cholesky_banded / cholesky_diagonal  # Columns are also in place, so can simply divide to invert
        cholesky_diagonal = 1 / cholesky_diagonal**2

        # Main algorithm loop
        max_bandwidth = min(half_bandwidth, desired_bands + 1)
        bandwidth_range = range(1, max_bandwidth)
        for j in reversed(range(Tk)):
            for i in reversed(range(max(j - desired_bands, 0), j+1)):
                save_row_index = desired_bands + i - j  # Omega row index to save current element
                next_row_index = range(save_row_index + 1, save_row_index + max_bandwidth)
                Omega[save_row_index, j] = -np.dot(cholesky_banded[bandwidth_range, i], Omega[next_row_index, j])
                if i == j:
                    # Update diagonal elements by adding contribution from Cholesky diagonal
                    Omega[save_row_index, i] += cholesky_diagonal[i]
                else:
                    # Update symmetric off-diagonal element
                    Omega[desired_bands + j - i, i] = Omega[save_row_index, j]

        # Discard upper set of elements and return
        return Omega[desired_bands:, :]

    def fit(self, X, Y, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25, rng = None):
        # Obtain sizes
        # X is T x p; Y is T x q; Factors assumed as T x k
        T, p = X.shape
        _, q = Y.shape
        d = p + q
        k = self.n_components
        Tk = T * k

        # Main error check on dimensions: small-dimensional problem requires T > d + k and d > k
        if T <= d + k:
            raise ValueError("PTFA requires enough observations and features:\nsample_size > n_predictors + n_targets + n_components and n_predictors + n_targets > n_components.")

        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else linalg.inv(V_prior)
        
        # Obtain indices of missing observations to create masked objects and initial imputation step
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X_missing_flag = np.any(X_missing_index)
            if X_missing_flag:
                X[X_missing_index] = 0.0
        else:
            X_missing_index = X.mask
            X_missing_flag = True
            X = X.filled(fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y_missing_flag = np.any(Y_missing_index)
            if Y_missing_flag:
                Y[Y_missing_index] = 0.0
        else:
            Y_missing_index = Y.mask
            Y_missing_flag = True
            Y = Y.filled(fill_value=0.0)

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - np.mean(Y, axis = 0, where = np.logical_not(Y_missing_index))) / np.std(Y, axis = 0, where = np.logical_not(Y_missing_index))
        Z = np.hstack([X, Y])

        # Initial values for the parameters
        if rng is None:
            rng = np.random.default_rng()
        L0 = rng.normal(size = [d, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets
        A0 = 0.5 * np.eye(k)
        f0_0 = np.zeros(k)

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
        
        # Start EM algorithm main loop
        for it in range(self.max_iter):
            ### Expectation step: Update posterior paramater for factors using sparse matrix computations ---
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            L_scaled_L = L0.T @ L_scaled
            V_times_A = self.V_prior_inv @ A0
            A_V_A = A0.T @ V_times_A
            Omega_0_inv = self.V_prior_inv + L_scaled_L

            # Save posterior precision in sparse representation as intermediate
            sparse_identity = sparse.eye(T, format='csc')
            main_diagonal = sparse.kron(sparse_identity, Omega_0_inv)
            lower_diagonal = sparse.kron(sparse.eye(T, k=-1, format='csc'), -V_times_A)
            sparse_identity[-1, -1] = 0.0   # Remove last element to create last block
            diagonal_correction = sparse.kron(sparse_identity, A_V_A)
            Omega_inv_sparse = main_diagonal + lower_diagonal + diagonal_correction

            # Save posterior precision to a symmetric banded matrix and compute its Cholesky decomposition
            # (Tk x Tk) -> (2k x Tk), only storing the 2k-1 lower diagonal bands + main diagonal
            Omega_inv_cholesky = np.zeros([2 * k, Tk])
            for diagonal in range(2 * k):
                Omega_inv_cholesky[diagonal, :(Tk - diagonal)] = Omega_inv_sparse.diagonal(-diagonal)
            Omega_inv_cholesky = linalg.cholesky_banded(Omega_inv_cholesky, overwrite_ab=True, lower=True)
                
            # Compute posterior mean using banded matrix solver
            M = np.ravel(Z @ L_scaled)
            M[:k] += V_times_A @ f0_0
            M = linalg.cho_solve_banded((Omega_inv_cholesky, True), b = M, overwrite_b=True).reshape([T, k])

            # If any missing data, update imputation step using current EM fit
            if X_missing_flag:
                X_hat = M @ L0[:p].T
                X[X_missing_index] = X_hat[X_missing_index]
                Z[:, :p] = X
            if Y_missing_flag:
                Y_hat = M @ L0[p:].T
                Y[Y_missing_index] = Y_hat[Y_missing_index]
                Z[:, p:] = Y

            ### Maximization step: Update factor loadings and variances ---
            # Calculate banded elements of the posterior covariance using lower-level function
            Omega_banded = self.bands_cholesky(Omega_inv_cholesky, 2 * k - 1)

            # Compute sums over block diagonals of the posterior covariance
            # All diagonal block, (T-1) diagonal blocks, lower diagonal block
            sum_array = np.zeros([3, k, k])
            for j in range(k):
                # Diagonal block from t = 0 to T-1
                save_index = range(j, k)
                row_index = range(k - j)
                column_index_all = range(j, Tk, k)
                Omega_blocks = Omega_banded[np.ix_(row_index, column_index_all)]
                sum_array[0, save_index, j] = np.sum(Omega_blocks, axis = 1)

                # (T-1) diagonal blocks from t = 0 to T-2
                column_index_withoutlast = range(j, Tk - k, k)
                Omega_blocks = Omega_banded[np.ix_(row_index, column_index_withoutlast)]
                sum_array[1, save_index, j] = np.sum(Omega_blocks, axis = 1)

                # Lower diagonal blocks from t = 1 to T-1
                save_index = range(k)
                row_index = range(k - j, 2 * k - j)
                Omega_blocks = Omega_banded[np.ix_(row_index, column_index_withoutlast)]
                sum_array[2, save_index, j] = np.sum(Omega_blocks, axis = 1)
            
            # Fill in missing upper diagonal elements for diagonal blocks
            for blocks in range(2):
                sum_array[blocks][np.triu_indices(k, 1)] = sum_array[blocks][np.tril_indices(k, -1)]

            # Update loadings and error variances
            V_0 = sum_array[0] + M.T @ M
            L1 = linalg.solve(V_0, M.T @ Z).T
            P1 = L1[:p]
            Q1 = L1[p:]
            sigma2_x1 = (1/(T * p)) * (np.sum(X**2) - np.trace(P1.T @ P1 @ V_0))
            sigma2_y1 = (1/(T * q)) * (np.sum(Y**2) - np.trace(Q1.T @ Q1 @ V_0))

            # Update dynamic parameters: Autoregressive coefficients and initial condition
            V_1 = sum_array[1] + M[:(T-1)].T @ M[:(T-1)]
            V_2 = sum_array[2] + M[1:].T @ M[:(T-1)]
            A1 = linalg.solve(V_1, V_2)
            f0_1 = linalg.solve(A1.T @ self.V_prior @ A1, A1.T @ self.V_prior @ M[0])

            # Compute distance between iterates
            P_distance = linalg.norm(P1 - L0[:p], "fro")
            Q_distance = linalg.norm(Q1 - L0[p:], "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            A_distance = linalg.norm(A1 - A0, "fro")
            f0_distance = linalg.norm(f0_0 - f0_1, 2)
            theta_distance = max([P_distance, Q_distance, sigma_x_distance, sigma_y_distance, A_distance, f0_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2:
                Y_hat = M @ Q1.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and it >= r2_iters:
                # Add stopping condition based on history of R-squared across iterations
                r2_convergence = np.allclose(np.mean(r2_list[-r2_iters:]),    # Average of previous r_iters values
                                             np.mean(r2_list[-1]),            # Current value
                                             self.tolerance)
                convergence = convergence or r2_convergence
            if convergence:
                # Break if distance between each estimate is less than a tolerance
                break
            else:
                # Prepare values for next iteration if convergence not reached
                L0 = L1
                sigma2_x0 = sigma2_x1
                sigma2_y0 = sigma2_y1
                A0 = A1
                f0_0 = f0_1
        
        # Update values of the class with results from EM algorithm
        self.P = P1
        self.Q = Q1
        self.sigma2_x = sigma2_x1
        self.sigma2_y = sigma2_y1
        self.A = A1
        self.f0 = f0_1
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.factors = M
        self.Omega = Omega_banded
        
    def fitted(self, compute_variance = False):
        # PTFA prediction in-sample:
        Y_hat = self.factors @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            # Recover components of posterior precision covariance 
            T = self.factors.shape[0]
            q = self.Q.shape[0]
            k = self.n_components
            Y_hat_variance = np.zeros([T, q, q])
            offsets = range(0, -k, -1)

            # Compute variance of each prediction (no covariance between predictions)
            for t in range(T):
                # Extract Omega_{t, t} block from banded representation
                Omega_tt = self.Omega[:k, range(t * k, (t + 1) * k)]
                Omega_tt = sparse.diags(Omega_tt, offsets, shape=(k, k)).toarray()
                Y_hat_variance[t] = self.Q @ Omega_tt @ self.Q.T
            return Y_hat, np.squeeze(Y_hat_variance)

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations and impute using EM fit
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X_hat = self.factors @ self.P.T
                X[X_missing_index] = X_hat[X_missing_index]
        else:
            X_missing_index = X.mask
            if np.any(X_missing_index):
                X_hat = self.factors @ self.P.T
                X = X.filled(X_hat)            
        
        # Center and scale predictors if required
        h = X.shape[0]
        if standardize and h > 1:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
        
        # Compute some necessary quantities using dynamic information estimates
        k = self.n_components
        Tk = self.factors.shape[0] * k
        P_scaled =  self.P / self.sigma2_x
        V_times_A = self.V_prior_inv @ self.A
        A_V_A = self.A.T @ V_times_A
        Omega_inv_X = self.V_prior_inv + P_scaled.T @ self.P

        # Recover Omega_{T, T} and M_T final diagonal block for updating one-step ahead forecast
        Omega_TT = self.Omega[:k, range(Tk-k, Tk)]
        offsets = range(0, -k, -1)
        Omega_TT = sparse.diags(Omega_TT, offsets, shape=(k, k)).toarray()
        Omega_correction = linalg.inv(self.A @ Omega_TT @ self.A.T + self.V_prior)
        M_T = self.factors[-1]

        # Save posterior precision in sparse representation as intermediate
        sparse_identity = sparse.eye(h, format='lil')
        main_diagonal = sparse.kron(sparse_identity, Omega_inv_X)
        lower_diagonal = sparse.kron(sparse.eye(h, k=-1, format='csc'), -V_times_A)
        sparse_identity[-1, -1] = 0.0   # Remove last element to create last block
        diagonal_correction = sparse.kron(sparse_identity, A_V_A)
        sparse_identity[1:, 1:] = 0.0   # Keep only first block for final correction
        first_block_correction = sparse.kron(sparse_identity, Omega_correction - self.V_prior_inv)
        Omega_inv_sparse = main_diagonal + lower_diagonal + diagonal_correction + first_block_correction

        # Save posterior precision to a symmetric banded matrix and compute its Cholesky decomposition
        # (Tk x Tk) -> (2k x Tk), only storing the 2k lower diagonal bands
        Omega_inv_cholesky = np.zeros([2 * k, h * k])
        for diagonal in range(2 * k):
            Omega_inv_cholesky[diagonal, :(h * k - diagonal)] = Omega_inv_sparse.diagonal(-diagonal)
        Omega_inv_cholesky = linalg.cholesky_banded(Omega_inv_cholesky, overwrite_ab=True, lower=True)
                
        # Compute predicted factors using banded matrix solver
        F_predicted = np.ravel(X @ P_scaled)
        F_predicted[:k] += Omega_correction @ self.A @ M_T
        F_predicted = linalg.cho_solve_banded((Omega_inv_cholesky, True), b = F_predicted, overwrite_b=True).reshape([h, k])

        # PTFA out-of-sample prediction:
        Y_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            # Calculate diagnal blocks of from posterior covariance using lower-level function
            Omega_X_banded = self.bands_cholesky(Omega_inv_cholesky, k - 1)

            # Compute variance of each prediction (no covariance between predictions)
            q = self.Q.shape[0]
            Y_hat_variance = np.zeros([h, q, q])
            for t in range(h):
                # Extract Omega_{t, t} block from banded representation
                Omega_X_tt = Omega_X_banded[:k, range(t * k, (t + 1) * k)]
                Omega_X_tt = sparse.diags(Omega_X_tt, offsets, shape=(k, k)).toarray()
                Y_hat_variance[t] = self.sigma2_y * np.eye(q) + self.Q @ Omega_X_tt @ self.Q.T
            return Y_hat, np.squeeze(Y_hat_variance)
        

# Method for comparison
class ProbabilisticPCA:
    """
    Probabilistic Principal Component Analysis (PPCA) class for dimensionality reduction using a probabilistic model.
    
    Implementation follows the original paper by Tipping and Bishop (1999, JRSS-B). This class provides a probabilistic
    approach to PCA that can handle missing data and provides uncertainty estimates, comparable to all PTFA classes.
    
    Attributes:
        n_components (int):              Number of components (factors) to estimate.
        L (np.ndarray):                  Estimated loadings matrix.
        sigma2 (float):                  Estimated isotropic noise variance.
        factors (np.ndarray):            Predicted factors (latent variables).
        V_prior (np.ndarray):            Prior covariance matrix for factors.
        max_iter (int):                  Maximum number of iterations for the EM algorithm.
        tolerance (float):               Convergence tolerance for the EM algorithm.
        r2_array (np.ndarray):           Array of R-squared values across iterations if tracking is enabled.
    
    Methods:
        __init__(self, n_components):
            Initializes the PPCA model with the specified number of components.
        fit(self, X, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, 
            max_iter=1000, r2_stop=False, r2_iters=25, rng=None):
            Fits the PPCA model to the given data using the EM algorithm.
            Parameters:
                X (np.ndarray):            Data matrix of shape (T, d).
                standardize (bool):        Whether to standardize the data before fitting.
                V_prior (np.ndarray):      Prior covariance matrix for factors.
                track_r2 (bool):           Track R-squared values across iterations.
                tolerance (float):         Convergence tolerance for the EM algorithm.
                max_iter (int):            Maximum number of iterations for the EM algorithm.
                r2_stop (bool):            Whether to stop based on R-squared convergence.
                r2_iters (int):            Number of iterations to consider for R-squared convergence.
                rng (np.random.Generator): Random number generator for reproducibility.
        fitted(self, compute_variance=False):
            Computes the fitted (reconstructed) values and optionally the prediction variance.
            Parameters:
                compute_variance (bool):   Whether to compute the reconstruction variance.
            Returns:
                np.ndarray:                Reconstructed data values.
                np.ndarray (optional):     Reconstruction variance matrix if compute_variance is True.
        predict(self, X, standardize=True, compute_variance=False):
            Reconstructs data using the fitted PPCA model (for out-of-sample data).
            Parameters:
                X (np.ndarray):            Data matrix of shape (T, d) to reconstruct.
                standardize (bool):        Whether to standardize the data before reconstructing.
                compute_variance (bool):   Whether to compute the reconstruction variance.
            Returns:
                np.ndarray:                Reconstructed data values.
                np.ndarray (optional):     Reconstruction variance matrix if compute_variance is True.
    """
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
                
        # Pre-allocate memory for estimates
        self.L = None
        self.sigma2 = None
        self.factors = None

    def fit(self, X, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = False, r2_iters = 25, rng = None):
        # Obtain sizes
        # X is T x d; Factors assumed as T x k
        T, d = X.shape
        k = self.n_components

        # Main error check on dimensions: small-dimensional problem requires T > d + k and d > k
        if T <= d + k:
            raise ValueError("PPCA requires enough observations and features:\nsample_size > n_predictors + n_components and n_predictors > n_components.")
        
        # Fill in components of the class controlling algorithm
        self.max_iter = max_iter
        self.tolerance = tolerance
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else linalg.inv(V_prior)
        
        # Obtain indices of missing observations to create masked objects and initial imputation step
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X_missing_flag = np.any(X_missing_index)
            if X_missing_flag:
                X[X_missing_index] = 0.0
        else:
            X_missing_index = X.mask
            X_missing_flag = True
            X = X.filled(fill_value=0.0)

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
            
        # Initial values for the parameters
        if rng is None:
            rng = np.random.default_rng()
        L0 = rng.normal(size = [d, k])
        sigma2_0 = X.var(axis = 0).mean()    # Mean variance across features
        
        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
        
        # Start EM algorithm main loop
        for it in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            L_scaled = L0 / sigma2_0
            Omega = linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = X @ L_scaled @ Omega

            # If any missing data, update imputation step using current EM fit
            if X_missing_flag:
                X_hat = M @ L0[:d].T
                X[X_missing_index] = X_hat[X_missing_index]

            # Maximization step: Update factor loadings and variances
            V = T * Omega + M.T @ M
            L1 = linalg.solve(V, M.T @ X).T
            sigma2_1 = (1/(T * d)) * (np.sum(X**2) - np.trace(L1.T @ L1 @ V))
            
            # Compute distance between iterates
            L_distance = linalg.norm(L1 - L0, "fro")
            sigma_x_distance = np.abs(sigma2_1 - sigma2_0)
            theta_distance = max([L_distance, sigma_x_distance])

            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                # Save current value of R-squared
                X_hat = M @ L1.T
                r2_values = r2_score(X, X_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and it >= r2_iters:
                # Add stopping condition based on history of R-squared across iterations
                r2_convergence = np.allclose(np.mean(r2_list[-r2_iters:]),    # Average of previous r_iters values
                                             np.mean(r2_list[-1]),            # Current value
                                             self.tolerance)
                convergence = convergence or r2_convergence
            if convergence:
                # Break if either distance between each estimate or R-squared decrease is small
                break
            else:
                # Prepare values for next iteration if convergence not reached
                L0 = L1
                sigma2_0 = sigma2_1
                
        # Update values of the class with results from EM algorithm
        self.L = L1
        self.sigma2 = sigma2_1
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.factors = M
        
    def fitted(self, compute_variance = False):
        # PPCA prediction in-sample:
        X_hat = self.factors @ self.L.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return X_hat
        else:
            Omega_inverse = self.V_prior_inv + self.L.T @ (self.L / self.sigma2)
            X_hat_variance = self.L @ linalg.solve(Omega_inverse, self.L.T)
            return X_hat, X_hat_variance

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations and impute using EM fit
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X_hat = self.factors @ self.L.T
                X[X_missing_index] = X_hat[X_missing_index]
        else:
            X_missing_index = X.mask
            if np.any(X_missing_index):
                X_hat = self.factors @ self.L.T
                X = X.filled(X_hat)            
        
        # Center and scale predictors if required
        h = X.shape[0]
        if standardize and h > 1:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))

        # Obtains predicted factors using X only: F_predicted = M (Posterior mean of factors)
        L_scaled = self.L / self.sigma2
        Omega_X_inverse = self.V_prior_inv + self.L.T @ L_scaled
        F_predicted = linalg.solve(Omega_X_inverse, L_scaled.T @ X.T).T

        # PTFA out-of-sample prediction:
        X_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return X_hat
        else:
            d = self.L.shape[0]
            X_hat_variance = self.sigma2 * np.eye(d) + self.L @ linalg.solve(Omega_X_inverse, self.L.T)
            return X_hat, X_hat_variance