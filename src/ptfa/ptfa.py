import numpy as np
from scipy import linalg, sparse
from sklearn.metrics import r2_score

class ProbabilisticTFA:
    """
    Probabilistic Targeted Factor Analysis (PTFA) class for fitting and predicting using a probabilistic model.
    Attributes:
        n_components (int): Number of components (factors) to estimate.
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
        fit(self, X, Y, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, max_iter=1000, r2_stop=True, r2_iters=100):
            Fits the PTFA model to the given data using the EM algorithm.
            Parameters:
                X (np.ndarray):          Predictor data matrix of shape (T, p).
                Y (np.ndarray):          Target data matrix of shape (T, q).
                standardize (bool):      Whether to standardize the data before fitting.
                V_prior (np.ndarray):    Prior covariance matrix for factors.
                track_r2 (bool):         track R-squared values across iterations.
                tolerance (float):       Convergence tolerance for the EM algorithm.
                max_iter (int):          Maximum number of iterations for the EM algorithm.
                r2_stop (bool):          Whether to stop based on R-squared convergence.
                r2_iters (int):          Number of iterations to consider for R-squared convergence.
        fitted(self, X, Y, standardize=True, compute_variance=False):
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
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.factors = None

    def fit(self, X, Y, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25):
        # Fill in components of the class controlling algorithm
        self.max_iter = max_iter
        self.tolerance = tolerance
        k = self.n_components
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else np.linalg.inv(V_prior)
        
        # Obtain sizes and missing indices
        # X is T x p; Y is T x q; Factors assumed as T x k
        T, p = X.shape
        _, q = Y.shape
        d = p + q
        
        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            if np.any(Y_missing_index):
                Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)
        else:
            Y_missing_index = Y.mask
            Y.set_fill_value(value = 0.0)

        # Center and scale predictors and targets separately
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        
        # If any elements are missing, do an initial imputation step
        Z = np.zeros([T, d])
        Z[:, :p] = X.filled() if np.any(X_missing_index) else X
        Z[:, p:] = Y.filled() if np.any(X_missing_index) else Y
        
        # Initial values for the parameters
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x0 = X.var(axis = 0).mean()    # Mean variance across features
        sigma2_y0 = Y.var(axis = 0).mean()    # Mean variance across targets

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
        
        # Start EM algorithm main loop
        for i in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            Omega = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Omega

            # If any missing data, update imputation step using current EM fit
            if np.any(X_missing_index):
                Z[:, :p] = X.filled(M @ L0[:p].T)
            if np.any(Y_missing_index):
                Z[:, p:] = Y.filled(M @ L0[p:].T)
            
            # Maximization step: Update factor loadings and variances
            V = T * Omega + M.T @ M
            L1 = np.linalg.solve(V, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            sigma2_x1 = (1/(T * p)) * (np.sum(X**2) - np.trace(P.T @ P @ V))
            sigma2_y1 = (1/(T * q)) * (np.sum(Y**2) - np.trace(Q.T @ Q @ V))
            
            # Compute distance between iterates
            P_distance = np.linalg.norm(P - L0[:p], "fro")
            Q_distance = np.linalg.norm(Q - L0[p:], "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                # Save current value of R-squared
                Y_hat = M @ Q.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and i >= r2_iters:
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
        self.P = P
        self.Q = Q
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
            q = self.Q.shape[0]
            Omega_inverse = self.V_prior_inv + self.P.T @ (self.P / self.sigma2_x) + self.Q.T @ (self.Q / self.sigma2_y)
            Y_hat_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse, self.Q.T)
            return Y_hat, Y_hat_variance

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations to create masked object
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)

        # Center and scale predictors and impute using EM fit if required
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        if np.any(X_missing_index):
            X = X.filled(self.factors @ self.P.T)
        
        # Obtains predicted factors using X only: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x
        Omega_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        F_predicted = np.linalg.solve(Omega_X_inverse, P_scaled.T @ X.T).T

        # PTFA out-of-sample prediction:
        Y_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            Y_hat_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_X_inverse, self.Q.T)
            return Y_hat, Y_hat_variance

class ProbabilisticTFA_MixedFrequency:
    """
    A class to perform Probabilistic Targeted Factor Analysis (TFA) with mixed frequency data.
    Attributes
    ----------
    n_components : int
        Number of latent factors.
    P : np.ndarray
        Factor loadings for high-frequency predictors.
    Q : np.ndarray
        Factor loadings for low-frequency targets.
    sigma2_x : float
        Variance of high-frequency predictors.
    sigma2_y : float
        Variance of low-frequency targets.
    factors : np.ndarray
        Estimated factors.
    V_prior : np.ndarray
        Prior covariance matrix for factors.
    max_iter : int
        Maximum number of iterations for the EM algorithm.
    tolerance : float
        Tolerance for convergence of the EM algorithm.
    r2_array : np.ndarray
        Array of R-squared values across iterations.
    Methods
    -------
    highfrequency_to_lowfrequency_reshape(X, low_frequency_T, periods):
        Reshape high-frequency data to low-frequency data.
    fit(X, Y, periods, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, max_iter=1000, r2_stop=True, r2_iters=100):
        Fit the model using the EM algorithm.
    fitted(X, Y, periods, standardize=True):
        Obtain fitted values for the low-frequency targets.
    predict(X, low_frequency_T, periods, standardize=True):
        Predict low-frequency targets using the fitted model.
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
        self.low_frequency_T = None

    def highfrequency_to_lowfrequency_reshape(self, X, low_frequency_T, periods):
        # Obtain available periods and any remainder needed to be filled out
        high_frequency_T, p = X.shape
        last_T = periods * (low_frequency_T - 1)
        remainder_T = high_frequency_T - last_T

        # Pre-allocate reshaped object and fill out available information
        reshaped_X = np.zeros([low_frequency_T, p * periods])
        for t in range(low_frequency_T - 1):
            row_index = range(t * periods, (t+1) * periods)
            reshaped_X[t] = np.ravel(X[row_index])

        # Fill out information corresponding to last entry (needed if remainder_T > 0)
        row_index = range(last_T, last_T + remainder_T)
        reshaped_X[low_frequency_T - 1, :(p * remainder_T)] = np.ravel(X[row_index])
        reshaped_X[low_frequency_T - 1, (p * remainder_T):] = np.nan
        return reshaped_X

    def fit(self, X, Y, periods, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25):
        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.periods = periods
        k = self.n_components
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else np.linalg.inv(V_prior)
        
        # Obtain sizes and missing indices
        # X is assumed inputed as high_frequency_T x p; Y is low_frequency_T x q
        # Future: Implement periods that changes for each low frequency interval
        low_frequency_T, q = Y.shape
        _, p = X.shape
        self.low_frequency_T = low_frequency_T
        
        # Obtain indices of missing observations to create masked object
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
        else:
            X_missing_index = X.mask
            X = X.data
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            if np.any(Y_missing_index):
                Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)
        else:
            Y_missing_index = Y.mask
            Y.set_fill_value(value = 0.0)   

        # Center and scale predictors and targets separately
        # Also initial imputation step
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        if np.any(Y_missing_index):
            Y = Y.filled(0.0)

        # Re-shape X into an low_frequency_T x (p * periods)
        # Also initial imputation step
        reshaped_X = self.highfrequency_to_lowfrequency_reshape(X, low_frequency_T, periods)
        X_missing_index = np.isnan(reshaped_X)
        reshaped_X[X_missing_index] = 0.0            
        
        # Initial values for the parameters
        P0 = np.random.default_rng().normal(size = [p, k])
        Q0 = np.random.default_rng().normal(size = [q, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []

        # Start EM algorithm main loop
        J_periods = np.full([periods, periods], 1 / periods)
        for i in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            P_scaled = P0 / sigma2_x0
            Q_scaled = Q0 / sigma2_y0
            Omega_P_term = np.kron(np.eye(periods), self.V_prior_inv + P0.T @ P_scaled)
            Omega_Q_term = np.kron(J_periods, Q0.T @ Q_scaled)
            Omega = np.linalg.inv(Omega_P_term + Omega_Q_term)
            ZL_matrix = np.zeros([low_frequency_T, periods * k])
            Y_times_Q = Y @ Q_scaled
            for j in range(periods):
                ZL_matrix[:, range(j * k, (j+1) * k)] = reshaped_X[:, range(j * p, (j+1) * p)] @ P_scaled + Y_times_Q
            M = ZL_matrix @ Omega
            
            # Update low-frequency prediction and additional necessary quantities
            M_sum = 0
            XM_sum = 0
            for j in range(periods):
                M_sum += M[:, range(j * k, (j+1) * k)]
                XM_sum += reshaped_X[:, range(j * p, (j+1) * p)].T @ M[:, range(j * k, (j+1) * k)]

            # Update missing data using current EM fitted values
            if np.any(X_missing_index):
                reshaped_X_hat = np.hstack([M[:, range(j * k, (j+1) * k)] @ P0.T for j in range(periods)])
                reshaped_X[X_missing_index] = reshaped_X_hat[X_missing_index]
            if np.any(Y_missing_index):
                Y_hat = (1/periods) * M_sum @ Q0.T
                Y = Y.filled(Y_hat)
            
            # Maximization step: Update factor loadings and variances
            V_array = np.reshape(low_frequency_T * Omega + M.T @ M, [periods, k, periods, k]).transpose(0, 2, 1, 3)
            V_diagsum = np.einsum('iijk->jk', V_array)
            V_allsum = np.einsum('ijkl->kl', V_array)
            P1 = np.linalg.solve(V_diagsum, XM_sum.T).T
            Q1 = periods * np.linalg.solve(V_allsum, M_sum.T @ Y).T
            Y_hat = (1/periods) * M_sum @ Q1.T
            sigma2_x1 = (1/(low_frequency_T * p * periods)) * (np.sum(X**2) - np.trace(P1.T @ P1 @ V_diagsum))
            sigma2_y1 = (periods/(low_frequency_T * q)) * np.sum(Y * (Y - Y_hat))
            
            # Compute distance between iterates
            P_distance = np.linalg.norm(P1 - P0, "fro")
            Q_distance = np.linalg.norm(Q1 - Q0, "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and i >= r2_iters:
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
        self.factors = M

    def fitted(self, compute_variance = False):
        # PTFA prediction in-sample:
        k = self.n_components
        F_sum = sum([self.factors[:, range(j * k, (j+1) * k)] for j in range(self.periods)])
        Y_hat = (1/self.periods) * F_sum @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            Omega_inverse = self.V_prior_inv + self.P.T @ (self.P / self.sigma2_x) + self.Q.T @ (self.Q / self.sigma2_y)
            Q_Omega_Q = ( self.Q @ np.kron(np.ones([1, self.periods]), np.eye(k)) @
                            np.linalg.solve(Omega_inverse, np.kron(np.ones(self.periods), np.eye(k)) @ self.Q.T) )
            Y_hat_variance = (1/self.periods**2) * Q_Omega_Q + (1/self.periods) * self.sigma2_y * np.eye(q)
            return Y_hat, Y_hat_variance
    
    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain necessary sizes
        _, p = X.shape
        k = self.n_components

        # Obtain indices of missing observations to create masked object
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)
        
        # Center and scale predictors
        if standardize:
            X = (X - np.mean(X, axis = 0, where = np.logical_not(X_missing_index))) / np.std(X, axis = 0, where = np.logical_not(X_missing_index))

        # Re-shape X into an low_frequency_T x (p * periods), padding if necessary
        reshaped_X = self.highfrequency_to_lowfrequency_reshape(X, self.low_frequency_T, self.periods)
        
        # Imputation step using EM fit if required
        X_missing_index = np.isnan(reshaped_X)
        if np.any(X_missing_index):
            reshaped_X_hat = np.hstack([self.factors[:, range(j * k, (j+1) * k)] @ self.P.T for j in range(self.periods)])
            reshaped_X = reshaped_X.filled(reshaped_X_hat)
            
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x
        Omega_P_term = np.kron(np.eye(self.periods), self.V_prior_inv + self.P.T @ P_scaled)
        XP_matrix = np.zeros([self.low_frequency_T, self.periods * k])
        for j in range(self.periods):
            XP_matrix[:, range(j * k, (j+1) * k)] = reshaped_X[:, range(j * p, (j+1) * p)] @ P_scaled
        F_predicted = np.linalg.solve(Omega_P_term, XP_matrix.T).T
        
        # Predict using estimated factors
        F_sum = sum([F_predicted[:, range(j * k, (j+1) * k)] for j in range(self.periods)])
        Y_hat = (1/self.periods) * F_sum @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            Q_Omega_Q = ( self.Q @ np.kron(np.ones([1, self.periods]), np.eye(k)) @
                            np.linalg.solve(Omega_P_term, np.kron(np.ones(self.periods), np.eye(k)) @ self.Q.T) )
            Y_hat_variance = (1/self.periods**2) * Q_Omega_Q + (1/self.periods) * self.sigma2_y * np.eye(q)
            return Y_hat, Y_hat_variance

class ProbabilisticTFA_StochasticVolatility:
    """
    A class to perform Probabilistic Targeted Factor Analysis (PTFA) with Stochastic Volatility.
    Attributes:
        n_components (int):    Number of components (factors).
        P (np.ndarray):        Factor loadings for predictors.
        Q (np.ndarray):        Factor loadings for targets.
        sigma2_x (np.ndarray): Time-varying volatilities for predictors equations.
        sigma2_y (np.ndarray): Time-varying volatilities for targets equations.
        factors (np.ndarray):  Predicted factors.
        max_iter (int):        Maximum number of iterations for the EM algorithm.
        tolerance (float):     Convergence tolerance for the EM algorithm.
        ewma_lambda_x (float): EWMA smoothing parameter for predictors.
        ewma_lambda_y (float): EWMA smoothing parameter for targets.
        V_prior (np.ndarray):  Prior covariance matrix for factors.
        r2_array (np.ndarray): Array of R-squared values across iterations.
    Methods:
        __init__(self, n_components):
            Initializes the class with the specified number of components.
        fit(self, X, Y, standardize=True, ewma_lambda_x=0.94, ewma_lambda_y=None, V_prior=None, track_r2=True, tolerance=1e-6, max_iter=1000, r2_stop=True, r2_iters=100):
            Fits the model to the data using the EM algorithm.
        fitted(self, X, Y, standardize=True):
            Returns the predicted targets using the fitted model.
        predict(self, X, standardize=True):
            Predicts the targets using the predictors and the fitted model.
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

    def fit(self, X, Y, standardize = True, ewma_lambda_x = 0.94, ewma_lambda_y = None, V_prior = None,
            track_r2 = True, tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25):
        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance          # EM stopping tolerance
        k = self.n_components
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else np.linalg.inv(V_prior)

        # Specific for stochastic volatility: EWMA smoothing parameter for feature process and targets
        self.ewma_lambda_x = ewma_lambda_x 
        self.ewma_lambda_y = ewma_lambda_y if ewma_lambda_y is not None else ewma_lambda_x
        
        # Obtain sizes
        # X is T x p; Y is T x q; Factors assumed as T x k
        T, p = X.shape
        _, q = Y.shape
        d = p + q

        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            if np.any(Y_missing_index):
                Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)
        else:
            Y_missing_index = Y.mask
            Y.set_fill_value(value = 0.0)

        # Center and scale predictors and targets separately
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        
        # If any elements are missing, do an initial imputation step
        Z = np.zeros([T, d])
        Z[:, :p] = X.filled() if np.any(X_missing_index) else X
        Z[:, p:] = Y.filled() if np.any(X_missing_index) else Y

        # Initial values for the parameters (time-varying volatilities start constant)
        L0 = np.random.default_rng().normal(size = [d, k])
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
        for i in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            for t in range(T):
                L_scaled_t = np.vstack([L0[:p] / sigma2_x[t], L0[p:] / sigma2_y[t]])
                Omega[t] = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled_t)
                M[t] = Z[t] @ L_scaled_t @ Omega[t]
            V = np.sum(Omega, axis = 0) + M.T @ M

            # If any missing data, update imputation step using current EM fit
            if np.any(X_missing_index):
                Z[:, :p] = X.filled(M @ L0[:p].T)
            if np.any(Y_missing_index):
                Z[:, p:] = Y.filled(M @ L0[p:].T)

            # Maximization step: Update factor loadings
            L1 = np.linalg.solve(V, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            
            # Update volatilities using EWMA
            Z_hat = M @ L1.T
            residuals_Z = Z - Z_hat
            sigma2_x[0] = (1/p) * (np.sum(residuals_Z[0, :p]**2) + np.trace(P.T @ P @ Omega[0]))
            sigma2_y[0] = (1/q) * (np.sum(residuals_Z[0, p:]**2) + np.trace(Q.T @ Q @ Omega[0]))
            for t in range(1, T):
                hat_sigma2_x_t = (1/p) * (np.sum(residuals_Z[t, :p]**2) + np.trace(P.T @ P @ Omega[t]))
                hat_sigma2_y_t = (1/q) * (np.sum(residuals_Z[t, p:]**2) + np.trace(Q.T @ Q @ Omega[t]))
                sigma2_x[t] = self.ewma_lambda_x * sigma2_x[t-1] + (1 - self.ewma_lambda_x) * hat_sigma2_x_t
                sigma2_y[t] = self.ewma_lambda_y * sigma2_y[t-1] + (1 - self.ewma_lambda_y) * hat_sigma2_y_t
            
            # Compute distance between iterates
            P_distance = np.linalg.norm(P - L0[:p], "fro")
            Q_distance = np.linalg.norm(Q - L0[p:], "fro")
            theta_distance = sum([P_distance, Q_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2 or r2_stop:
                r2_values = r2_score(Y, Z_hat[:, p:], multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and i >= r2_iters:
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
        self.P = P
        self.Q = Q
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.factors = M
        
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
                Omega_inverse_t = self.V_prior_inv + self.P.T @ (self.P / self.sigma2_x[t]) + self.Q.T @ (self.Q / self.sigma2_y[t])
                Y_hat_variance[t] = self.sigma2_y[t] * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse_t, self.Q.T)
            return Y_hat, Y_hat_variance

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)
        
        # Center and scale predictors and impute using EM fit if required
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        if np.any(X_missing_index):
            X = X.filled(self.factors @ self.P.T)
        
        # Obtains predicted factors using X only: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x.mean()
        Omega_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        F_predicted = np.linalg.solve(Omega_X_inverse, P_scaled.T @ X.T).T

        # PTFA out-of-sample prediction:
        Y_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            Y_hat_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_X_inverse, self.Q.T)
            return Y_hat, Y_hat_variance

class ProbabilisticTFA_DynamicFactors:
    """    
    Probabilistic Targeted Factor Analysis (PTFA) with Dynamic Factors
    This class implements a PTFA model with dynamic factors anduses an Expectation-Maximization (EM) algorithm to fit the model parameters.
    Attributes:
        n_components (int): Number of components (factors) in the model.
        P (np.ndarray):           Factor loadings for predictors.
        Q (np.ndarray):           Factor loadings for targets.
        sigma2_x (float):         Variance of the predictors.
        sigma2_y (float):         Variance of the targets.
        A (np.ndarray):           Autoregressive coefficients.
        f0 (np.ndarray):          Initial condition for the factors.
        factors (np.ndarray):     Predicted factors.
        V_prior (np.ndarray):     Prior covariance matrix for the factors.
        V_prior_inv (np.ndarray): Inverse of the prior covariance matrix.
        max_iter (int):           Maximum number of iterations for the EM algorithm.
        tolerance (float):        Convergence tolerance for the EM algorithm.
        r2_array (np.ndarray):    Array of R-squared values across iterations.
    Methods:
        __init__(self, n_components):
            Initializes the ProbabilisticTFA_DynamicFactors class with the specified number of components.
        bands_cholesky(self, cholesky_banded, desired_bands=0):
            Computes the inverse elements of a banded matrix using its Cholesky decomposition.
        fit(self, X, Y, standardize=True, V_prior=None, track_r2=True, tolerance=1e-6, max_iter=1000, r2_stop=True, r2_iters=100):
            Fits the model to the given data using the EM algorithm.
        fitted(self, X, Y, standardize=True, prediction_variance=False):
            Computes the fitted values for the given data.
        predict(self, X, standardize=True, prediction_variance=False, method="mean"):
            Predicts the target values for the given predictors.
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
                Element (i, j) of Omega is located in [maximum_bands + i - j, t*k + r] for l, r in {0, ..., k-1} 
        """
        # Pre-allocate final objective using both the upper and lower bands for now
        half_bandwidth, Tk = cholesky_banded.shape
        total_rows = 2 * desired_bands + 1
        Omega = np.zeros([total_rows, Tk])
        
        # Transform Cholesky decomposition to LDL' decomposition
        cholesky_diagonal = cholesky_banded[0]                 # Given banded structure, diagonal is first row
        cholesky_banded = cholesky_banded / cholesky_diagonal  # Columns are also in place, so can simply divide to invert
        cholesky_diagonal = 1 / cholesky_diagonal**2
        
        # Main algorithm loop
        bandwidth_range = range(1, half_bandwidth)
        for j in reversed(range(Tk)):
            for i in reversed(range(max(j - desired_bands - 1, 0), j)):
                save_row_index = desired_bands + i - j
                next_row_index = range(min(save_row_index + 1, total_rows - 1),
                                       min(save_row_index + half_bandwidth, total_rows - 1))
                Omega[save_row_index, j] = -np.dot(cholesky_banded[bandwidth_range, i], Omega[next_row_index, j])
                Omega[min(desired_bands + j - i, total_rows - 1), i] = Omega[save_row_index, j]
                if i == j:
                    Omega[save_row_index, i] += cholesky_diagonal[i]
        
        # Discard upper set of elements and return
        return Omega[desired_bands:, :]

    def fit(self, X, Y, standardize = True, V_prior = None, track_r2 = True,
            tolerance = 1e-6, max_iter = 1000, r2_stop = True, r2_iters = 25):
        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance
        k = self.n_components
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else np.linalg.inv(V_prior)
        
        # Obtain sizes
        # X is T x p; Y is T x q; Factors assumed as T x k
        T, p = X.shape
        _, q = Y.shape
        d = p + q

        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            if np.any(Y_missing_index):
                Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)
        else:
            Y_missing_index = Y.mask
            Y.set_fill_value(value = 0.0)

        # Center and scale predictors and targets separately
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        
        # If any elements are missing, do an initial imputation step
        Z = np.zeros([T, d])
        Z[:, :p] = X.filled() if np.any(X_missing_index) else X
        Z[:, p:] = Y.filled() if np.any(X_missing_index) else Y

        # Initial values for the parameters
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets
        A0 = np.eye(k)
        f0_0 = np.zeros(k)

        # Track R-squared of fit if necessary
        if track_r2 or r2_stop:
            r2_list = []
        
        # Start EM algorithm main loop
        for i in range(self.max_iter):
            ### Expectation step: Update posterior paramater for factors using sparse matrix computations ---
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            L_scaled_L = L0.T @ L_scaled
            Sigma_v_A = - self.V_prior_inv @ A0
            A_Sigma_v_A = - A0.T @ Sigma_v_A
            Omega_0_inv = self.V_prior_inv + A_Sigma_v_A + L_scaled_L
            
            # Save posterior precision in sparse representation as intermediate
            main_diagonal = sparse.kron(sparse.eye(T, format='csc'), Omega_0_inv)
            lower_diagonal = sparse.kron(sparse.eye(T, k=-1, format='csc'), Sigma_v_A)
            last_block = sparse.lil_matrix((T * k, T * k))  # Create an empty sparse matrix of the same size
            last_block[(T - 1) * k : T * k, (T - 1) * k : T * k] = self.V_prior_inv + L_scaled_L
            Omega_inv_sparse = sparse.tril(main_diagonal + lower_diagonal + sparse.csc_array(last_block))

            # Save posterior precision to a symmetric banded matrix and compute its Cholesky decomposition
            # (Tk x Tk) -> (2k x Tk), only storing the 2k lower diagonal bands
            Omega_inv_cholesky = np.zeros([2 * k, T * k])
            for diagonal in range(2 * k):
                Omega_inv_cholesky[diagonal, :(T * k - diagonal)] = Omega_inv_sparse.diagonal(-diagonal)
            Omega_inv_cholesky = linalg.cholesky_banded(Omega_inv_cholesky, overwrite_ab=True, lower=True)
                
            # Compute posterior mean using banded matrix solver
            M = np.ravel(Z @ L_scaled)
            M[:k] = M[:k] - Sigma_v_A @ f0_0
            M = linalg.cho_solve_banded((Omega_inv_cholesky, True), b = M, overwrite_b=True).reshape([T, k])
            
            # If any missing data, update imputation step using current EM fit
            if np.any(X_missing_index):
                Z[:, :p] = X.filled(M @ L0[:p].T)
            if np.any(Y_missing_index):
                Z[:, p:] = Y.filled(M @ L0[p:].T)

            ### Maximization step: Update factor loadings and variances ---
            # Calculate banded elements of the posterior covariance using lower-level function
            Omega_banded = self.bands_cholesky(Omega_inv_cholesky, 3 * k - 1)

            # Compute sums over block diagonals of the posterior covariance
            required_blocks = 3    # Diagonal block + two lower diagonal blocks
            sum_array = np.zeros([required_blocks, k, k])
            for block in range(required_blocks):
                for j in range(k):
                    if block == 0:
                        # Compute lower-diagonal block of V_0 = sum_{t=1}^{T} Omega_{t, t}
                        sum_array[0][j:, j] = np.sum(Omega_banded[np.ix_(range(k - j), range(j, T * k, k))], axis = 1)
                    else:
                        # Compute Bar_Omega_j = sum_{t=j}^{T-1} Omega_{t - j, t}
                        row_index = range(block * k - j, (block+1) * k - j)
                        column_index = range(j, (T - block) * k, k)
                        sum_array[block][:, j] = np.sum(Omega_banded[np.ix_(row_index, column_index)], axis = 1)
            sum_array[0][np.triu_indices(k, 1)] = sum_array[0][np.tril_indices(k, -1)]  # Fill-in missing block

            # Update loadings and error variances
            V_0 = sum_array[0] + M.T @ M
            L1 = np.linalg.solve(V_0, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            sigma2_x1 = (1/(T * p)) * (np.sum(X**2) - np.trace(P.T @ P @ V_0))
            sigma2_y1 = (1/(T * q)) * (np.sum(Y**2) - np.trace(Q.T @ Q @ V_0))
            
            # Update dynamic parameters: Autoregressive coefficients and initial condition
            V_1 = sum_array[1] + M[:(T-1)].T @ M[1:]
            V_2 = sum_array[2] + M[:(T-2)].T @ M[2:]
            A1 = np.linalg.solve(V_2, V_1)
            f0_1 = np.linalg.solve(A1.T @ self.V_prior @ A1, A1.T @ self.V_prior @ M[0])

            # Compute distance between iterates
            P_distance = np.linalg.norm(P - L0[:p], "fro")
            Q_distance = np.linalg.norm(Q - L0[p:], "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            A_distance = np.linalg.norm(A1 - A0, "fro")
            f0_distance = np.linalg.norm(f0_0 - f0_1, 2)
            theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance, A_distance, f0_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2:
                Y_hat = M @ Q.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            convergence = (theta_distance <= self.tolerance)
            if r2_stop and i >= r2_iters:
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
        self.P = P
        self.Q = Q
        self.sigma2_x = sigma2_x1
        self.sigma2_y = sigma2_y1
        self.A = A1
        self.f0 = f0_1
        self.r2_array = np.asarray(r2_list) if track_r2 else None
        self.factors = M
        
    def fitted(self, compute_variance = False):
        # PTFA prediction in-sample:
        Y_hat = self.factors @ self.Q.T
        
        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            Omega_inverse = self.V_prior_inv + self.P.T @ (self.P / self.sigma2_x) + self.Q.T @ (self.Q / self.sigma2_y)
            Y_hat_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse, self.Q.T)
            return Y_hat, Y_hat_variance

    def predict(self, X, standardize = True, compute_variance = False):
        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            if np.any(X_missing_index):
                X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        else:
            X_missing_index = X.mask
            X.set_fill_value(value = 0.0)
        
        # Center and scale predictors and impute using EM fit if required
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        if np.any(X_missing_index):
            X = X.filled(self.factors @ self.P.T)
        
        # Compute some necessary quantities using dynamic information estimates
        T = X.shape[0]
        k = self.n_components
        P_scaled_P = (self.P.T @ self.P) / self.sigma2_x
        Sigma_v_A = - self.V_prior_inv @ self.A
        A_Sigma_v_A = - self.A.T @ Sigma_v_A
        Omega_0_inv_X = self.V_prior_inv + A_Sigma_v_A + P_scaled_P
            
        # Save posterior precision in sparse representation as intermediate
        main_diagonal = sparse.kron(sparse.eye(T, format='csc'), Omega_0_inv_X)
        lower_diagonal = sparse.kron(sparse.eye(T, k=-1, format='csc'), Sigma_v_A)
        last_block = sparse.lil_matrix((T * k, T * k))  # Create an empty sparse matrix of the same size
        last_block[(T - 1) * k : T * k, (T - 1) * k : T * k] = self.V_prior_inv + P_scaled_P
        Omega_inv_sparse = sparse.tril(main_diagonal + lower_diagonal + sparse.csc_array(last_block))

        # Save posterior precision to a symmetric banded matrix and compute its Cholesky decomposition
        # (Tk x Tk) -> (2k x Tk), only storing the 2k lower diagonal bands
        Omega_inv_cholesky = np.zeros([2 * k, T * k])
        for diagonal in range(2 * k):
            Omega_inv_cholesky[diagonal, :(T * k - diagonal)] = Omega_inv_sparse.diagonal(-diagonal)
        Omega_inv_cholesky = linalg.cholesky_banded(Omega_inv_cholesky, overwrite_ab=True, lower=True)
                
        # Compute predicted factors using banded matrix solver
        F_predicted = np.ravel(X @ self.P / self.sigma2_x)
        F_predicted[:k] = F_predicted[:k] - Sigma_v_A @ self.f0
        F_predicted = linalg.cho_solve_banded((Omega_inv_cholesky, True), b = F_predicted, overwrite_b=True).reshape([T, k])

        # PTFA out-of-sample prediction:
        Y_hat = F_predicted @ self.Q.T

        # Return value depends on whether fitted variance is also required
        if not compute_variance:
            return Y_hat
        else:
            # Approximated PTFA out-of-sample prediction variance
            q = self.Q.shape[0]
            Y_hat_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_0_inv_X, self.Q.T)
            return Y_hat, Y_hat_variance

                