import numpy as np
from scipy import linalg, sparse
from sklearn.metrics import r2_score
import warnings

class ProbabilisticTFA:
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
                
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.factors = None

    def fit(self, X, Y, standardize = False, track_r2 = False, tolerance = 1e-6, max_iter = 1000, V_prior = None):
        # Fill in components of the class controlling algorithm
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

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])

        # Initial values for the parameters
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

        # Track R-squared of fit if necessary
        if track_r2:
            r2_list = []
        
        # Start EM algorithm main loop
        for _ in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            Omega = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Omega
            
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
            if track_r2:
                Y_hat = M @ Q.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            if (theta_distance <= self.tolerance):
                # Break if distance between each estimate is less than a tolerance
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
        
    def fitted(self, X, Y, standardize = False, prediction_variance = False):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P, self.Q])
        L_scaled = np.vstack([self.P / self.sigma2_x, self.Q / self.sigma2_y])
        Omega_inverse = self.V_prior_inv + L.T @ L_scaled
        F_predicted = np.linalg.solve(Omega_inverse, L_scaled.T @ Z.T).T

        # Predict using estimated factors and update class
        self.factors = F_predicted
        Y_hat = F_predicted @ self.Q.T

        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            fitted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse, self.Q.T)
            return Y_hat, fitted_variance

    def predict(self, X, standardize = False, prediction_variance = False, method = "mean"):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        
        # Explore differences in prediction methods
        P_scaled = self.P / self.sigma2_x
        Omega_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        if method == "mean":
            # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
            F_predicted = np.linalg.solve(Omega_X_inverse, P_scaled.T @ X.T).T

            # Predict using estimated factors
            Y_hat = F_predicted @ self.Q.T
        elif method == "loading":
            # Predict using estimated loadings directly
            p = self.P.shape[0]
            C_X = self.sigma2_x * np.eye(p) + self.P @ self.V_prior @ self.P.T
            Y_hat = X @ np.linalg.solve(C_X, self.P @ self.V_prior @ self.Q.T)
        
        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            predicted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_X_inverse, self.Q.T)
            return Y_hat, predicted_variance

class ProbabilisticTFA_Missing:
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.Z_hat_ = None
        self.factors = None

    def fit(self, X, Y, standardize = False, track_r2 = False, tolerance = 1e-6, max_iter = 1000, V_prior = None):
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
            X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)

        # Center and scale predictors and targets separately if necessary
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)

        # Initial imputation step before stacking
        Z = np.hstack([X.filled(0.0), Y.filled(0.0)])
        Z_vars = np.var(Z, axis = 0)

        # Initial values for the parameters
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x0 = np.mean(Z_vars[:p])    # Mean variance across features
        sigma2_y0 = np.mean(Z_vars[p:])    # Mean variance across targets     

        # Track R-squared of fit if necessary
        if track_r2:
            r2_list = []
        
        # Start EM algorithm main loop
        for _ in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            Omega = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Omega
            
            # Update missing data using current EM fitted values
            Z_hat = M @ L0.T
            Z = np.hstack([X.filled(Z_hat[:, :p]), Y.filled(Z_hat[:, p:])])

            # Maximization step: Update factor loadings and variances
            V = T * Omega + M.T @ M
            L1 = np.linalg.solve(V, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            sigma2_x1 = (1/(T * p)) * (np.sum(Z[:, :p]**2) - np.trace(P.T @ P @ V))
            sigma2_y1 = (1/(T * q)) * (np.sum(Z[:, p:]**2) - np.trace(Q.T @ Q @ V))
            
            # Compute distance between iterates
            P_distance = np.linalg.norm(P - L0[:p], "fro")
            Q_distance = np.linalg.norm(Q - L0[p:], "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2:
                Y_hat = M @ Q.T
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            if (theta_distance <= self.tolerance):
                # Break if distance between each estimate is less than a tolerance
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
        self.Z_imputed = Z

    def fitted(self, X, Y, standardize = False, prediction_variance = False):
        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)

        # Center and scale predictors and targets separately if necessary
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P, self.Q])
        L_scaled = np.vstack([self.P / self.sigma2_x, self.Q / self.sigma2_y])
        Omega_inverse = self.V_prior_inv + L.T @ L_scaled
        F_predicted = np.linalg.solve(Omega_inverse, L_scaled.T @ self.Z_imputed.T).T
        # Z_imputed is saved from fit

        # Predict using estimated factors and update class
        self.factors = F_predicted
        Y_hat = F_predicted @ self.Q.T

        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            fitted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse, self.Q.T)
            return Y_hat, fitted_variance
    
    def predict(self, X, standardize = False, prediction_variance = False, method = "mean"):
        # Obtain indices of missing observations to create masked objects
        if not np.ma.isMaskedArray(X):
            X_missing_index = np.isnan(X)
            X = np.ma.MaskedArray(data=X, mask=X_missing_index, fill_value=0.0)
        if not np.ma.isMaskedArray(Y):
            Y_missing_index = np.isnan(Y)
            Y = np.ma.MaskedArray(data=Y, mask=Y_missing_index, fill_value=0.0)

        # Center and scale predictors and targets separately if necessary
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        
        # Fill missing values with prediction from model
        p = X.shape[1]
        X = X.filled(self.Z_imputed[:, :p])

        # Explore differences in prediction methods
        P_scaled = self.P / self.sigma2_x
        Omega_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        if method == "mean":
            # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
            F_predicted = np.linalg.solve(Omega_X_inverse, P_scaled.T @ X.T).T

            # Predict using estimated factors
            Y_hat = F_predicted @ self.Q.T
        elif method == "loading":
            # Predict using estimated loadings directly
            C_X = self.sigma2_x * np.eye(p) + self.P @ self.V_prior @ self.P.T
            Y_hat = X @ np.linalg.solve(C_X, self.P @ self.V_prior @ self.Q.T)
        
        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            predict_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_X_inverse, self.Q.T)
            return Y_hat, predict_variance

class ProbabilisticTFA_MixedFrequency:
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.factors = None

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
        return reshaped_X, remainder_T

    def fit(self, X, Y, periods, standardize = False, track_r2 = False, tolerance = 1e-6, max_iter = 1000, V_prior = None):
        # Fill in components of the class
        self.max_iter = max_iter
        self.tolerance = tolerance
        k = self.n_components
        if V_prior is None:
            self.V_prior = np.eye(k)
        self.V_prior_inv = np.eye(k) if V_prior is None else np.linalg.inv(V_prior)
        
        # Obtain sizes
        # X is assumed inputed as high_frequency_T x p; Y is low_frequency_T x q
        # Future: Implement periods that changes for each low frequency interval
        low_frequency_T, q = Y.shape
        _, p = X.shape

        # Center and scale predictors and targets separately
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)

        # Re-shape X into an low_frequency_T x (p * periods)
        reshaped_X, remainder_T = self.highfrequency_to_lowfrequency_reshape(X, low_frequency_T, periods)
        
        # Initial values for the parameters
        P0 = np.random.default_rng().normal(size = [p, k])
        Q0 = np.random.default_rng().normal(size = [q, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

        # Track R-squared of fit if necessary
        if track_r2:
            r2_list = []

        # Start EM algorithm main loop
        J_periods = np.full([periods, periods], 1 / periods)
        for _ in range(self.max_iter):
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
            
            # Update missing high-frequency features using current EM fitted values
            X_hat_missing = M[low_frequency_T - 1, (k * remainder_T):] @ np.kron(np.eye(periods - remainder_T), P0.T)
            reshaped_X[low_frequency_T - 1, (p * remainder_T):] = X_hat_missing
            
            # Maximization step: Update factor loadings and variances
            V_array = np.reshape(low_frequency_T * Omega + M.T @ M, [periods, k, periods, k]).transpose(0, 2, 1, 3)
            V_diagsum = np.einsum('iijk->jk', V_array)
            V_allsum = np.einsum('ijkl->kl', V_array)
            M_sum = 0
            XM_sum = 0
            for j in range(periods):
                M_sum += M[:, range(j * k, (j+1) * k)]
                XM_sum += reshaped_X[:, range(j * p, (j+1) * p)].T @ M[:, range(j * k, (j+1) * k)]
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
            if track_r2:
                r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            if (theta_distance <= self.tolerance):
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

    def fitted(self, X, Y, periods, standardize = False):
        # Obtain necessary sizes
        low_frequency_T, _ = Y.shape
        _, p = X.shape
        k = self.n_components
        
        # Center and scale predictors and targets separately
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)

        # Re-shape X into an low_frequency_T x (p * periods), padding if necessary
        reshaped_X, _ = self.highfrequency_to_lowfrequency_reshape(X, low_frequency_T, periods)

        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x
        Q_scaled = self.Q / self.sigma2_y
        J_periods = np.full([periods, periods], 1 / periods)
        Omega_P_term = np.kron(np.eye(periods), self.V_prior_inv + self.P.T @ P_scaled)
        Omega_Q_term = np.kron(J_periods, self.Q.T @ Q_scaled)
        ZL_matrix = np.zeros([low_frequency_T, periods * k])
        Y_times_Q = Y @ Q_scaled
        for j in range(periods):
            ZL_matrix[:, range(j * k, (j+1) * k)] = reshaped_X[:, range(j * p, (j+1) * p)] @ P_scaled + Y_times_Q
        F_predicted = np.linalg.solve(Omega_P_term + Omega_Q_term, ZL_matrix.T).T
        
        # Predict using estimated factors and update class
        self.factors = F_predicted
        F_sum = sum([F_predicted[:, range(j * k, (j+1) * k)] for j in range(periods)])
        Y_hat = (1/periods) * F_sum @ self.Q.T
        return Y_hat
    
    def predict(self, X, low_frequency_T, periods, standardize = False):
        # Obtain necessary sizes
        _, p = X.shape
        k = self.n_components
        
        # Center and scale predictors and targets separately
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)

        # Re-shape X into an low_frequency_T x (p * periods), padding if necessary
        reshaped_X, _ = self.highfrequency_to_lowfrequency_reshape(X, low_frequency_T, periods)
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        P_scaled = self.P / self.sigma2_x
        Omega_P_term = np.kron(np.eye(periods), self.V_prior_inv + self.P.T @ P_scaled)
        XP_matrix = np.zeros([low_frequency_T, periods * k])
        for j in range(periods):
            XP_matrix[:, range(j * k, (j+1) * k)] = reshaped_X[:, range(j * p, (j+1) * p)] @ P_scaled
        F_predicted = np.linalg.solve(Omega_P_term, XP_matrix.T).T
        
        # Predict using estimated factors
        F_sum = np.sum([F_predicted[:, range(j * k, (j+1) * k)] for j in range(periods)])
        Y_hat = (1/periods) * F_sum @ self.Q.T
        return Y_hat

class ProbabilisticTFA_StochasticVolatility:
    def __init__(self, n_components):
        # Fill in components of the class
        self.n_components = n_components
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None
        self.factors = None

    def fit(self, X, Y, standardize = False, track_r2 = False, tolerance = 1e-6, max_iter = 1000, V_prior = None,
                 ewma_lambda_x = 0.94, ewma_lambda_y = None):
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

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])

        # Initial values for the parameters (time-varying volatilities start constant)
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x_initial = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y_initial = np.var(Y, axis = 0).mean()    # Mean variance across targets
        sigma2_x = np.full(T, sigma2_x_initial)
        sigma2_y = np.full(T, sigma2_y_initial)

        # Track R-squared of fit if necessary
        if track_r2:
            r2_list = []
        
        # Start EM algorithm main loop
        M = np.zeros([T, k])
        Omega = np.zeros([T, k, k])
        for _ in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            for t in range(T):
                L_scaled_t = np.vstack([L0[:p] / sigma2_x[t], L0[p:] / sigma2_y[t]])
                Omega[t] = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled_t)
                M[t] = Z[t] @ L_scaled_t @ Omega[t]
            V = np.sum(Omega, axis = 0) + M.T @ M

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
            if track_r2:
                r2_values = r2_score(Y, Z_hat[:, p:], multioutput = "raw_values")
                r2_list.append(r2_values)

            # Check convergence condition
            if (theta_distance <= self.tolerance):
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
        
    def fitted(self, X, Y, standardize = False):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        T, p = X.shape
        F_predicted = np.zeros([T, self.n_components])
        L = np.vstack([self.P, self.Q])
        for t in range(T):
            L_scaled_t = np.vstack([L[:p] / self.sigma2_x[t], L[p:] / self.sigma2_y[t]])
            Omega_inverse = self.V_prior_inv + L.T @ L_scaled_t
            F_predicted[t] = np.linalg.solve(Omega_inverse, L_scaled_t.T @ Z[t])
        
        # Predict using estimated factors and update class
        self.factors = F_predicted
        Y_hat = F_predicted @ self.Q.T
        return Y_hat

        # # Compute prediction variance if necessary and return
        # if not prediction_variance:
        #     return Y_hat
        # else:
        #     q = self.Q.shape[0]
        #     fitted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse, self.Q.T)
        #     return Y_hat, fitted_variance

    def predict(self, X, standardize = False):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        
        # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
        T = X.shape[0]
        F_predicted = np.zeros([T, self.n_components])
        for t in range(T):
            P_scaled = self.P / self.sigma2_x[t]
            Omega_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
            F_predicted[t] = np.linalg.solve(Omega_X_inverse, P_scaled.T @ X[t])

        # Predict using estimated factors
        Y_hat = F_predicted @ self.Q.T
        return Y_hat
        
        # # Compute prediction variance if necessary and return
        # if not prediction_variance:
        #     return Y_hat
        # else:
        #     q = self.Q.shape[0]
        #     predicted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_X_inverse, self.Q.T)
        #     return Y_hat, predicted_variance

class ProbabilisticTFA_DynamicFactors:
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

    def fit(self, X, Y, standardize = False, track_r2 = False, tolerance = 1e-6, max_iter = 1000, V_prior = None):
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

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])

        # Initial values for the parameters
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets
        A0 = np.eye(k)
        f0_0 = np.zeros(k)

        # Track R-squared of fit if necessary
        if track_r2:
            r2_list = []
        
        # Start EM algorithm main loop
        for _ in range(self.max_iter):
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
            if (theta_distance <= self.tolerance):
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
        
    def fitted(self, X, Y, standardize = False, prediction_variance = False):
        # Center and scale predictors and targets separately before stacking
        T = X.shape[0]
        k = self.n_components
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P, self.Q])
        L_scaled = np.vstack([self.P / self.sigma2_x, self.Q / self.sigma2_y])
        L_scaled_L = L.T @ L_scaled
        Sigma_v_A = - self.V_prior_inv @ self.A
        A_Sigma_v_A = - self.A.T @ Sigma_v_A
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
                
        # Compute predicted factors using banded matrix solver
        F_predicted = np.ravel(Z @ L_scaled)
        F_predicted[:k] = F_predicted[:k] - Sigma_v_A @ self.f0
        F_predicted = linalg.cho_solve_banded((Omega_inv_cholesky, True), b = F_predicted, overwrite_b=True).reshape([T, k])

        # Predict using estimated factors and update class
        self.factors = F_predicted
        Y_hat = F_predicted @ self.Q.T
        return Y_hat

        # # Compute prediction variance if necessary and return
        # if not prediction_variance:
        # else:
        #     q = self.Q.shape[0]
        #     fitted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_inverse, self.Q.T)
        #     return Y_hat, fitted_variance

    def predict(self, X, standardize = False, prediction_variance = False, method = "mean"):
        # Center and scale predictors and targets separately before stacking
        T = X.shape[0]
        k = self.n_components
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
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

        # Predict using estimated factors
        Y_hat = F_predicted @ self.Q.T
        return Y_hat
    
        # # Compute prediction variance if necessary and return
        # if not prediction_variance:
            
        # else:
        #     q = self.Q.shape[0]
        #     predicted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Omega_X_inverse, self.Q.T)
        #     return Y_hat, predicted_variance

        