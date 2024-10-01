import numpy as np
from sklearn.metrics import r2_score

def generate_synthetic_data(n, p, q, k, sigma_x, sigma_y, seed_value=None):
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [n, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [n, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [n, q])
    
    return X, Y, F, P, Q

def generate_synthetic_missing_data(n, p, q, k, sigma_x, sigma_y, proportion_x=0.1, proportion_y=None,
                                    seed_value=None, return_nan=True):
    # Generate latent variables
    rng = np.random.default_rng(seed=seed_value)
    F = rng.normal(size = [n, k])
    
    # Generate loadings
    P = rng.normal(size = [p, k])
    Q = rng.normal(size = [q, k])
    
    # Generate predictor and response variables with added noise
    X = F @ P.T + rng.normal(scale = sigma_x, size = [n, p])
    Y = F @ Q.T + rng.normal(scale = sigma_y, size = [n, q])
    
    # Select indices to turn into missing observations
    if proportion_y is None:
        proportion_y = proportion_x
    X_size = n * p
    Y_size = n * q
    num_missing_X = int(proportion_x * X_size)
    num_missing_Y = int(proportion_y * Y_size)
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
    else:
        # Return the data as a masked object
        X = np.ma.MaskedArray(data=X, mask=X_missing_mask, fill_value=0.0)
        Y = np.ma.MaskedArray(data=Y, mask=Y_missing_mask, fill_value=0.0)
    
    return X, Y, F, P, Q

class ProbabilisticPLS_Old: 
    def __init__(self, n_components, max_iter=1000, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.P_ = None
        self.Q_ = None
        self.sigma_x_ = None
        self.sigma_y_ = None
        self.M_ = None

    def fit(self, X, Y):
        n, p = X.shape
        _, q = Y.shape
        k = self.n_components
        
        # Initialize parameters
        P = np.random.randn(p, k)
        Q = np.random.randn(q, k)
        sigma_x = np.var(X, axis=0).mean()  # Mean variance across features
        sigma_y = np.var(Y, axis=0).mean()  # Mean variance across targets
        #print(sigma_x)
        #print(sigma_y)

        #def orthogonalize(matrix):
        #    q, _ = np.linalg.qr(matrix)
        #    return q    
        
        # EM algorithm
        for iteration in range(self.max_iter):
            #print(iteration)
            # E-step: Compute posterior mean M and covariance Sigma
            A = (P.T @ P) / sigma_x + (Q.T @ Q) / sigma_y + np.eye(k)
            Sigma = np.linalg.inv(A)
            B = (X @ P) / sigma_x + (Y @ Q) / sigma_y
            M = np.linalg.solve(A, B.T).T
            
            # M-step: Update P, Q, sigma_x, and sigma_y
            P_new = (X.T @ M) @ np.linalg.inv(M.T @ M + n * Sigma)
            Q_new = (Y.T @ M) @ np.linalg.inv(M.T @ M + n * Sigma)
            
            sigma_x_new = (1 / (n * p)) * (np.linalg.norm(X - M @ P_new.T, 'fro')**2) + (1 / p) * np.trace(P_new.T @ P_new @ Sigma)
            sigma_y_new = (1 / (n * q)) * (np.linalg.norm(Y - M @ Q_new.T, 'fro')**2) + (1 / q)  * np.trace(Q_new.T @ Q_new @ Sigma)
            
            # Orthogonalize P and Q
            #P_new = orthogonalize(P_new)
            #Q_new = orthogonalize(Q_new)   
            #M = orthogonalize(M)

            # Check for convergence
            if np.linalg.norm(P - P_new) < self.tol and np.linalg.norm(Q - Q_new) < self.tol and abs(sigma_x - sigma_x_new) < self.tol and abs(sigma_y - sigma_y_new) < self.tol:
                break
            
            P, Q, sigma_x, sigma_y, it = P_new, Q_new, sigma_x_new, sigma_y_new, iteration
            
        self.P_ = P
        self.Q_ = Q
        self.sigma_x_ = sigma_x
        self.sigma_y_ = sigma_y
        self.M_ = M
        
    def predict(self, X):
        return self.M_ @ self.Q_.T

class ProbabilisticPLS:
    def __init__(self, n_components, tolerance = 1e-6, max_iter = 1000, V_prior = None):
        # Fill in components of the class
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        if V_prior is None:
            self.V_prior = np.eye(n_components)
        self.V_prior_inv = np.eye(n_components) if V_prior is None else np.linalg.inv(V_prior)
        
        # Pre-allocate memory for estimates
        self.P_ = None
        self.Q_ = None
        self.sigma2_x_ = None
        self.sigma2_y_ = None

    def fit(self, X, Y, standardize = False, track_r2 = False):
        # Decide whether to pass to standard or missing data EM based on data
        X_missing_index = np.isnan(X)
        Y_missing_index = np.isnan(Y)
        no_missing = np.sum(X_missing_index) == 0 and np.sum(Y_missing_index) == 0

        # Dispatch to relevant lower-level function
        if no_missing:
            self.fit_ppls(X, Y, standardize, track_r2)
        else:
            self.fit_ppls_missing(X, X_missing_index, Y, Y_missing_index, standardize, track_r2)
    
    def fitted(self, X, Y, standardize = False, prediction_variance = False):
        # Decide whether to pass to standard or missing data EM based on data
        X_missing_index = np.isnan(X)
        Y_missing_index = np.isnan(Y)
        no_missing = np.sum(X_missing_index) == 0 and np.sum(Y_missing_index) == 0
        
        # Dispatch to relevant lower-level function
        if no_missing:
            return self.fitted_ppls(X, Y, standardize, prediction_variance)
        else:
            return self.fitted_ppls_missing(X, X_missing_index, Y, Y_missing_index, standardize, prediction_variance)
        
    def predict(self, X, standardize = False, prediction_variance = False, method = "mean"):
        # Decide whether to pass to standard or missing data EM based on data
        X_missing_index = np.isnan(X)
        no_missing = np.sum(X_missing_index) == 0
        
        # Dispatch to relevant lower-level function
        if no_missing:
            return self.predict_ppls(X, standardize, prediction_variance, method)
        else:
            return self.predict_ppls_missing(X, X_missing_index, standardize, prediction_variance, method)

    def fit_ppls(self, X, Y, standardize, track_r2):
        # Obtain sizes
        # X is n x p; Y is n x q; Factors assumed as n x k
        n, p = X.shape
        _, q = Y.shape
        k = self.n_components
        d = p + q

        # Initial values for the parameters
        L0 = np.random.default_rng().normal(size = [d, k])
        sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
        sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])

        # Track R-squared of fit if necessary
        if track_r2:
            r2_list = []
        
        # Start EM algorithm main loop
        for _ in range(self.max_iter):
            # Expectation step: Update posterior paramater for factors
            L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
            Sigma = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Sigma
            
            # Maximization step: Update factor loadings and variances
            V = n * Sigma + M.T @ M
            L1 = np.linalg.solve(V, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            sigma2_x1 = (1/(n * p)) * (np.trace(X.T @ X) - np.trace(P.T @ P @ V))
            sigma2_y1 = (1/(n * q)) * (np.trace(Y.T @ Y) - np.trace(Q.T @ Q @ V))
            
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
        self.P_ = P
        self.Q_ = Q
        self.sigma2_x_ = sigma2_x1
        self.sigma2_y_ = sigma2_y1
        self.r2_array_ = np.asarray(r2_list) if track_r2 else None
        
    def fitted_ppls(self, X, Y, standardize, prediction_variance):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P_, self.Q_])
        L_scaled = np.vstack([self.P_ / self.sigma2_x_, self.Q_ / self.sigma2_y_])
        Sigma_inverse = self.V_prior_inv + L.T @ L_scaled
        F_predicted = np.linalg.solve(Sigma_inverse, L_scaled.T @ Z.T).T

        # Predict using estimated factors
        Y_hat = F_predicted @ self.Q_.T

        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q_.shape[0]
            fitted_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_inverse, self.Q_.T)
            return Y_hat, fitted_variance

    def predict_ppls(self, X, standardize, prediction_variance, method = "mean"):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        
        # Explore differences in prediction methods
        P_scaled = self.P_ / self.sigma2_x_
        Sigma_X_inverse = self.V_prior_inv + self.P_.T @ P_scaled
        if method == "mean":
            # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
            F_predicted = np.linalg.solve(Sigma_X_inverse, P_scaled.T @ X.T).T

            # Predict using estimated factors
            Y_hat = F_predicted @ self.Q_.T
        elif method == "loading":
            # Predict using estimated loadings directly
            p = self.P_.shape[0]
            C_X = self.sigma2_x_ * np.eye(p) + self.P_ @ self.V_prior @ self.P_.T
            Y_hat = X @ np.linalg.solve(C_X, self.P_ @ self.V_prior @ self.Q_.T)
        
        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q_.shape[0]
            predicted_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_X_inverse, self.Q_.T)
            return Y_hat, predicted_variance
        
    def fit_ppls_missing(self, X, X_missing_index, Y, Y_missing_index, standardize, track_r2):
        # Obtain sizes
        # X is n x p; Y is n x q; Factors assumed as n x k
        n, p = X.shape
        _, q = Y.shape
        k = self.n_components
        d = p + q

        # Center and scale predictors and targets separately if necessary
        if standardize:
            X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - Y.mean(axis = 0, where = np.logical_not(Y_missing_index))) / Y.std(axis = 0, where = np.logical_not(Y_missing_index))

        # Initial imputation step before stacking
        X[X_missing_index] = 0
        Y[Y_missing_index] = 0
        Z = np.hstack([X, Y])
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
            Sigma = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Sigma
            
            # Update missing data using current EM fitted values
            Z_hat = M @ L0.T
            X[X_missing_index] = Z_hat[:, :p][X_missing_index]
            Y[Y_missing_index] = Z_hat[:, p:][Y_missing_index]
            Z = np.hstack([X, Y])

            # Maximization step: Update factor loadings and variances
            V = n * Sigma + M.T @ M
            L1 = np.linalg.solve(V, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            sigma2_x1 = (1/(n * p)) * (np.trace(X.T @ X) - np.trace(P.T @ P @ V))
            sigma2_y1 = (1/(n * q)) * (np.trace(Y.T @ Y) - np.trace(Q.T @ Q @ V))
            
            # Compute distance between iterates
            P_distance = np.linalg.norm(P - L0[:p], "fro")
            Q_distance = np.linalg.norm(Q - L0[p:], "fro")
            sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
            # Prediction and tracking of R-squared across iterations
            if track_r2:
                Y_hat = M @ Q.T
                r2_values = r2_score(Y[np.logical_not(Y_missing_index)],
                                     Y_hat[np.logical_not(Y_missing_index)],
                                     multioutput = "raw_values")
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
        self.P_ = P
        self.Q_ = Q
        self.sigma2_x_ = sigma2_x1
        self.sigma2_y_ = sigma2_y1
        self.r2_array_ = np.asarray(r2_list) if track_r2 else None
        self.Z_hat_ = Z_hat

    def fitted_ppls_missing(self, X, X_missing_index, Y, Y_missing_index, standardize, prediction_variance):
        # Center and scale predictors and targets separately if necessary
        p = self.P_.shape[0]
        if standardize:
            X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - Y.mean(axis = 0, where = np.logical_not(Y_missing_index))) / Y.std(axis = 0, where = np.logical_not(Y_missing_index))

        # Fill missing values with prediction from model
        X[X_missing_index] = self.Z_hat_[:, :p][X_missing_index]
        Y[Y_missing_index] = self.Z_hat_[:, p:][Y_missing_index]
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P_, self.Q_])
        L_scaled = np.vstack([self.P_ / self.sigma2_x_, self.Q_ / self.sigma2_y_])
        Sigma_inverse = self.V_prior_inv + L.T @ L_scaled
        F_predicted = np.linalg.solve(Sigma_inverse, L_scaled.T @ Z.T).T

        # Predict using estimated factors
        Y_hat = F_predicted @ self.Q_.T

        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q_.shape[0]
            fitted_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_inverse, self.Q_.T)
            return Y_hat, fitted_variance
    
    def predict_ppls_missing(self, X, X_missing_index, standardize, prediction_variance, method = "mean"):
        # Center and scale predictors and targets separately before stacking
        p = self.P_.shape[0]
        if standardize:
            X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
        
        # Fill missing values with prediction from model
        X[X_missing_index] = self.Z_hat_[:, :p][X_missing_index]

        # Explore differences in prediction methods
        P_scaled = self.P_ / self.sigma2_x_
        Sigma_X_inverse = self.V_prior_inv + self.P_.T @ P_scaled
        if method == "mean":
            # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
            F_predicted = np.linalg.solve(Sigma_X_inverse, P_scaled.T @ X.T).T

            # Predict using estimated factors
            Y_hat = F_predicted @ self.Q_.T
        elif method == "loading":
            # Predict using estimated loadings directly
            C_X = self.sigma2_x_ * np.eye(p) + self.P_ @ self.V_prior @ self.P_.T
            Y_hat = X @ np.linalg.solve(C_X, self.P_ @ self.V_prior @ self.Q_.T)
        
        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q_.shape[0]
            predict_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_X_inverse, self.Q_.T)
            return Y_hat, predict_variance
    
# class ProbabilisticPLS_Stacked: 
#     def __init__(self, n_components, tolerance = 1e-6, max_iter = 1000, V_prior = None):
#         # Fill in components of the class
#         self.n_components = n_components
#         self.max_iter = max_iter
#         self.tolerance = tolerance
#         self.V_prior_inv = np.eye(n_components) if V_prior is None else np.linalg.inv(V_prior)
        
#         # Pre-allocate memory for estimates
#         self.P = None
#         self.Q = None
#         self.sigma2_x = None
#         self.sigma2_y = None

#     def fit(self, X, Y, standardize = False, track_r2 = False):
#         # Obtain sizes
#         # X is n x p; Y is n x q; Factors assumed as n x k
#         n, p = X.shape
#         _, q = Y.shape
#         k = self.n_components
#         d = p + q

#         # Initial values for the parameters
#         L0 = np.random.default_rng().normal(size = [d, k])
#         sigma2_x0 = np.var(X, axis = 0).mean()    # Mean variance across features
#         sigma2_y0 = np.var(Y, axis = 0).mean()    # Mean variance across targets

#         # Center and scale predictors and targets separately before stacking
#         if standardize:
#             X = (X - X.mean(axis = 0)) / X.std(axis = 0)
#             Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
#         Z = np.hstack([X, Y])

#         # Track R-squared of fit if necessary
#         if track_r2:
#             r2_list = []
        
#         # Start EM algorithm main loop
#         for _ in range(self.max_iter):
#             # Expectation step: Update posterior paramater for factors
#             L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
#             Sigma = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
#             M = Z @ L_scaled @ Sigma
            
#             # Maximization step: Update factor loadings and variances
#             V = n * Sigma + M.T @ M
#             L1 = np.linalg.solve(V, M.T @ Z).T
#             P = L1[:p]
#             Q = L1[p:]
#             sigma2_x1 = (1/(n * p)) * (np.trace(X.T @ X) - np.trace(P.T @ P @ V))
#             sigma2_y1 = (1/(n * q)) * (np.trace(Y.T @ Y) - np.trace(Q.T @ Q @ V))
            
#             # Compute distance between iterates
#             P_distance = np.linalg.norm(P - L0[:p], "fro")
#             Q_distance = np.linalg.norm(Q - L0[p:], "fro")
#             sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
#             sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
#             theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
#             # Prediction and tracking of R-squared across iterations
#             if track_r2:
#                 Y_hat = M @ Q.T
#                 r2_values = r2_score(Y, Y_hat, multioutput = "raw_values")
#                 r2_list.append(r2_values)

#             # Check convergence condition
#             if (theta_distance <= self.tolerance):
#                 # Break if distance between each estimate is less than a tolerance
#                 break
#             else:
#                 # Prepare values for next iteration if convergence not reached
#                 L0 = L1
#                 sigma2_x0 = sigma2_x1
#                 sigma2_y0 = sigma2_y1
        
#         # Update values of the class with results from EM algorithm
#         self.P_ = P
#         self.Q_ = Q
#         self.sigma2_x_ = sigma2_x1
#         self.sigma2_y_ = sigma2_y1
#         self.r2_array_ = np.asarray(r2_list) if track_r2 else None
        
#     def fitted(self, X, Y, standardize = False):
#         # Center and scale predictors and targets separately before stacking
#         # q = self.Q_.shape[0]
#         if standardize:
#             X = (X - X.mean(axis = 0)) / X.std(axis = 0)
#             Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
#         Z = np.hstack([X, Y])
        
#         # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
#         L = np.vstack([self.P_, self.Q_])
#         L_scaled = np.vstack([self.P_ / self.sigma2_x_, self.Q_ / self.sigma2_y_])
#         Sigma_inverse = self.V_prior_inv + L.T @ L_scaled
#         F_predicted = np.linalg.solve(Sigma_inverse, L_scaled.T @ Z.T).T

#         # Predict using estimated factors
#         Y_hat = F_predicted @ self.Q_.T
#         # fitted_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_inverse, self.Q_.T)
#         return Y_hat #, fitted_variance

#     def predict(self, X, standardize = False):
#         # Center and scale predictors and targets separately before stacking
#         # q = self.Q_.shape[0]
#         if standardize:
#             X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        
#         # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
#         P_scaled = self.P_ / self.sigma2_x_
#         Sigma_X_inverse = self.V_prior_inv + self.P_.T @ P_scaled
#         F_predicted = np.linalg.solve(Sigma_X_inverse, P_scaled.T @ X.T).T

#         # Predict using estimated factors
#         Y_hat = F_predicted @ self.Q_.T
#         # predict_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_X_inverse, self.Q_.T)
#         return Y_hat # , predict_variance

# class ProbabilisticPLS_Missing:
#     def __init__(self, n_components, tolerance = 1e-6, max_iter = 1000, V_prior = None):
#         # Fill in components of the class
#         self.n_components = n_components
#         self.max_iter = max_iter
#         self.tolerance = tolerance
#         self.V_prior_inv = np.eye(n_components) if V_prior is None else np.linalg.inv(V_prior)
        
#         # Pre-allocate memory for estimates
#         self.P = None
#         self.Q = None
#         self.sigma2_x = None
#         self.sigma2_y = None

#     def fit(self, X, Y, standardize = False, track_r2 = False):
#         # Obtain sizes
#         # X is n x p; Y is n x q; Factors assumed as n x k
#         n, p = X.shape
#         _, q = Y.shape
#         k = self.n_components
#         d = p + q

#         # Center and scale predictors and targets separately if necessary
#         X_missing_index = np.isnan(X)
#         Y_missing_index = np.isnan(Y)
#         if standardize:
#             X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
#             Y = (Y - Y.mean(axis = 0, where = np.logical_not(Y_missing_index))) / Y.std(axis = 0, where = np.logical_not(Y_missing_index))

#         # Initial imputation step before stacking
#         X[X_missing_index] = 0
#         Y[Y_missing_index] = 0
#         Z = np.hstack([X, Y])
#         Z_vars = np.var(Z, axis = 0)

#         # Initial values for the parameters
#         L0 = np.random.default_rng().normal(size = [d, k])
#         sigma2_x0 = np.mean(Z_vars[:p])    # Mean variance across features
#         sigma2_y0 = np.mean(Z_vars[p:])    # Mean variance across targets     

#         # Track R-squared of fit if necessary
#         if track_r2:
#             r2_list = []
        
#         # Start EM algorithm main loop
#         for _ in range(self.max_iter):
#             # Expectation step: Update posterior paramater for factors
#             L_scaled = np.vstack([L0[:p] / sigma2_x0, L0[p:] / sigma2_y0])
#             Sigma = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
#             M = Z @ L_scaled @ Sigma
            
#             # Update missing data using current EM fitted values
#             Z_hat = M @ L0.T
#             X[X_missing_index] = Z_hat[:, :p][X_missing_index]
#             Y[Y_missing_index] = Z_hat[:, p:][Y_missing_index]
#             Z = np.hstack([X, Y])

#             # Maximization step: Update factor loadings and variances
#             V = n * Sigma + M.T @ M
#             L1 = np.linalg.solve(V, M.T @ Z).T
#             P = L1[:p]
#             Q = L1[p:]
#             sigma2_x1 = (1/(n * p)) * (np.trace(X.T @ X) - np.trace(P.T @ P @ V))
#             sigma2_y1 = (1/(n * q)) * (np.trace(Y.T @ Y) - np.trace(Q.T @ Q @ V))
            
#             # Compute distance between iterates
#             P_distance = np.linalg.norm(P - L0[:p], "fro")
#             Q_distance = np.linalg.norm(Q - L0[p:], "fro")
#             sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
#             sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
#             theta_distance = sum([P_distance, Q_distance, sigma_x_distance, sigma_y_distance])
            
#             # Prediction and tracking of R-squared across iterations
#             if track_r2:
#                 Y_hat = M @ Q.T
#                 r2_values = r2_score(Y[np.logical_not(Y_missing_index)],
#                                      Y_hat[np.logical_not(Y_missing_index)],
#                                      multioutput = "raw_values")
#                 r2_list.append(r2_values)

#             # Check convergence condition
#             if (theta_distance <= self.tolerance):
#                 # Break if distance between each estimate is less than a tolerance
#                 break
#             else:
#                 # Prepare values for next iteration if convergence not reached
#                 L0 = L1
#                 sigma2_x0 = sigma2_x1
#                 sigma2_y0 = sigma2_y1
        
#         # Update values of the class with results from EM algorithm
#         self.P_ = P
#         self.Q_ = Q
#         self.sigma2_x_ = sigma2_x1
#         self.sigma2_y_ = sigma2_y1
#         self.r2_array_ = np.asarray(r2_list) if track_r2 else None
#         self.Z_hat_ = Z_hat

#     def fitted(self, X, Y, standardize = False):
#         # Center and scale predictors and targets separately if necessary
#         p = self.P_.shape[0]
#         X_missing_index = np.isnan(X)
#         Y_missing_index = np.isnan(Y)
#         if standardize:
#             X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
#             Y = (Y - Y.mean(axis = 0, where = np.logical_not(Y_missing_index))) / Y.std(axis = 0, where = np.logical_not(Y_missing_index))

#         # Fill missing values with prediction from model
#         X[X_missing_index] = self.Z_hat_[:, :p][X_missing_index]
#         Y[Y_missing_index] = self.Z_hat_[:, p:][Y_missing_index]
#         Z = np.hstack([X, Y])
        
#         # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
#         L = np.vstack([self.P_, self.Q_])
#         L_scaled = np.vstack([self.P_ / self.sigma2_x_, self.Q_ / self.sigma2_y_])
#         Sigma_inverse = self.V_prior_inv + L.T @ L_scaled
#         F_predicted = np.linalg.solve(Sigma_inverse, L_scaled.T @ Z.T).T

#         # Predict using estimated factors
#         Y_hat = F_predicted @ self.Q_.T
#         # fitted_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_inverse, self.Q_.T)
#         return Y_hat #, fitted_variance

#     def predict(self, X, standardize = False):
#         # Center and scale predictors and targets separately before stacking
#         p = self.P_.shape[0]
#         X_missing_index = np.isnan(X)
#         if standardize:
#             X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
        
#         # Fill missing values with prediction from model
#         X[X_missing_index] = self.Z_hat_[:, :p][X_missing_index]

#         # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
#         P_scaled = self.P_ / self.sigma2_x_
#         Sigma_X_inverse = self.V_prior_inv + self.P_.T @ P_scaled
#         F_predicted = np.linalg.solve(Sigma_X_inverse, P_scaled.T @ X.T).T

#         # Predict using estimated factors
#         Y_hat = F_predicted @ self.Q_.T
#         # predict_variance = self.sigma2_y_ * np.eye(q) + self.Q_ @ np.linalg.solve(Sigma_X_inverse, self.Q_.T)
#         return Y_hat # , predict_variance