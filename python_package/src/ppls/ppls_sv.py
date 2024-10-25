import numpy as np
from sklearn.metrics import r2_score

class ProbabilisticPLS:
    def __init__(self, n_components, tolerance = 1e-6, max_iter = 1000, V_prior = None, ewma_lambda=0.94):
        # Fill in components of the class
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.ewma_lambda = ewma_lambda  # EWMA smoothing parameter
        if V_prior is None:
            self.V_prior = np.eye(n_components)
        self.V_prior_inv = np.eye(n_components) if V_prior is None else np.linalg.inv(V_prior)
        
        # Pre-allocate memory for estimates
        self.P = None
        self.Q = None
        self.sigma2_x = None
        self.sigma2_y = None

    def fit(self, X, Y, standardize = False, track_r2 = False):
        # Decide whether to pass to standard or missing data EM based on data
        X_missing_index = np.isnan(X)
        Y_missing_index = np.isnan(Y)
        missing_data = np.any(X_missing_index) or np.any(Y_missing_index)

        # Dispatch to relevant lower-level function
        if not missing_data:
            self.fit_ppls(X, Y, standardize, track_r2)
        else:
            self.fit_ppls_missing(X, X_missing_index, Y, Y_missing_index, standardize, track_r2)
    
    def fitted(self, X, Y, standardize = False, prediction_variance = False):
        # Decide whether to pass to standard or missing data EM based on data
        X_missing_index = np.isnan(X)
        Y_missing_index = np.isnan(Y)
        missing_data = np.any(X_missing_index) or np.any(Y_missing_index)

        # Dispatch to relevant lower-level function
        if not missing_data:
            return self.fitted_ppls(X, Y, standardize, prediction_variance)
        else:
            return self.fitted_ppls_missing(X, X_missing_index, Y, Y_missing_index, standardize, prediction_variance)
        
    def predict(self, X, standardize = False, prediction_variance = False, method = "mean"):
        # Decide whether to pass to standard or missing data EM based on data
        X_missing_index = np.isnan(X)
        missing_data = np.any(X_missing_index)
        
        # Dispatch to relevant lower-level function
        if not missing_data:
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
        #sigma2_x0 = 1 
        #sigma2_y0 = 1
        # Initialize time-varying volatilities with constant initial values
        sigma2_x_t = np.full(n, sigma2_x0)
        sigma2_y_t = np.full(n, sigma2_y0)

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
            L_scaled = np.vstack([L0[:p] / sigma2_x_t.mean(), L0[p:] / sigma2_y_t.mean()])
            Sigma = np.linalg.inv(self.V_prior_inv + L0.T @ L_scaled)
            M = Z @ L_scaled @ Sigma
            
            # Maximization step: Update factor loadings and variances
            V = n * Sigma + M.T @ M
            L1 = np.linalg.solve(V, M.T @ Z).T
            P = L1[:p]
            Q = L1[p:]
            #sigma2_x1 = (1/(n * p)) * (np.trace(X.T @ X) - np.trace(P.T @ P @ V))
            #sigma2_y1 = (1/(n * q)) * (np.trace(Y.T @ Y) - np.trace(Q.T @ Q @ V))
            # Compute residuals
            X_hat = M @ P.T
            Y_hat = M @ Q.T
            residuals_X = X - X_hat
            residuals_Y = Y - Y_hat      
            # Update volatilities using EWMA on the residuals over time
            sigma2_x_t = self._update_ewma_volatility(residuals_X, self.ewma_lambda)
            sigma2_y_t = self._update_ewma_volatility(residuals_Y, self.ewma_lambda)
            # Compute distance between iterates
            P_distance = np.linalg.norm(P - L0[:p], "fro")
            Q_distance = np.linalg.norm(Q - L0[p:], "fro")
            #sigma_x_distance = np.abs(sigma2_x1 - sigma2_x0)
            #sigma_y_distance = np.abs(sigma2_y1 - sigma2_y0)
            theta_distance = sum([P_distance, Q_distance])
            
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
                #sigma2_x0 = sigma2_x1
                #sigma2_y0 = sigma2_y1
        
        # Update final values of the class with results from EM algorithm
        self.P = P
        self.Q = Q
        self.sigma2_x = sigma2_x_t.mean()
        self.sigma2_y = sigma2_y_t.mean()
        self.sigma2_xx = sigma2_x_t  # Store time-varying volatility for X
        self.sigma2_yy = sigma2_y_t  # Store time-varying volatility for Y
        self.r2_array_ = np.asarray(r2_list) if track_r2 else None

    def _update_ewma_volatility(self, residuals, lambda_factor):
        """
        Compute EWMA volatility for the residuals over time.
        :param residuals: Residuals of the model (n x p) where n is time and p is the number of features.
        :param lambda_factor: The EWMA decay factor (0 < lambda_factor < 1).
        :return: A (n,) array with the time-varying volatility.
        """
        n = residuals.shape[0]  # Get number of time points
        #print(residuals.shape)
        sigma2 = np.zeros(n)  # Initialize a vector to store the variances over time

        # Initialize the first variance estimate (across all features, isotropic assumption)
        sigma2[0] = np.mean(residuals[0, :]**2)  # Mean variance at t = 0
        #sigma2[0] = 1
        # Apply EWMA for each time step
        for t in range(1, n):
            sigma2[t] = lambda_factor * sigma2[t-1] + (1 - lambda_factor) * np.mean(residuals[t, :]**2)

        return sigma2  # Return the variance (no need for square root since it's variance)
        
    def fitted_ppls(self, X, Y, standardize, prediction_variance):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
            Y = (Y - Y.mean(axis = 0)) / Y.std(axis = 0)
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P, self.Q])
        L_scaled = np.vstack([self.P / self.sigma2_x, self.Q / self.sigma2_y])
        Sigma_inverse = self.V_prior_inv + L.T @ L_scaled
        F_predicted = np.linalg.solve(Sigma_inverse, L_scaled.T @ Z.T).T

        # Predict using estimated factors
        Y_hat = F_predicted @ self.Q.T

        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            fitted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Sigma_inverse, self.Q.T)
            return Y_hat, fitted_variance

    def predict_ppls(self, X, standardize, prediction_variance, method = "mean"):
        # Center and scale predictors and targets separately before stacking
        if standardize:
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        
        # Explore differences in prediction methods
        P_scaled = self.P / self.sigma2_x
        Sigma_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        if method == "mean":
            # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
            F_predicted = np.linalg.solve(Sigma_X_inverse, P_scaled.T @ X.T).T

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
            predicted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Sigma_X_inverse, self.Q.T)
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
        self.P = P
        self.Q = Q
        self.sigma2_x = sigma2_x1
        self.sigma2_y = sigma2_y1
        self.r2_array_ = np.asarray(r2_list) if track_r2 else None
        self.Z_hat_ = Z_hat

    def fitted_ppls_missing(self, X, X_missing_index, Y, Y_missing_index, standardize, prediction_variance):
        # Center and scale predictors and targets separately if necessary
        p = self.P.shape[0]
        if standardize:
            X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
            Y = (Y - Y.mean(axis = 0, where = np.logical_not(Y_missing_index))) / Y.std(axis = 0, where = np.logical_not(Y_missing_index))

        # Fill missing values with prediction from model
        X[X_missing_index] = self.Z_hat_[:, :p][X_missing_index]
        Y[Y_missing_index] = self.Z_hat_[:, p:][Y_missing_index]
        Z = np.hstack([X, Y])
        
        # Obtain predicted factors: F_predicted = M (Posterior mean of factors)
        L = np.vstack([self.P, self.Q])
        L_scaled = np.vstack([self.P / self.sigma2_x, self.Q / self.sigma2_y])
        Sigma_inverse = self.V_prior_inv + L.T @ L_scaled
        F_predicted = np.linalg.solve(Sigma_inverse, L_scaled.T @ Z.T).T

        # Predict using estimated factors
        Y_hat = F_predicted @ self.Q.T

        # Compute prediction variance if necessary and return
        if not prediction_variance:
            return Y_hat
        else:
            q = self.Q.shape[0]
            fitted_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Sigma_inverse, self.Q.T)
            return Y_hat, fitted_variance
    
    def predict_ppls_missing(self, X, X_missing_index, standardize, prediction_variance, method = "mean"):
        # Center and scale predictors and targets separately before stacking
        p = self.P.shape[0]
        if standardize:
            X = (X - X.mean(axis = 0, where = np.logical_not(X_missing_index))) / X.std(axis = 0, where = np.logical_not(X_missing_index))
        
        # Fill missing values with prediction from model
        X[X_missing_index] = self.Z_hat_[:, :p][X_missing_index]

        # Explore differences in prediction methods
        P_scaled = self.P / self.sigma2_x
        Sigma_X_inverse = self.V_prior_inv + self.P.T @ P_scaled
        if method == "mean":
            # Obtain predicted factors using X only: F_predicted = M (Posterior mean of factors)
            F_predicted = np.linalg.solve(Sigma_X_inverse, P_scaled.T @ X.T).T

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
            predict_variance = self.sigma2_y * np.eye(q) + self.Q @ np.linalg.solve(Sigma_X_inverse, self.Q.T)
            return Y_hat, predict_variance