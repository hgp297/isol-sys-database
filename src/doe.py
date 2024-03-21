############################################################################
#               Prediction object for ML models and plotting

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) 

############################################################################

# TODO: implement Sangri's IMSE_w with LOOCV bias

class GP:
    
    # sets up the problem by grabbing the x covariates
    def __init__(self, data):
        self._raw_data = data
        self.k = len(data)
        
    def set_covariates(self, var_list):
        self.X = self._raw_data[var_list]
        
    # sets up prediction variable
    def set_outcome(self, outcome_var):
        self.y = self._raw_data[[outcome_var]]
        
    # Train GP classifier
    def fit_gpc(self, kernel_name, noisy=True):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessClassifier
        import sklearn.gaussian_process.kernels as krn
        
        import numpy as np
        n_vars = self.X.shape[1]
        
        if kernel_name=='rbf_ard':
            kernel = 1.0 * krn.RBF(np.ones(n_vars))
        elif kernel_name=='rbf_iso':
            kernel = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel = 1.0 * krn.Matern(
                    length_scale=np.ones(n_vars), 
                    nu=1.5)

        if noisy==True:
            kernel = kernel + krn.WhiteKernel(noise_level=0.5)
        # pipeline to scale -> GPC
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpc', GaussianProcessClassifier(kernel=kernel,
                                                  warm_start=True,
                                                  random_state=985,
                                                  max_iter_predict=250))
                ])
    
        gp_pipe.fit(self.X, self.y)
        tr_scr = gp_pipe.score(self.X, self.y)
        print("The GP training score is %0.2f"
              %tr_scr)
        
        self.gpc = gp_pipe
    
    # Train GP regression
    def fit_gpr(self, kernel_name):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        import sklearn.gaussian_process.kernels as krn
        
        import numpy as np
        n_vars = self.X.shape[1]
        
        if kernel_name=='rbf_ard':
            kernel = 1.0 * krn.RBF(np.ones(n_vars))
        elif kernel_name=='rbf_iso':
            kernel = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel = 1.0 * krn.Matern(
                    length_scale=np.ones(n_vars), 
                    nu=1.5)
            
        kernel = kernel + krn.WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e3))
            
        # pipeline to scale -> GPR
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpr', GaussianProcessRegressor(kernel=kernel,
                                                 random_state=985,
                                                 n_restarts_optimizer=10))
                ])
    
        gp_pipe.fit(self.X, self.y)
        
        self.gpr = gp_pipe
            
    def predict_gpc_latent(self, X):
        """Return latent mean and variance for the test vector X.
        Uses Laplace approximation (Williams & Rasmussen Algorithm 3.2)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.
    
        Returns
        -------
        f_star : array-like of shape (n_samples, n_classes)
            Latent mean
        var_f_star : array-like of shape (n_samples, n_classes)
            Latent variance
        """
        # Based on Algorithm 3.2 of GPML
        from scipy.linalg import solve
        # from scipy.special import erf
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        mdl_gpc = self.gpc.named_steps.gpc.base_estimator_
        
        # Based on Algorithm 3.2 of GPML
        scaler = StandardScaler()
        scaler.fit(self.X)
        X_tr = scaler.transform(X)
        K_star = mdl_gpc.kernel_(mdl_gpc.X_train_, X_tr)  # K_star =k(x_star)
        f_star = K_star.T.dot(mdl_gpc.y_train_ - mdl_gpc.pi_)  # Line 4
        v = solve(mdl_gpc.L_, mdl_gpc.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = mdl_gpc.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
        
        return(f_star, var_f_star)
    
    def doe_rejection_sampler(self, n_pts, pr, bound_df, design_filter=False):
        import numpy as np
        n_max = 10000
        n_var = bound_df.shape[1]
        
        min_list = [val for val in bound_df.loc['min']]
        max_list = [val for val in bound_df.loc['max']]
        
        # start with a sample of x's from a uniform distribution
        x_var = np.array([np.random.uniform(min_list[i], max_list[i], n_max)
                          for i in range(n_var)]).T
        
        
        # find the maximum that the function will be in the domain
        from scipy.optimize import basinhopping
        
        # bounds for design space
        bnds = tuple((min_list[i], max_list[i]) for i in range(n_var))
        
        x0 = np.array([np.random.uniform(min_list[i], max_list[i])
                          for i in range(n_var)])
        
        # use basin hopping to avoid local minima
        minimizer_kwargs={'args':(pr, bound_df),'bounds':bnds}
        res = basinhopping(self.fn_tmse, x0, minimizer_kwargs=minimizer_kwargs,
                           niter=100, seed=985)
        
        # res = minimize(self.fn_test, x0, 
        #                args=(pr),
        #                method='Nelder-Mead', tol=1e-8,
        #                bounds=bnds)
        
        c_max = -res.fun
        
        # sample rejection variable
        u_var = np.array([np.random.uniform(0.0, c_max, n_max)]).T
        
        # evaluate the function at x
        fx = -self.fn_tmse(x_var, pr, bound_df).T
        x_keep = x_var[u_var.ravel() < fx,:]
        
        # TODO: change if T_ratio
        # if T_m > 4, zeta must be > 0.15
        if design_filter == True:
            var_list = bound_df.columns.tolist()
            Tm_idx = var_list.index('T_m')
            zeta_idx = var_list.index('zeta_e')
            
            # remove designs that have high period but low damping
            x_keep = x_keep[~((x_keep[:,Tm_idx] > 4) & 
                              (x_keep[:,zeta_idx] < 0.18))]
        
        # import matplotlib.pyplot as plt
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["mathtext.fontset"] = "dejavuserif"
        # import matplotlib as mpl
        # label_size = 16
        # mpl.rcParams['xtick.labelsize'] = label_size
        # mpl.rcParams['ytick.labelsize'] = label_size
        # plt.scatter(x_keep[:,0], x_keep[:,1])
        
        return x_keep[np.random.choice(x_keep.shape[0], n_pts, replace=False),:]
        
  
    def doe_tmse(self, pr, bound_df):
        """Return point that maximizes tMSE criterion (Lyu et al 2021)
        Latent variance is evaluated at the point according to current model 
        with n data points. No hyperparameter optimization is done.

        Parameters
        ----------
        pr : scalar. Probability contour where target is desired.
    
        Returns
        -------
        x_next: array['gap', 'RI', 'TM', 'zeta'] that maximizes tMSE
        """
        from scipy.optimize import minimize
        import random
        import numpy as np
        import pandas as pd
        
        n_var = bound_df.shape[1]
        
        min_list = [val for val in bound_df.loc['min']]
        max_list = [val for val in bound_df.loc['max']]
        
        # initialize a guess
        x0 = np.array([[random.uniform(min_list[i], max_list[i])
                          for i in range(n_var)]])
        
        # bounds for design space
        bnds = tuple((min_list[i], max_list[i]) for i in range(n_var))
        
        # find argmax tmse criterion
        res = minimize(self.fn_tmse, x0, 
                       args=(pr),
                       method='Nelder-Mead', tol=1e-6,
                       bounds=bnds)
        
        x_next = pd.DataFrame(res.x.reshape(1,-1), columns=bound_df.columns)
        return x_next
        
    def fn_W(self, X_cand, pr):
        
        import pandas as pd
        import numpy as np
        if X_cand.ndim == 1:
            X_cand = np.array([X_cand])
        X_cand = pd.DataFrame(X_cand, columns=['gapRatio',
                                               'RI',
                                               'Tm',
                                               'zetaM'])
        
        # try GPC
        try:
            fmu, fs2 = self.predict_gpc_latent(X_cand)
            
            # find target in logistic space
            from scipy.stats import logistic
            T = logistic.ppf(pr)
            
        # if fail, it's GPR
        except:
            # function returns standard deviation
            fmu, fs1 = self.gpr.predict(X_cand, return_std=True)
            fs2 = fs1**2
            
            # target exists directly in probability space
            T = pr
        
        # weight is from Lyu / Picheny
        from numpy import exp
        pi = 3.14159
        Wx = 1/((2*pi*(fs2))**0.5) * exp((-1/2)*((fmu - T)**2/(fs2)))
        
        return(-Wx)
    
    # returns single-point (myopic) tmse condition, Lyu 2021
    def fn_tmse(self, X_cand, pr, bound_df):
        
        import pandas as pd
        import numpy as np
        if X_cand.ndim == 1:
            X_cand = np.array([X_cand])
            
        X_cand = pd.DataFrame(X_cand, columns=bound_df.columns)
        
        # try GPC
        try:
            fmu, fs2 = self.predict_gpc_latent(X_cand)
            
            # find target in logistic space
            from scipy.stats import logistic
            T = logistic.ppf(pr)
            
        # if fail, it's GPR
        except:
            # function returns standard deviation
            fmu, fs1 = self.gpr.predict(X_cand, return_std=True)
            fs2 = fs1**2
            
            # target exists directly in probability space
            T = pr
        
        # weight is from Lyu / Picheny
        from numpy import exp, log
        pi = 3.14159
        Wx = 1/((2*pi*(fs2))**0.5) * exp((-1/2)*((fmu - T)**2/(fs2)))
        log_Wx = -log((2*pi*(fs2))**0.5) * (-1/2)*((fmu - T)**2/(fs2))
        
        return(-fs2 * Wx)
    
    def fn_test(self, X_cand, pr):
        
        import pandas as pd
        import numpy as np
        if X_cand.ndim == 1:
            X_cand = np.array([X_cand])
            
        X_cand = pd.DataFrame(X_cand, columns=['gapRatio',
                                               'RI',
                                               'Tm',
                                               'zetaM'])
        
        # function returns latent mean
        fmu = self.gpr.predict(X_cand)
        return(-fmu)