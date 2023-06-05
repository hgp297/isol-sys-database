############################################################################
#               Prediction object for ML models and plotting

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) 

############################################################################



class GP:
    
    # sets up the problem by grabbing the x covariates
    def __init__(self, data):
        self._raw_data = data
        self.k = len(data)
        self.X = data[['gapRatio', 'RI', 'Tm', 'zetaM']]
        
    # sets up prediction variable
    def set_outcome(self, outcome_var):
        self.y = self._raw_data[[outcome_var]]
        
    # Train GP classifier
    def fit_gpc(self, kernel_name, noisy=True):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessClassifier
        import sklearn.gaussian_process.kernels as krn
        
        if kernel_name=='rbf_ard':
            kernel = 1.0 * krn.RBF([1.0, 1.0, 1.0, 1.0])
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
                    length_scale=[1.0, 1.0, 1.0, 1.0], 
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
        
        if kernel_name=='rbf_ard':
            kernel = 1.0 * krn.RBF([1.0, 1.0, 1.0, 1.0])
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
                    length_scale=[1.0, 1.0, 1.0, 1.0], 
                    nu=1.5)
            
        kernel = kernel + krn.WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
            
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
    
    def doe_rejection_sampler(self, n_pts, pr):
        import numpy as np
        n_max = 1000
        
        # start with a sample of x's from a uniform distribution
        x_var = np.array([np.random.uniform(0.3, 2.0, n_max),
                       np.random.uniform(0.5, 2.0, n_max),
                       np.random.uniform(2.5, 4.0, n_max),
                       np.random.uniform(0.1, 0.2, n_max)]).T
        
        # TODO: parameterize the function
        # find the maximum that the function will be in the domain
        # bounds for design space
        from scipy.optimize import basinhopping
        bnds = ((0.3, 2.0), (0.5, 2.0), (2.5, 4.0), (0.1, 0.2))
        
        x0 = np.array([[np.random.uniform(0.3, 2.0),
                       np.random.uniform(0.5, 2.0),
                       np.random.uniform(2.5, 4.0),
                       np.random.uniform(0.1, 0.2)]])
        
        # use basin hopping to avoid local minima
        minimizer_kwargs={'args':(pr),'bounds':bnds}
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
        fx = -self.fn_tmse(x_var, pr).T
        x_keep = x_var[u_var.ravel() < fx,:]
        
        # import matplotlib.pyplot as plt
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["mathtext.fontset"] = "dejavuserif"
        # import matplotlib as mpl
        # label_size = 16
        # mpl.rcParams['xtick.labelsize'] = label_size
        # mpl.rcParams['ytick.labelsize'] = label_size
        # plt.scatter(x_keep[:,0], x_keep[:,1])
        
        return x_keep[np.random.choice(x_keep.shape[0], n_pts, replace=False),:]
        
  
    def doe_tmse(self, pr):
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
        
        # initialize a guess
        x0 = np.array([[random.uniform(0.3, 2.0),
                       random.uniform(0.5, 2.0),
                       random.uniform(2.5, 4.0),
                       random.uniform(0.1, 0.2)]])
        
        # bounds for design space
        bnds = ((0.3, 2.0), (0.5, 2.0), (2.5, 4.0), (0.1, 0.2))
        
        # find argmax tmse criterion
        res = minimize(self.fn_tmse, x0, 
                       args=(pr),
                       method='Nelder-Mead', tol=1e-6,
                       bounds=bnds)
        
        x_next = pd.DataFrame(res.x.reshape(1,-1), columns=['gapRatio',
                                                            'RI',
                                                            'Tm',
                                                            'zetaM'])
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
    def fn_tmse(self, X_cand, pr):
        
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