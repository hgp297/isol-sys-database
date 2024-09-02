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
        
    def set_covariates(self, var_list):
        self.X = self._raw_data[var_list]
        
    # sets up prediction variable
    def set_outcome(self, outcome_var, use_ravel=False):
        if use_ravel:
            self.y = self._raw_data[[outcome_var]].values.ravel()
        else:
            self.y = self._raw_data[[outcome_var]]
        
    def test_train_split(self, percentage):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            test_size=percentage,
                                                            random_state=985)
        
        from numpy import ravel
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = ravel(y_train)
        self.y_test = ravel(y_test)
        
    def fit_linear(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression 
        
        lin_pipe = Pipeline([('scaler', StandardScaler()),
                             ('lin_reg', LinearRegression(fit_intercept=False))])
        
        # fit linear
        lin_pipe.fit(self.X, self.y)
        
        self.lin_reg = lin_pipe
        
    def fit_kde(self):
        import numpy as np
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KernelDensity
        from sklearn.pipeline import Pipeline
        
        kde_pipe = Pipeline([('scaler', StandardScaler()),
                                 ('kde', KernelDensity())])
        
        # cross-validate several parameters
        parameters = [
            {'kde__bandwidth':np.logspace(-1, 1, 20)}
            ]
        
        kde_cv = GridSearchCV(kde_pipe, param_grid=parameters)
        kde_cv.fit(self.X_train)
        
        # set pipeline to use CV params
        kde_pipe.set_params(**kde_cv.best_params_)
        kde_pipe.fit(self.X_train)
        
        self.kde = kde_pipe
        
    # Train GP classifier
    def fit_gpc(self, kernel_name, noisy=True):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessClassifier
        import sklearn.gaussian_process.kernels as krn
        
        import numpy as np
        n_vars = self.X_train.shape[1]
        
        if kernel_name=='rbf_ard':
            kernel_base = 1.0 * krn.RBF(np.ones(n_vars))
        elif kernel_name=='rbf_iso':
            kernel_base = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel_base = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel_base = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel_base = 1.0 * krn.Matern(
                    length_scale=np.ones(n_vars), 
                    nu=1.5)

        if noisy==True:
            kernel_obj = kernel_base + krn.WhiteKernel(noise_level=0.5)
        # pipeline to scale -> GPC
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpc', GaussianProcessClassifier(kernel=kernel_obj,
                                                  warm_start=True,
                                                  random_state=985,
                                                  max_iter_predict=250))
                ])
    
        gp_pipe.fit(self.X_train, self.y_train)
        tr_scr = gp_pipe.score(self.X_train, self.y_train)
        print("The GP training score is %0.3f"
              %tr_scr)
        
        te_scr = gp_pipe.score(self.X_test, self.y_test)
        print("The GP testing score is %0.3f"
              %te_scr)
        
        self.gpc = gp_pipe
        
    def get_kernel(self, X_pr, kernel_name='rbf', gamma=0.25,
                               degree=3):
        if kernel_name=='rbf':
            from sklearn.metrics.pairwise import rbf_kernel
            K_pr = rbf_kernel(X_pr, self.X_train, gamma=gamma)
        elif kernel_name=='poly':
            from sklearn.metrics.pairwise import polynomial_kernel
            K_pr = polynomial_kernel(X_pr, self.X_train, degree=degree)
        elif kernel_name=='sigmoid':
            from sklearn.metrics.pairwise import sigmoid_kernel
            K_pr = sigmoid_kernel(X_pr, self.X_train)
            
        self.K_pr = K_pr
        self.log_reg_kernel.K_pr = K_pr
        return(K_pr)
    
    # TODO: how to pass gamma into predictor
    def fit_kernel_logistic(self, kernel_name='rbf', neg_wt=1.0, gamma=None,
                            degree=3):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
            
        # pipeline to scale -> logistic
        if neg_wt == False:
            wts = None
        else:
            wts = {0: neg_wt, 1:1.0}
        
        if kernel_name=='rbf':
            from sklearn.metrics.pairwise import rbf_kernel
            K_train = rbf_kernel(self.X_train, self.X_train, gamma=gamma)
            K_test = rbf_kernel(self.X_test, self.X_train, gamma=gamma)
        elif kernel_name=='poly':
            from sklearn.metrics.pairwise import polynomial_kernel
            K_train = polynomial_kernel(self.X_train, self.X_train,
                                        degree=degree)
            K_train = polynomial_kernel(self.X_test, self.X_train,
                                        degree=degree)
        elif kernel_name=='sigmoid':
            from sklearn.metrics.pairwise import sigmoid_kernel
            K_train = sigmoid_kernel(self.X_train, self.X_train)
            K_test = sigmoid_kernel(self.X_train, self.X_train)
            
#        log_reg_pipe = Pipeline([('log_reg_kernel', LogisticRegressionCV(
#                                         class_weight=wts,
#                                         solver='newton-cg'))])
        
        log_reg_pipe = Pipeline([('scaler', StandardScaler()),
                                 ('log_reg_kernel', LogisticRegressionCV(
                                         class_weight=wts,
                                         solver='newton-cg'))])
        
        # LRCV finds optimum C value, L2 penalty
        log_reg_pipe.fit(K_train, self.y_train)
            
        # Get test accuracy
        C = log_reg_pipe._final_estimator.C_
        tr_scr = log_reg_pipe.score(K_train, self.y_train)
        
        print('The best logistic training score is %0.3f'
              %tr_scr)
        
        te_scr = log_reg_pipe.score(K_test, self.y_test)
        print('Kernel logistic testing score: %0.3f'
              %te_scr)
        
        self.log_reg_kernel = log_reg_pipe    
    
    # Train SVM classification
    def fit_svc(self, neg_wt=1.0, kernel_name='rbf'):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics.pairwise import rbf_kernel
        
        # pipeline to scale -> SVC
        if neg_wt == False:
            wts = None
        else:
            wts = {0: neg_wt, 1:1.0}
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svc', SVC(kernel=rbf_kernel, gamma='auto',
                                        probability=True,
                                        class_weight=wts))])
        
        # cross-validate several parameters
        parameters = [
            {'svc__C':[0.1, 1.0, 10.0, 100.0, 1000.0]}
            ]
        
        svc_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svc_cv.fit(self.X_train, self.y_train)
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svc_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        print("The best SVC parameters are %s with a training score of %0.3f"
              % (svc_cv.best_params_, svc_cv.best_score_))
        
        te_scr = sv_pipe.score(self.X_test, self.y_test)
        print('SVC testing score: %0.3f' %te_scr)
        self.svc = sv_pipe
       
    # Train random forest classification
    def fit_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        
        mdl_rf = RandomForestClassifier(n_estimators=20, random_state=985)
        mdl_rf.fit(self.X_train, self.y_train)
        
        tr_scr = mdl_rf.score(self.X_train, self.y_train)
        print("The random forest training score is %0.3f"
              %tr_scr)
        
        te_scr = mdl_rf.score(self.X_test, self.y_test)
        print("The random forest testing score is %0.3f"
              %te_scr)
        self.rf = mdl_rf
    
    # fit decision tree classification
    def fit_dt(self):
        from sklearn import tree
        
        mdl_dt = tree.DecisionTreeClassifier(max_depth=3, random_state=985)
        mdl_dt.fit(self.X_train, self.y_train)
        
        tr_scr = mdl_dt.score(self.X_train, self.y_train)
        print("The decision tree training score is %0.3f"
              %tr_scr)
        
        te_scr = mdl_dt.score(self.X_test, self.y_test)
        print("The decision tree testing score is %0.3f"
              %te_scr)
        
        # tree.plot_tree(mdl_dt)
        self.dt = mdl_dt
        
    def fit_ensemble(self):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import GradientBoostingClassifier

        
        # mdl_en = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), 
        #                             n_estimators=100)
        
        # mdl_en = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=1),
        #                             n_estimators=20, random_state=985)
        
        # mdl_en = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
        #                                     max_depth=1, random_state=985)
        
        mdl_en = RandomForestClassifier(n_estimators=20, random_state=985)
        
        print('Current ensembling method is', str(type(mdl_en)).split('.')[-1])
        
        mdl_en.fit(self.X_train, self.y_train)
        
        tr_scr = mdl_en.score(self.X_train, self.y_train)
        print("The ensemble training score is %0.3f"
              %tr_scr)
        
        te_scr = mdl_en.score(self.X_test, self.y_test)
        print("The ensemble testing score is %0.3f"
              %te_scr)
        self.ensemble_model = mdl_en
            
    def fit_gpr_mean_fcn(self, kernel_name):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        import sklearn.gaussian_process.kernels as krn
        
        import numpy as np
        n_vars = self.X.shape[1]
        
        if kernel_name=='rbf_ard':
            kernel_obj = 1.0 * krn.RBF(np.ones(n_vars))
        elif kernel_name=='rbf_iso':
            kernel_obj = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel_obj = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel_obj = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel_obj = 1.0 * krn.Matern(
                    length_scale=np.ones(n_vars), 
                    nu=1.5)
        
        # fitting linear fits normalized X on y (all in lin_reg pipeline)
        self.fit_linear()
        prior_y = self.lin_reg.predict(self.X)
        
        y_residual = self.y.to_numpy(dtype=float) - prior_y
        
        # pipeline to scale -> GPR
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpr', GaussianProcessRegressor(kernel=kernel_obj,
                                                 random_state=985,
                                                 n_restarts_optimizer=10))
                ])
    
        gp_pipe.fit(self.X, y_residual)
        
        self.gpr_residual = gp_pipe
        
    def predict_gpr_mean_fcn(self, X_test):
        
        # pipeline scales X, then gives mean function
        prior_y = self.lin_reg.predict(X_test)
        
        # pipeline scales X, then gives prediction of residuals
        y_gp, gp_std = self.gpr_residual.predict(X_test, return_std=True)
        y_pred = prior_y.ravel() + y_gp
        return y_pred, gp_std
    
    # Train GP regression
    def fit_gpr(self, kernel_name, noise_bound=None):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        import sklearn.gaussian_process.kernels as krn
        
        import numpy as np
        n_vars = self.X.shape[1]
        
        if kernel_name=='rbf_ard':
            kernel_base = 1.0 * krn.RBF(np.ones(n_vars))
        elif kernel_name=='rbf_iso':
            kernel_base = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel_base = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel_base = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel_base = 1.0 * krn.Matern(
                    length_scale=np.ones(n_vars), 
                    nu=1.5)
            
        if noise_bound is None:
            noise_bound = (1e-8, 1e2)
            
        kernel_obj = kernel_base + krn.WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bound)
        
        # pipeline to scale -> GPR
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpr', GaussianProcessRegressor(kernel=kernel_obj,
                                                 random_state=985,
                                                 n_restarts_optimizer=10))
                ])
    
        gp_pipe.fit(self.X, self.y)
        
        self.gpr = gp_pipe
        
    # Train GP regression
    def fit_het_gpr(self, kernel_name, mdl_var, noise_bound=None):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        import sklearn.gaussian_process.kernels as krn
        from sklearn.model_selection import GridSearchCV
        
        import numpy as np
        n_vars = self.X.shape[1]
        
        if kernel_name=='rbf_ard':
            kernel_base = 1.0 * krn.RBF(np.ones(n_vars))
        elif kernel_name=='rbf_iso':
            kernel_base = krn.ConstantKernel(1e-3, (1e-8, 1e3)) * krn.RBF(
                1.0, length_scale_bounds=(1e-8, 1e8))
        elif kernel_name=='rq':
            kernel_base = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel_base = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel_base = 1.0 * krn.Matern(
                    length_scale=np.ones(n_vars), 
                    nu=1.5)
        
        if noise_bound is None:
            noise_bound = (1e-8, 1e8)
            
        # kernel_obj = (kernel_base + 
        #           1.0*
        #           krn.Heteroscedastic_Variance(X_fit=X_fit, 
        #                                        dual_coef=dual_coef, 
        #                                        rbf_gamma=rbf_gamma))
        
        kernel_obj = kernel_base + krn.WhiteKernel(noise_level=0.01, 
                                                    noise_level_bounds=noise_bound)
        # kernel_obj = kernel_base
        
        # # pass in secondary model, predict diagonal het-noise
        n = self.X.shape[0]
        var_est = mdl_var.kr.predict(self.X)
        a_inv_cs_diag = np.exp(var_est.ravel())/n
        
        # breakpoint()
            
        # pipeline to scale -> GPR
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpr', GaussianProcessRegressor(kernel=kernel_obj,
                                                 alpha = a_inv_cs_diag,
                                                 random_state=985,
                                                 n_restarts_optimizer=10))
                ])
        
        # cross-validate several parameters
        # parameters = {'gpr__alpha':np.logspace(-1,7,9)}
        
        # breakpoint()
        # gpr_cv = GridSearchCV(gp_pipe, param_grid=parameters)
        # gpr_cv.fit(self.X, self.y)
        
        # # set pipeline to use CV params
        # gp_pipe.set_params(**gpr_cv.best_params_)
        # gp_pipe.fit(self.X, self.y)
        
        # print("The best GPR parameters are %s"
        #       % (gpr_cv.best_params_))
    
        gp_pipe.fit(self.X, self.y)
        
        self.gpr_het = gp_pipe
        
    def fit_kernel_ridge(self, kernel_name='rbf'):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        
        
        
        kr_pipe = Pipeline([('scaler', StandardScaler()),
                             ('kr', KernelRidge(kernel=kernel_name))])
        
        '''
        from scipy.stats import loguniform
        from sklearn.model_selection import RandomizedSearchCV
        param_distributions = {
            "kr__alpha": loguniform(1e-3, 1e3),
            "kr__gamma": loguniform(1e-5, 1e3),
        }
        kr_cv = RandomizedSearchCV(
            kr_pipe,
            param_distributions=param_distributions,
            n_iter=500,
            random_state=985,
        )
        kr_cv.fit(self.X, self.y)
        '''
        
        # cross-validate several parameters
        from numpy import logspace
        parameters = [
            {'kr__alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
              'kr__gamma':logspace(-5, 3, 9)}
            ]
        
        kr_cv = GridSearchCV(kr_pipe, param_grid=parameters)
        kr_cv.fit(self.X, self.y)
        
        # set pipeline to use CV params
        print("The best kernel ridge parameters are %s"
              % (kr_cv.best_params_))
        kr_pipe.set_params(**kr_cv.best_params_)
        kr_pipe.fit(self.X, self.y)
        
        self.kr = kr_pipe
    
    
    # Train SVM regression
    def fit_svr(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        from numpy import logspace
        from sklearn.metrics.pairwise import rbf_kernel
        
        # pipeline to scale -> SVR
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svr', SVR(kernel=rbf_kernel))])
        
        # cross-validate several parameters
        parameters = [
            {'svr__C':[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
             'svr__epsilon':[0.01, 0.1, 1.0],
             'svr__gamma':logspace(-3, 3, 7)}
            ]
        
        svr_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svr_cv.fit(self.X_train, self.y_train)
        
        print("The best SVR parameters are %s"
              % (svr_cv.best_params_))
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svr_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        self.svr = sv_pipe
        
    # Train regular ridge regression
    def fit_ols_ridge(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        from numpy import logspace
        
        or_pipe = Pipeline([('scaler', StandardScaler()),
                             ('o_ridge', RidgeCV(alphas=logspace(-2, 2, 5)))]
            )
        
        or_pipe.fit(self.X_train, self.y_train)
        
        self.o_ridge = or_pipe
        
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
    
    def doe_mse_loocv(self, bound_df, design_filter=False):
        """Return max single point corresponding to the MSE LOOCV weighted metric.

        Parameters
        ----------
        bound_df: Dataframe of variables and their upper/lower bound
        design_filter: Boolean determining whether or not to manually filter out
        designs for the sake of constructability of bearings (currently manual for TFP)
    
        Returns
        -------
        Array of n_pts corresponding to max (MSE-LOOCV) metric
        """
        import numpy as np
        n_var = bound_df.shape[1]
        
        min_list = [val for val in bound_df.loc['min']]
        max_list = [val for val in bound_df.loc['max']]
        
        # find the maximum that the function will be in the domain
        from scipy.optimize import basinhopping
        
        # bounds for design space
        bnds = tuple((min_list[i], max_list[i]) for i in range(n_var))
        
        x0 = np.array([np.random.uniform(min_list[i], max_list[i])
                          for i in range(n_var)])
        
        # use basin hopping to avoid local minima
        minimizer_kwargs={'args':(bound_df), 'bounds':bnds}
        res = basinhopping(self.fn_LOOCV_error, x0, minimizer_kwargs=minimizer_kwargs,
                           niter=100, seed=985)
        
        x_keep = res.x
        
        # if T_m > 4, zeta must be > 0.15
        if design_filter == True:
            var_list = bound_df.columns.tolist()
            Tm_idx = var_list.index('T_ratio')
            zeta_idx = var_list.index('zeta_e')
            
            # remove designs that have high period but low damping
            x_keep = x_keep[~((x_keep[Tm_idx] > 4) & 
                              (x_keep[zeta_idx] < 0.2))]
            
        # if failed to keep any point, try the rejection sampler
        if x_keep.shape[0] < 1:
            print('MSE point not suitable, using rejection sampler...')
            rej_samp_list = self.doe_rejection_sampler(5, 0.5, bound_df)
            x_keep = rej_samp_list[np.random.choice(rej_samp_list.shape[0], 1, replace=False),:]
            
        return x_keep
    
    def doe_rejection_sampler(self, n_pts, pr, bound_df, rho_wt=1.0, design_filter=False):
        """Return points sampled proportional to a custom DoE metric using 
        rejection sampling.

        Parameters
        ----------
        n_pts : Scalar of number of desired sampled points
        pr: For targeted MSE, provide the probability contour that the DoE weights
        MSE against
        bound_df: Dataframe of variables and their upper/lower bound
        design_filter: Boolean determining whether or not to manually filter out
        designs for the sake of constructability of bearings (currently manual for TFP)
    
        Returns
        -------
        Array of n_pts sampled points proportional to (LOOCV) metric
        """
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
        minimizer_kwargs={'args':(bound_df, rho_wt), 'bounds':bnds}
        res = basinhopping(self.fn_LOOCV_error, x0, minimizer_kwargs=minimizer_kwargs,
                           niter=100, seed=985)
    
        # get maximum of function for rejection variable scaling
        c_max = -res.fun
        
        # sample rejection variable
        u_var = np.array([np.random.uniform(0.0, c_max, n_max)]).T
        
        '''
        # evaluate the function at x
        fx = -self.fn_tmse(x_var, pr, bound_df).T
        x_keep = x_var[u_var.ravel() < fx,:]
        '''
        
        # evaluate the function at x
        fx = np.apply_along_axis(self.fn_LOOCV_error, 1, x_var, bound_df, rho_wt)*-1
        x_keep = x_var[u_var.ravel() < fx.ravel(),:]
        
        
        # TODO: temporary manual design filters
        # if T_m > 4, zeta must be > 0.15
        if design_filter == True:
            var_list = bound_df.columns.tolist()
            Tm_idx = var_list.index('T_ratio')
            zeta_idx = var_list.index('zeta_e')
            
            # remove designs that have high period but low damping
            x_keep = x_keep[~((x_keep[:,Tm_idx] > 4) & 
                              (x_keep[:,zeta_idx] < 0.2))]
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(13, 6))
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["mathtext.fontset"] = "dejavuserif"
        # import matplotlib as mpl
        # label_size = 16
        # mpl.rcParams['xtick.labelsize'] = label_size
        # mpl.rcParams['ytick.labelsize'] = label_size
        # choice = np.random.choice(x_keep.shape[0], x_keep.shape[0], replace=False)
        # plt.scatter(x_keep[choice,0], x_keep[choice,1])
        
        # return a choice of n_pts random qualifying points
        if x_keep.shape[0] < n_pts:
            return(x_keep[np.random.choice(x_keep.shape[0], n_pts, replace=True),:])
        else:
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
        
        gp_obj = self.gpr._final_estimator
        # weight is from Lyu / Picheny
        from numpy import exp, log
        pi = 3.14159
        Wx = 1/((2*pi*(fs2))**0.5) * exp((-1/2)*((fmu - T)**2/(fs2)))
        log_Wx = -log((2*pi*(fs2))**0.5) * (-1/2)*((fmu - T)**2/(fs2))
        
        return(-fs2 * Wx)
    
    
    def fn_LOOCV_error(self, X_cand, bound_df, rho=1.0):
        """Return point LOOCV error approximation for a given point
        
        Parameters
        ----------
        X_cand: np.array of candidate point
    
        Returns
        -------
        x_next: e^cv^2 from Yi & Taflanidis paper
        """
        
        import numpy as np
        import pandas as pd
        
        # get trained GP info
        gp_obj = self.gpr._final_estimator
        X_train = gp_obj.X_train_
        
        # isotropic RBF, probably
        # use decaying distance metric
        point = np.array([np.asarray(X_cand)])
        from scipy.spatial.distance import cdist
        
        if len(gp_obj.kernel_.theta < 4):
            lengthscale = gp_obj.kernel_.theta[1]
        # ARD RBF, probably
        else:
            lengthscale = gp_obj.kernel_.theta[1:5]
        
        dist_list = cdist(point/lengthscale, X_train/lengthscale).flatten()
        
        # calculate LOOCV error of training set (Kyprioti)
        L = gp_obj.L_
        K_mat = L @ L.T
        alpha_ = gp_obj.alpha_.flatten()
        K_inv_diag = np.linalg.inv(K_mat).diagonal()
        e_cv_sq = np.divide(alpha_, K_inv_diag)**2
        
        '''
        # smoothing function, exponentially decaying
        gamma = np.exp(-dist_list**2)
        
        # aggregate LOOCV_error and distance
        numerator = np.sum(np.multiply(gamma, e_cv_sq))
        denominator = np.sum(gamma)
        '''
        
        
        # smoothing function, exponentially decaying
        # aggregate LOOCV_error and distance
        # use LSE
        from scipy.special import logsumexp
        denominator = logsumexp(-dist_list**2)
        numerator = logsumexp(-dist_list**2, b=e_cv_sq)
        e_cv2_cand = np.exp(numerator - denominator)
        
        '''
        import warnings
        
        warnings.filterwarnings('error')
        
        try:
            e_cv2_cand = numerator/denominator
        except RuntimeWarning:
            from scipy.special import logsumexp
            # use log-sum-exp trick to avoid underflows
            test_den = logsumexp(-dist_list**2)
            test_num = logsumexp(-dist_list**2, b=e_cv_sq)
            e_cv2_cand = np.exp(test_num - test_den)
            
        # reset warning so regular warnings bypass
        warnings.resetwarnings()
        '''
        
        # find predictive variance
        if X_cand.ndim == 1:
            X_df = np.array([X_cand])
            
        X_df = pd.DataFrame(X_df, columns=bound_df.columns)
        
        fmu, fs1 = self.gpr.predict(X_df, return_std=True)
        fs2 = fs1**2
        
        # return negative for minimization purposes
        return(-fs2 * e_cv2_cand ** rho)
    
    '''
    def fn_LOOCV_IMSE(self, X_cand, bound_df):
        """Return point LOOCV error approximation for a given point
        
        Parameters
        ----------
        X_cand: np.array of candidate point
    
        Returns
        -------
        x_next: e^cv^2 from Yi & Taflanidis paper
        """
        
        import numpy as np
        import pandas as pd
        
        min_list = [val for val in bound_df.loc['min']]
        max_list = [val for val in bound_df.loc['max']]
        
        res = 20
        
        xx, yy, uu, vv = np.meshgrid(np.linspace(min_list[0], max_list[0],
                                                 res),
                                     np.linspace(min_list[1], max_list[1],
                                                 res),
                                     np.linspace(min_list[2], max_list[2],
                                                 res),
                                     np.linspace(min_list[3], max_list[3],
                                                 res))
                                     
        X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                             'RI':yy.ravel(),
                             'T_ratio':uu.ravel(),
                             'zeta_e':vv.ravel()})
        
        fmu_q, fs1_q = self.gpr.predict(X_space, return_std=True)
        fs2_q = fs1_q**2
        
        # get trained GP info
        gp_obj = self.gpr._final_estimator
        X_train = gp_obj.X_train_
        lengthscale = gp_obj.kernel_.theta[1]
        
        # calculate LOOCV error of training set (Kyprioti)
        L = gp_obj.L_
        K_mat = L @ L.T
        alpha_ = gp_obj.alpha_.flatten()
        K_inv_diag = np.linalg.inv(K_mat).diagonal()
        e_cv_sq = np.divide(alpha_, K_inv_diag)**2
        
        # use decaying distance metric
        point = np.array([np.asarray(X_cand)])
        from scipy.spatial.distance import cdist
        dist_list = cdist(point/lengthscale, X_train/lengthscale).flatten()
        
        # smoothing function, exponentially decaying
        # aggregate LOOCV_error and distance
        # use LSE
        from scipy.special import logsumexp
        denominator = logsumexp(-dist_list**2)
        numerator = logsumexp(-dist_list**2, b=e_cv_sq)
        e_cv2_cand = np.exp(numerator - denominator)
    '''

