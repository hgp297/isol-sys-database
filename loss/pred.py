############################################################################
#               Prediction object for ML models and plotting

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) 

############################################################################
import pandas as pd
import numpy as np

class Prediction:
    
    # sets up the problem by grabbing the x covariates
    def __init__(self, data):
        self._raw_data = data
        self.k = len(data)
        self.X = data[['gapRatio', 'RI', 'Tm', 'zetaM']]
        
    # sets up prediction variable
    def set_outcome(self, outcome_var):
        self.y = self._raw_data[[outcome_var]]
        
    # if classification is done, plot the predictions
    def plot_classification(self, mdl_clf, xvar='gapRatio', yvar='RI', 
                            contour_pr=0.5, contour_pred=0):
        import matplotlib.pyplot as plt
        
        xx = self.xx
        yy = self.yy
        if 'gpc' in mdl_clf.named_steps.keys():
            Z = mdl_clf.predict_proba(self.X_plot)[:, 1]
        elif 'log_reg_kernel' in mdl_clf.named_steps.keys():
            Z = mdl_clf.decision_function(self.K_pr)
        else:
            Z = mdl_clf.decision_function(self.X_plot)
            
        Z = Z.reshape(xx.shape)
        
        plt.figure()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(),
                    yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap=plt.cm.PuOr_r,
        )
        plt_density = 200
        
        if 'gpc' in mdl_clf.named_steps.keys():
            plt.contour(xx, yy, Z, levels=[contour_pr],
                        linewidths=2, linestyles="dashed")
        else:
            plt.contour(xx, yy, Z, levels=[contour_pred],
                        linewidths=2, linestyles="dashed")
        
        plt.scatter(self.X_train[xvar][:plt_density],
                    self.X_train[yvar][:plt_density],
                    s=30, c=self.y_train[:plt_density],
                    cmap=plt.cm.Paired, edgecolors="k")
        plt.xlabel(xvar, fontsize=18)
        plt.ylabel(yvar, fontsize=18)
        if 'svc' in mdl_clf.named_steps.keys():
            plt.title('Classification (SVC)')
        elif 'log_reg' in mdl_clf.named_steps.keys():
            plt.title('Classification (logistic)')
        elif 'log_reg_kernel' in mdl_clf.named_steps.keys():
            plt.title('Classification (kernel logistic)')
        elif 'gpc' in mdl_clf.named_steps.keys():
            plt.title('Classification (GP)')
        plt.show()
        
    # make a generalized 2D plotting grid, defaulted to gap and Ry
    # grid is based on the bounds of input data
    def make_2D_plotting_space(self, res, x_var='gapRatio', y_var='RI',
                               x_bounds=None, y_bounds=None):
        
        if x_bounds == None:
            x_min = min(self.X[x_var])
            x_max = max(self.X[x_var])
        else:
            x_min = x_bounds[0]
            x_max = x_bounds[1]
        if y_bounds == None:
            y_min = min(self.X[y_var])
            y_max = max(self.X[y_var])
        else:
            y_min = y_bounds[0]
            y_max = y_bounds[1]
        xx, yy = np.meshgrid(np.linspace(x_min,
                                         x_max,
                                         res),
                             np.linspace(y_min,
                                         y_max,
                                         res))

        # don't code like this
        if (x_var=='gapRatio') and (y_var=='RI'):
            third_var = 'Tm'
            fourth_var = 'zetaM'
            
        if (x_var=='gapRatio') and (y_var=='Tm'):
            third_var = 'RI'
            fourth_var = 'zetaM'
            
        if (x_var=='gapRatio') and (y_var=='zetaM'):
            third_var = 'RI'
            fourth_var = 'Tm'
            
        if (x_var=='Tm') and (y_var=='zetaM'):
            third_var = 'gapRatio'
            fourth_var = 'RI'
            
        if (x_var=='Tm') and (y_var=='gapRatio'):
            third_var = 'zetaM'
            fourth_var = 'RI'
            
        if (x_var=='RI') and (y_var=='gapRatio'):
            third_var = 'Tm'
            fourth_var = 'zetaM'
            
        if (x_var=='zetaM') and (y_var=='gapRatio'):
            third_var = 'Tm'
            fourth_var = 'RI'
            
        if (x_var=='RI') and (y_var=='Tm'):
            third_var = 'gapRatio'
            fourth_var = 'zetaM'
           
        self.xx = xx
        self.yy = yy
        X_pl = pd.DataFrame({x_var:xx.ravel(),
                             y_var:yy.ravel(),
                             third_var:np.repeat(self.X[third_var].median(),
                                                 res*res),
                             fourth_var:np.repeat(self.X[fourth_var].median(), 
                                                  res*res)})
        self.X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]
                             
        return(self.X_plot)
        
    def make_design_space(self, res):
        xx, yy, uu, vv = np.meshgrid(np.linspace(0.3, 1.8,
                                                 res),
                                     np.linspace(min(self.X['RI']),
                                                 max(self.X['RI']),
                                                 res),
                                     np.linspace(min(self.X['Tm']),
                                                 max(self.X['Tm']),
                                                 res),
                                     np.linspace(min(self.X['zetaM']),
                                                 max(self.X['zetaM']),
                                                 res))
                                     
        self.X_space = pd.DataFrame({'gapRatio':xx.ravel(),
                             'RI':yy.ravel(),
                             'Tm':uu.ravel(),
                             'zetaM':vv.ravel()})
    
        return(self.X_space)
        
    # train test split to be done before any learning 
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
 
###############################################################################
    # Classification models
###############################################################################       
    
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
    
        gp_pipe.fit(self.X_train, self.y_train)
        tr_scr = gp_pipe.score(self.X_train, self.y_train)
        print("The GP training score is %0.2f"
              %tr_scr)
        
        te_scr = gp_pipe.score(self.X_test, self.y_test)
        print('GP testing score: %0.2f' %te_scr)
        self.gpc = gp_pipe
        
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
        scaler.fit(self.X_train)
        X_tr = scaler.transform(X)
        K_star = mdl_gpc.kernel_(mdl_gpc.X_train_, X_tr)  # K_star =k(x_star)
        f_star = K_star.T.dot(mdl_gpc.y_train_ - mdl_gpc.pi_)  # Line 4
        v = solve(mdl_gpc.L_, mdl_gpc.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = mdl_gpc.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
        
        # # Line 7:
        # # Approximate \int log(z) * N(z | f_star, var_f_star)
        # # Approximation is due to Williams & Barber, "Bayesian Classification
        # # with Gaussian Processes", Appendix A: Approximate the logistic
        # # sigmoid by a linear combination of 5 error functions.
        # # For information on how this integral can be computed see
        # # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
        # LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
        # COEFS = np.array(
        #     [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
        # )[:, np.newaxis]
        
        
        # alpha = 1 / (2 * var_f_star)
        # gamma = LAMBDAS * f_star
        # integrals = (
        #     np.sqrt(np.pi / alpha)
        #     * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2)))
        #     / (2 * np.sqrt(var_f_star * 2 * np.pi))
        # )
        # pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()
        
        return(f_star, var_f_star)
        
    # Train logistic classification
    def fit_log_reg(self, neg_wt=1.0):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        
        # pipeline to scale -> logistic
        wts = {0: neg_wt, 1:1.0}
        log_reg_pipe = Pipeline([('scaler', StandardScaler()),
                                 ('log_reg', LogisticRegressionCV(
                                         class_weight=wts))])
        
        # LRCV finds optimum C value, L2 penalty
        log_reg_pipe.fit(self.X_train, self.y_train)
        
        # Get test accuracy
        C = log_reg_pipe._final_estimator.C_
        tr_scr = log_reg_pipe.score(self.X_train, self.y_train)
        
        print('The best logistic C value is %f with a training score of %0.2f'
              % (C, tr_scr))
        
        te_scr = log_reg_pipe.score(self.X_test, self.y_test)
        print('Logistic testing score: %0.2f'
              %te_scr)
        
        self.log_reg = log_reg_pipe
        
    # Train SVM classification
    def fit_svc(self, neg_wt=1.0, kernel_name='rbf'):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # pipeline to scale -> SVC
        wts = {0: neg_wt, 1:1.0}
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svc', SVC(kernel=kernel_name, gamma='auto',
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
        
        print("The best SVC parameters are %s with a training score of %0.2f"
              % (svc_cv.best_params_, svc_cv.best_score_))
        
        te_scr = sv_pipe.score(self.X_test, self.y_test)
        print('SVC testing score: %0.2f' %te_scr)
        self.svc = sv_pipe
        
    def get_kernel(self, X_pr, kernel_name='rbf', gamma=100.0,
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
        
    # TODO: cross validate gamma, get other kernels working
    def fit_kernel_logistic(self, kernel_name='rbf', neg_wt=1.0, gamma=None,
                            degree=3):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
            
        # pipeline to scale -> logistic
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
        
        print('The best logistic C value is %f with a training score of %0.2f'
              % (C, tr_scr))
        
        te_scr = log_reg_pipe.score(K_test, self.y_test)
        print('Kernel logistic testing score: %0.2f'
              %te_scr)
        
        self.log_reg_kernel = log_reg_pipe   
        
    

###############################################################################
    # Regression models
###############################################################################
    # TODO: conjoin this with classif so one object can handle both impact cases
    
    # Train SVM regression
    def fit_svr(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        
        # pipeline to scale -> SVR
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svr', SVR(kernel='rbf'))])
        
        # cross-validate several parameters
        parameters = [
            {'svr__C':[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
             'svr__epsilon':[0.01, 0.1, 1.0],
             'svr__gamma':np.logspace(-3, 3, 7)}
            ]
        
        svr_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svr_cv.fit(self.X_train, self.y_train)
        
        print("The best SVR parameters are %s"
              % (svr_cv.best_params_))
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svr_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        self.svr = sv_pipe
        
    # Train kernel ridge regression
    def fit_kernel_ridge(self, kernel_name='rbf'):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        
        kr_pipe = Pipeline([('scaler', StandardScaler()),
                             ('kr', KernelRidge(kernel=kernel_name))])
        
        # cross-validate several parameters
        parameters = [
            {'kr__alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
             'kr__gamma':np.logspace(-3, 3, 7)}
            ]
        
        kr_cv = GridSearchCV(kr_pipe, param_grid=parameters)
        kr_cv.fit(self.X_train, self.y_train)
        
        # set pipeline to use CV params
        print("The best kernel ridge parameters are %s"
              % (kr_cv.best_params_))
        kr_pipe.set_params(**kr_cv.best_params_)
        kr_pipe.fit(self.X_train, self.y_train)
        
        self.kr = kr_pipe
        
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
    
        gp_pipe.fit(self.X_train, self.y_train)
        
        self.gpr = gp_pipe
        
    # Train regular ridge regression
    def fit_ols_ridge(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        
        or_pipe = Pipeline([('scaler', StandardScaler()),
                             ('o_ridge', RidgeCV(alphas=np.logspace(-2, 2, 5)))]
            )
        
        or_pipe.fit(self.X_train, self.y_train)
        
        self.o_ridge = or_pipe
        
###############################################################################
    # Full prediction models
###############################################################################

# two ways of doing this
        
        # 1) predict impact first (binary), then fit the impact predictions 
        # with the impact-only SVR and likewise with non-impacts. This creates
        # two tiers of predictions that are relatively flat (impact dominated)
        # 2) using expectations, get probabilities of collapse and weigh the
        # two (cost|impact) regressions with Pr(impact). Creates smooth
        # predictions that are somewhat moderate
        
def predict_DV(X, impact_pred_mdl, hit_loss_mdl, miss_loss_mdl,
               outcome='cost_50%'):
        
#        # get points that are predicted impact from full dataset
#        preds_imp = impact_pred_mdl.svc.predict(self.X)
#        df_imp = self.X[preds_imp == 1]
    
        # get probability of impact
        if 'log_reg_kernel' in impact_pred_mdl.named_steps.keys():
            probs_imp = impact_pred_mdl.predict_proba(impact_pred_mdl.K_pr)
        else:
            probs_imp = impact_pred_mdl.predict_proba(X)
    
        miss_prob = probs_imp[:,0]
        hit_prob = probs_imp[:,1]
        
        # weight with probability of collapse
        # E[Loss] = (impact loss)*Pr(impact) + (no impact loss)*Pr(no impact)
        # run SVR_hit model on this dataset
        outcome_str = outcome+'_pred'
        expected_DV_hit = pd.DataFrame(
                {outcome_str:np.multiply(
                        hit_loss_mdl.predict(X),
                        hit_prob)})
                
#        # get points that are predicted no impact from full dataset
#        df_mss = self.X[preds_imp == 0]
        
        # run SVR_miss model on this dataset
        expected_DV_miss = pd.DataFrame(
                {outcome_str:np.multiply(
                        miss_loss_mdl.predict(X),
                        miss_prob)})
        
        expected_DV = expected_DV_hit + expected_DV_miss
        
        # self.median_loss_pred = pd.concat([loss_pred_hit,loss_pred_miss], 
        #                                   axis=0).sort_index(ascending=True)
        
        return(expected_DV)
    
#%% Calculate upfront cost of data
# TODO: use PACT to get the replacement cost of these components
# TODO: include the deckings/slabs for more realistic initial costs

def get_steel_coefs(df, steel_per_unit=1.25, W=3037.5, Ws=2227.5):
    col_str = df['col']
    beam_str = df['beam']
    rbeam_str = df['roofBeam']
    
    col_wt = np.array([float(member.split('X',1)[1]) for member in col_str])
    beam_wt = np.array([float(member.split('X',1)[1]) for member in beam_str])
    rbeam_wt = np.array([float(member.split('X',1)[1]) for member in rbeam_str])
    
    # find true steel costs
    n_frames = 4
    n_cols = 12
    L_col = 39.0 #ft
    
    n_beam_per_frame = 6
    L_beam = 30.0 #ft
    
    n_roof_per_frame = 3
    L_roof = 30.0 #ft
    
    bldg_wt = ((L_col * n_cols)*col_wt +
               (L_beam * n_beam_per_frame * n_frames)*beam_wt +
               (L_roof * n_roof_per_frame * n_frames)*rbeam_wt
               )
    
    steel_cost = steel_per_unit*bldg_wt
    
    # find design base shear as a feature
    pi = 3.14159
    g = 386.4
    kM = (1/g)*(2*pi/df['Tm'])**2
    S1 = 1.017
    Dm = g*S1*df['Tm']/(4*pi**2*df['Bm'])
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*df['zetaM'])
    Vs = np.array(Vst/df['RI']).reshape(-1,1)
    
    # linear regress cost as f(base shear)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X=Vs, y=steel_cost)
    return({'coef':reg.coef_, 'intercept':reg.intercept_})
    
# TODO: add economy of scale for land
# TODO: investigate upfront cost's influence by Tm
def calc_upfront_cost(X_query, steel_coefs,
                      land_cost_per_sqft=2837/(3.28**2),
                      W=3037.5, Ws=2227.5):
    
    from scipy.interpolate import interp1d
    zeta_ref = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    Bm_ref = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    interp_f = interp1d(zeta_ref, Bm_ref)
    Bm = interp_f(X_query['zetaM'])
    
    # calculate moat gap
    pi = 3.14159
    g = 386.4
    S1 = 1.017
    SaTm = S1/X_query['Tm']
    moat_gap = X_query['gapRatio'] * (g*(SaTm/Bm)*X_query['Tm']**2)/(4*pi**2)
    
    # calculate design base shear
    kM = (1/g)*(2*pi/X_query['Tm'])**2
    Dm = g*S1*X_query['Tm']/(4*pi**2*Bm)
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*X_query['zetaM'])
    Vs = np.array(Vst/X_query['RI']).reshape(-1,1)
    
    steel_cost = np.array(steel_coefs['intercept'] +
                          steel_coefs['coef']*Vs).ravel()
    # land_area = 2*(90.0*12.0)*moat_gap - moat_gap**2
    land_area = (90.0*12.0 + moat_gap)**2
    land_cost = land_cost_per_sqft/144.0 * land_area
    
    return(steel_cost + land_cost)