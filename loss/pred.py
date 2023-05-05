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
    def plot_classification(self, mdl_clf, xvar='gapRatio', yvar='RI'):
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
            
        plt_density = 100
        
        if 'gpc' in mdl_clf.named_steps.keys():
            plt.contour(xx, yy, Z, levels=[0.5],
                        linewidths=2, linestyles="dashed")
        else:
            plt.contour(xx, yy, Z, levels=[0],
                        linewidths=2, linestyles="dashed")
        plt.scatter(self.X_train[xvar][:plt_density],
                    self.X_train[yvar][:plt_density],
                    s=30, c=self.y_train[:plt_density],
                    cmap=plt.cm.Paired, edgecolors="k")
        plt.xlabel(xvar)
        plt.ylabel(yvar)
        if 'svc' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (SVC)')
        elif 'log_reg' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (logistic)')
        elif 'log_reg_kernel' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (kernel logistic)')
        elif 'gpc' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (GP)')
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
        xx, yy, uu, vv = np.meshgrid(np.linspace(0.8, 2.2,
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
    def fit_gpc(self, kernel_name):
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