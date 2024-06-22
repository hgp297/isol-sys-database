#%% support classes for c(x) approximation (heteroscedastic nugget variance)
class Heteroscedastic_Variance(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """Heteroscedastic nugget variance kernel.

    Uses a pre-computed predictor function to express an approximation for the
    heteroscedastic nugget variance. Creates a diagonal matrix.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.
    """

    def __init__(self, X_fit, dual_coef, rbf_gamma):
        self.X_fit = X_fit
        self.dual_coef = dual_coef
        self.rbf_gamma = rbf_gamma
        
    @property
    def hyperparameter_gamma(self):
        return Hyperparameter('rbf_gamma', 'numeric', 'fixed')
    
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            from numpy import diag, exp
            from sklearn.metrics.pairwise import rbf_kernel
            K_pr = rbf_kernel(X, self.X_fit, gamma=self.rbf_gamma)
            y_pr = K_pr @ self.dual_coef
            K = diag(exp(y_pr.ravel()))
            # K = self.noise_level * np.eye(_num_samples(X))
            if eval_gradient:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        from numpy import exp
        from sklearn.metrics.pairwise import rbf_kernel
        K_pr = rbf_kernel(X, self.X_fit, gamma=self.rbf_gamma)
        y_pr = K_pr @ self.dual_coef
        return exp(y_pr.ravel())
        # return np.full(
        #     _num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype
        # )
        

    def __repr__(self):
        return "Custom heteroscedastic nugget variance"