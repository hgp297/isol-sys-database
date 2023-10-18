############################################################################
#               Generalized bearing object

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: March 2023

# Description:  Object stores all information about about a bearing. Primarily
# used for plotting and troubleshooting

# Open issues:  [0] 

############################################################################

# class takes a pandas Series (row) and creates an object that holds
# design information

class Bearing:
        
    # import attributes as building characteristics from pd.Series
    def __init__(self, design):
        for key, value in design.items():
            setattr(self, key, value)
            
    def get_backbone(self):
        bearing_type = self.isolator_system
        
        # design displacement
        D_m = self.D_m
        
        if bearing_type == 'LRB':
            Q_L = self.Q * self.W / self.N_lb
            k_M = self.k_e * self.W / self.N_lb
        else:
            Q_L = self.Q
            k_M = self.k_e
        
        # from Q, zeta, and T_m
        k_2 = (k_M*D_m - Q_L)/D_m
        alpha = 1/self.k_ratio
        
        # yielding force
        k_1 = self.k_ratio * k_2
        D_y = Q_L/(k_1 - k_2)
        
        f_y = Q_L + k_2*D_y
        
        import numpy as np
        dt = 0.001
        pi = 3.14159
        t = np.arange(0, 2, dt)
        omega = 2*pi/2
        
        '''
        # system
        p = D_m*omega**2*np.sin(omega*t)
        m = 1
        omega_n = (k_M/m)**0.5
        c = 2*self.zeta_e*m*omega_n
        '''
        
        u = D_m*np.sin(omega*t)
        
        # Initial state determination
        numPoints = len(u)
        fs = np.zeros(numPoints)
        kT = np.zeros(numPoints)
        up = np.zeros(numPoints)
        q = np.zeros(numPoints)
        up[0] = 0.0
        q[0] = 0.0
        
        for i in range(numPoints-1):
            (up[i+1], fs[i+1], 
             q[i+1], kT[i+1]) = bilin_state_determination(k_1, alpha, u[i+1], 
                                                          up[i], q[i], f_y)
        
        return(u, fs)
        
        
def NL_newmark_SDOF(m, K1, K2, c, p, dt, tol, fy, method,
                    u0=0.0, v0=0.0, fs0=0.0, up0=0.0, q0=0.0):
    
    import numpy as np
    
    alpha = K2/K1

    numPoints = len(p)
    
    if method == 'constant':
        beta = 1/4
        gamma = 1/2
    else:
        beta = 1/6
        gamma = 1/2

    # Initial state determination
    fs = np.zeros(numPoints)
    kT = np.zeros(numPoints)
    fs[0] = fs0
    kT[0] = K1

    # Initial conditions
    u = np.zeros(numPoints)
    v = np.zeros(numPoints)
    a = np.zeros(numPoints)
    u[0] = u0
    v[0] = v0
    a[0] = (p[0]-c*v[0]-fs[0])/m

    # Constants
    a1 = m/(beta*dt**2) + gamma/(beta*dt)*c
    a2 = m/(beta*dt) + (gamma/beta-1)*c
    a3 = (1/(2*beta)-1)*m + dt*(gamma/(2*beta)-1)*c

    kTHat = np.zeros(numPoints)
    pHat = np.zeros(numPoints)
    RHat = np.zeros(numPoints)
    du = np.zeros(numPoints)
    up = np.zeros(numPoints)
    q = np.zeros(numPoints)
    up[0] = up0
    q[0] = q0
    
    # Loop through time i in forcing function to calculate response at i+1
    for i in range(numPoints-1):
        u[i+1] = u[i]
        fs[i+1] = fs[i]
        kT[i+1] = kT[i]
        pHat[i+1] = p[i+1] + a1*u[i] + a2*v[i] + a3*a[i]
        RHat[i+1] = pHat[i+1] - fs[i+1] - a1*u[i+1]
        diff = abs(RHat[i+1])
        
        j = 0
        while j < 10000 and diff > tol:
            kTHat[i+1] = kT[i+1] + a1
            du[i+1] = RHat[i+1]/kTHat[i+1]
            u[i+1] = u[i+1] + du[i+1]
            
            (up_next, fs_next, 
             q_next, kT_next) = bilin_state_determination(K1, alpha, u[i+1], 
                                                          up[i], q[i], fy)
            
            up[i+1] = up_next
            fs[i+1] = fs_next
            q[i+1] = q_next
            kT[i+1] = kT_next
            
            '''
            # state determination
            F_trial = K1*(u[i+1]-up[i])
            xi_trial = F_trial - q[i]
            f_trial = abs(xi_trial)-fy

            if f_trial < 0: #elastic
                fs[i+1] = F_trial
                up[i+1] = up[i]
                q[i+1] = q[i]
                kT[i+1] = K1
            else: #yielding
                dgamma = f_trial/(K1+H)
                dup[i+1] = dgamma*np.sign(xi_trial)
                fs[i+1] = F_trial - K1*(dup[i+1])
                up[i+1] = up[i] + dup[i+1]
                q[i+1] = q[i] + dgamma*H*np.sign(xi_trial)
                kT[i+1] = K2
            '''
            
            v[i+1] = (gamma/(beta*dt)*(u[i+1] - u[i]) + 
                      (1-gamma/beta)*v[i] + dt*(1-gamma/(2*beta)))
            a[i+1] = ((u[i+1] - u[i])/(beta*dt**2) - 
                      v[i]/(beta*dt) - (1/(2*beta)-1)*a[i])
            RHat[i+1] = pHat[i+1] - fs[i+1] - a1*u[i+1]
            diff = abs(RHat[i+1])
            j += 1
            
    return u, v, a, fs

# bilinear state determination algo (from CE 223)
def bilin_state_determination(K, alpha, u_tr, up_n, q_n, Fy):
    K2 = alpha*K
    H = alpha*K/(1 - alpha)
    
    # state determination
    F_trial = K*(u_tr - up_n)
    xi_trial = F_trial - q_n
    f_trial = abs(xi_trial)-Fy
    
    from numpy import sign
    
    if f_trial < 0: # elastic
        fs_next = F_trial
        up_next = up_n
        q_next = q_n
        kT_next = K
    else:
        dgamma = f_trial/(K + H)
        dup_next = dgamma*sign(xi_trial)
        fs_next = F_trial - K*(dup_next)
        up_next = up_n + dup_next
        q_next = q_n + dgamma*H*sign(xi_trial)
        kT_next = K2
        
    return up_next, fs_next, q_next, kT_next