###############################################################################
#               Ground motion selector

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: April 2023

# Description:  Function takes the df of the design and first creates a spectrum
#               based on the S_1 and T_m values. It then scales all ground motions
#               and filters based on lowest usable freq

# Open issues:  (1) 

###############################################################################



def scale_ground_motion(input_df,
                        db_dir='../resource/ground_motions/gm_db.csv',
                        spec_dir='../resource/ground_motions/gm_spectra.csv'):
    
    import pandas as pd
    import numpy as np
    
    S_1 = input_df['S_1']
    T_m = input_df['T_m']
    
    # default='warn', ignore SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  
    
    gm_info = pd.read_csv(db_dir)
    unscaled_spectra = pd.read_csv(spec_dir)
    
    # info from building class
    S_s = 2.2815
    
    # Scale both Ss and S1
    # Create design spectrum
    
    T_short = S_1/S_s
    target_spectrum  = unscaled_spectra[['Period (sec)']]
    target_spectrum['Target pSa (g)'] = np.where(
        target_spectrum['Period (sec)'] < T_short, 
        S_s, S_1/target_spectrum['Period (sec)'])
    
    # calculate desired target spectrum average (0.2*Tm, 1.5*Tm)
    T_fb = input_df['T_fbe']
    t_lower = min(T_fb, 0.2*T_m)
    t_upper = 1.5*T_m

    # geometric mean from Eads et al. (2015)
    target_range = target_spectrum[
        target_spectrum['Period (sec)'].between(t_lower,t_upper)]['Target pSa (g)']
    target_average = target_range.prod()**(1/target_range.size)
    
    # get the spectrum average for the unscaled GM spectra
    # only concerned about H1 spectra
    H1s = unscaled_spectra.filter(regex=("-1 pSa \(g\)$"))
    us_range = H1s[target_spectrum['Period (sec)'].between(t_lower, t_upper)]
    us_average = us_range.prod()**(1/len(us_range.index))

    # determine scale factor to get unscaled to target
    scale_factor = target_average/us_average
    scale_factor = scale_factor.reset_index()
    scale_factor.columns = ['full_RSN', 'sf_average_spectral']
    
    # rename back to old convention and merge with previous dataframe
    scale_factor[' Record Sequence Number'] = scale_factor['full_RSN'].str.extract('(\d+)')
    scale_factor = scale_factor.astype({' Record Sequence Number': int})
    gm_info = pd.merge(gm_info,
        scale_factor, 
        on=' Record Sequence Number').drop(columns=['full_RSN'])
    
    # grab only relevant columns
    db_cols = [' Record Sequence Number',
               'sf_average_spectral',
               ' Earthquake Name',
               ' Lowest Useable Frequency (Hz)',
               ' Horizontal-1 Acc. Filename']
    gm_concise = gm_info[db_cols]

    # Filter by lowest usable frequency
    T_max = t_upper
    freq_min = 1/T_max
    gm_concise = gm_concise[gm_concise[' Lowest Useable Frequency (Hz)'] < freq_min]

    # List unique earthquakes
    uniq_EQs = pd.unique(gm_concise[' Earthquake Name'])
    final_GM = None

    import numpy as np
    
    # Select earthquakes that are least severely scaled
    # This section ensures no more than 3 motions per event
    for earthquake in uniq_EQs:
        match_eqs = gm_concise[gm_concise[' Earthquake Name'] == earthquake]
        match_eqs['scale_difference'] = abs(match_eqs['sf_average_spectral']-1.0)
        
        # take 3 random ones (shuffle then take)
        match_eqs = match_eqs.reindex(np.random.permutation(match_eqs.index))
        random_set = match_eqs.head(3)

        if final_GM is None:
            GM_headers = list(match_eqs.columns)
            final_GM = pd.DataFrame(columns=GM_headers)
        
        final_GM = pd.concat([random_set,final_GM], sort=False)
        final_GM[' Horizontal-1 Acc. Filename'] = final_GM[
            ' Horizontal-1 Acc. Filename'].str.strip()

    final_GM = final_GM.reset_index()
    final_GM = final_GM.drop(columns=['index', 'scale_difference'])
    final_GM.columns = ['RSN', 'sf_average_spectral', 
                        'earthquake_name', 'lowest_frequency', 'filename']
    
    # filter excessively scaled GMs
    final_GM = final_GM[final_GM['sf_average_spectral'] < 20.0]
    
    # select random GM from the list
    from random import randrange
    ind = randrange(len(final_GM.index))
    filename = str(final_GM['filename'].iloc[ind]) # ground motion name
    gm_name = filename.replace('.AT2', '') # remove extension from file name
    sf = float(final_GM['sf_average_spectral'].iloc[ind])  # scale factor used
    
    # ensure that selected GM does not fall below 90% of target spectrum in range
    import re
    rsn = re.search('(\d+)', gm_name).group(1)
    gm_unscaled_name = 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
    gm_spectrum = unscaled_spectra[['Period (sec)', gm_unscaled_name]]
    gm_spectrum.columns  = ['Period', 'Sa']
    gm_spectrum['scaled_Sa'] = gm_spectrum['Sa']*sf
    return(gm_name, sf, target_average)

def get_gm_ST(input_df, T_query):
    Tn, gm_A, gm_D, uddg = generate_spectrum(input_df)
    from numpy import interp
    Sa_query = interp(T_query, Tn, gm_A)
    return(Sa_query)
    
def get_ST(input_df, T_query, 
           db_dir='../resource/ground_motions/gm_db.csv',
           spec_dir='../resource/ground_motions/gm_spectra.csv'):

    import re
    import pandas as pd
    import numpy as np

    # load in sections of the sheet
    unscaled_spectra = pd.read_csv(spec_dir)
    
    GM_file = input_df['gm_selected']
    scale_factor = input_df['scale_factor']

    rsn = re.search('(\d+)', GM_file).group(1)
    gm_unscaled_name = 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
    gm_spectrum = unscaled_spectra[['Period (sec)', gm_unscaled_name]]
    gm_spectrum.columns  = ['Period', 'Sa']

    Sa_query_unscaled  = np.interp(T_query, gm_spectrum.Period, gm_spectrum.Sa)
    Sa_query = scale_factor*Sa_query_unscaled
    return(Sa_query)

def plot_spectrum(input_df,
                  spec_dir='../resource/ground_motions/gm_spectra.csv'):
    
    import pandas as pd

    # load in sections of the sheet
    unscaled_spectra = pd.read_csv(spec_dir)
    
    GM_name = input_df['gm_selected']

    import matplotlib.pyplot as plt
    import numpy as np
    
    S_1 = input_df['S_1']
    T_m = input_df['T_m']
    
    # info from building class
    S_s = 2.2815
    
    # default='warn', ignore SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  
    
    # Scale both Ss and S1
    # Create design spectrum
    
    T_short = S_1/S_s
    target_spectrum  = unscaled_spectra[['Period (sec)']]
    target_spectrum['Target_Sa'] = np.where(
        target_spectrum['Period (sec)'] < T_short, 
        S_s, S_1/target_spectrum['Period (sec)'])
    
    # calculate desired target spectrum average (0.2*Tm, 1.5*Tm)
    T_fb = input_df['T_fbe']
    t_lower = min(T_fb, 0.2*T_m)
    t_upper = 1.5*T_m
    
    # geometric mean from Eads et al. (2015)
    target_range = target_spectrum[
        target_spectrum['Period (sec)'].between(t_lower,t_upper)]['Target_Sa']
    target_average = target_range.prod()**(1/target_range.size)
    
    # generate true spectra for ground motion
    # scale factor applied internally, spectrum for damping of zeta_e
    import time
    t0 = time.time()
    Tn, gm_A, gm_D, uddg = generate_spectrum(input_df)
    tp = time.time() - t0
    print("Created spectrum in %.2f s" % tp)
    
    plt.figure()
    plt.plot(Tn, gm_A)
    plt.plot(target_spectrum['Period (sec)'], target_spectrum.Target_Sa)
    plt.axvline(t_lower, linestyle=':', color='red')
    plt.axvline(t_upper, linestyle=':', color='red')
    plt.axvline(T_m, linestyle='--', color='red')
    plt.axhline(target_average, linestyle=':', color='black')
    plt.title('Spectrum for '+GM_name)
    plt.xlabel(r'Period $T_n$ (s)')
    plt.ylabel(r'Spectral acceleration $Sa$ (g)')
    plt.xlim([0, 5])
    plt.grid(True)
    
    plt.figure()
    t_vec = np.linspace(0, 60.0, len(uddg))
    plt.plot(t_vec, uddg)
    plt.title('Ground acceleration '+GM_name)
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$\ddot{u}_g$ (g)')
    plt.grid(True)
    
    g = 386.4
    pi = 3.14159
    from numpy import interp
    
    # from ASCE Ch. 17, get damping multiplier
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    
    # from T_m, zeta_M, S_1
    B_m = interp(input_df['zeta_e'], zetaRef, BmRef)
    
    disp_target = target_spectrum.Target_Sa*g*target_spectrum['Period (sec)']**2/(4*pi**2*B_m)
    
    plt.figure()
    plt.plot(Tn, gm_D)
    plt.plot(target_spectrum['Period (sec)'], disp_target)
    plt.axvline(T_m, linestyle='--', color='red')
    plt.title('Displacement spectrum for '+GM_name)
    plt.xlabel(r'Period $T_n$ (s)')
    plt.ylabel(r'Displacement $D$ (in)')
    plt.ylim([0, 50])
    plt.xlim([0, 5])
    plt.grid(True)

def generate_spectrum(input_df,
                      gm_path = '../resource/ground_motions/PEERNGARecords_Unscaled/',
                      spec_dir = '../resource/ground_motions/'):
    
    gm_name = input_df['gm_selected']
    scale_factor = input_df['scale_factor']
    
    # call procedure to convert the ground-motion file
    from ReadRecord import ReadRecord
    dt, nPts = ReadRecord(gm_path+gm_name+'.AT2', gm_path+gm_name+'.g3')
   
    import pandas as pd
    import numpy as np
    
    # read g3 file and catch uneven columns
    df_uddg = pd.read_csv(gm_path+gm_name+'.g3', 
                          header=None, delim_whitespace=True)
    array_uddg = df_uddg.values.flatten()
    
    # scaled here, drop NaNs
    uddg = array_uddg[~np.isnan(array_uddg)]*scale_factor
    
    # Tn vector to match PEER
    spec_df = pd.read_csv(spec_dir+'period_range.csv',
                           names=['Tn'], header=None)
    zeta = input_df['zeta_e']
    
    # calculate damped frequency using frequency domain
    spec_df[['A', 'D']] = spec_df.apply(lambda row: 
                                        spectrum_frequency_domain(row, zeta, uddg, dt),
                                        axis='columns', result_type='expand')
    
    return spec_df['Tn'], spec_df['A'], spec_df['D'], uddg


# make function to parallellize spectrum
# uddg is in g
def spectrum_time_domain(df, zeta, uddg, dt):
    Tn = df['Tn']
    pi = 3.14159
    m = 1
    omega_n = 2*pi/Tn
    k = m*omega_n**2
    
    c = 2*m*omega_n*zeta
    g = 386.4
    p = -m*uddg*g
    
    u_caa, v_caa, a_caa = newmark_SDOF(m, k, c, p, dt, 0, 0, 'constant')
    
    A = max(abs(u_caa))*omega_n**2/g
    D = max(abs(u_caa))
    
    return(A, D)

# use frequency domain to calculate spectrum (faster)
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

# frequency domain spectrum analysis (CE 223)
def spectrum_frequency_domain(df, zeta, uddg, dt):
    Tn = df['Tn']
    pi = 3.14159
    m = 1
    omega_n = 2*pi/Tn
    
    g = 386.4
    nw = next_power_of_2(len(uddg))
    half_nw = int(nw/2)
    
    # first half of array is positive frequency (fft specification)
    # second half is negative
    dw = 2*pi/(dt*nw)
    import numpy as np
    omega = np.zeros(nw)
    for i in range(half_nw):
        omega[i] = i*dw
    for j in range(half_nw, nw):
        omega[j] = (-half_nw + (j - half_nw + 1))*dw
    
    # pad ground motion to make it cyclical
    uddg_pad = np.zeros(nw)
    uddg_pad[:len(uddg)] = uddg*g
    
    
    # perform Fourier transform of ground motion
    from scipy.fft import ifft, fft
    Uddgw = fft(-m*uddg_pad)
    
    # transfer function
    H = 1/(omega_n**2 - omega**2 + 2*zeta*1j*omega_n*omega)
    uw = -H*Uddgw
    
    # get result in time domain
    ut = np.real(ifft(uw))
    
    A = max(abs(ut))*omega_n**2/g
    D = max(abs(ut))
    
    return(A, D)

def newmark_SDOF(m, k, c, p, dt, u0, v0, method):
    numPoints = len(p)
    
    if method == 'constant':
        beta = 1/4
        gamma = 1/2
    else:
        beta = 1/6
        gamma = 1/2
        
    import numpy as np
    
    # Initial conditions
    u = np.zeros(numPoints)
    v = np.zeros(numPoints)
    a = np.zeros(numPoints)
    
    u[0] = u0
    v[0] = v0
    a[0] = (p[0]-c*v[0]-k*u[0])/m
    
    a1 = m/(beta*dt**2) + gamma/(beta*dt)*c
    a2 = m/(beta*dt) + (gamma/beta-1)*c
    a3 = (1/(2*beta)-1)*m + dt*(gamma/(2*beta)-1)*c
    k_hat = k + a1
    
    # Loop through time i in forcing function to calculate response at i+1
    for i in range(numPoints-1):
        p_hat = p[i+1] + a1*u[i] + a2*v[i] + a3*a[i]
        
        u[i+1] = p_hat/k_hat
        v[i+1] = (gamma/(beta*dt)*(u[i+1] - u[i]) + 
                  (1-gamma/beta)*v[i] + dt*(1-gamma/(2*beta)))
        a[i+1] = ((u[i+1] - u[i])/(beta*dt**2) - 
                  v[i]/(beta*dt) - (1/(2*beta)-1)*a[i])
        
    return u, v, a
# a, b = scale_ground_motion()