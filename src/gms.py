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
    
    return(gm_name, sf, target_average)

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
    
    import re
    import pandas as pd

    # load in sections of the sheet
    unscaled_spectra = pd.read_csv(spec_dir)
    
    GM_name = input_df['gm_selected']
    scale_factor = input_df['scale_factor']

    rsn = re.search('(\d+)', GM_name).group(1)
    gm_unscaled_name = 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
    gm_spectrum = unscaled_spectra[['Period (sec)', gm_unscaled_name]]
    gm_spectrum.columns  = ['Period', 'Sa']

    import matplotlib.pyplot as plt
    import numpy as np
    
    S_1 = input_df['S_1']
    T_m = input_df['T_m']
    
    # default='warn', ignore SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  
    
    # info from building class
    S_s = 2.2815
    
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
    
    plt.figure()
    plt.plot(gm_spectrum.Period, gm_spectrum.Sa*scale_factor)
    plt.plot(target_spectrum['Period (sec)'], target_spectrum.Target_Sa)
    plt.axvline(t_lower, linestyle=':', color='red')
    plt.axvline(t_upper, linestyle=':', color='red')
    plt.axvline(T_m, linestyle=':', color='red')
    plt.axhline(target_average, linestyle=':', color='black')
    plt.title('Spectrum for '+GM_name)
    plt.xlabel(r'Period $T_n$ (s)')
    plt.ylabel(r'Spectral acceleration $Sa$ (g)')
    plt.xlim([0, 5])
    plt.grid(True)
    
# a, b = scale_ground_motion()