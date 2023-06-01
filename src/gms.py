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
    # TODO: T_fb shouldn't be fixed
    T_fb = 0.6
    t_lower = T_fb
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
    elig_freq = gm_concise[gm_concise[' Lowest Useable Frequency (Hz)'] < freq_min]

    # List unique earthquakes
    uniq_EQs = pd.unique(elig_freq[' Earthquake Name'])
    final_GM = None

    # Select earthquakes that are least severely scaled
    for earthquake in uniq_EQs:
        match_eqs = elig_freq[elig_freq[' Earthquake Name'] == earthquake]
        match_eqs['scale_difference'] = abs(match_eqs['sf_average_spectral']-1.0)
        # take 3 least scaled ones
        least_scaled = match_eqs.sort_values(by=['scale_difference']).iloc[:3] 

        if final_GM is None:
            GM_headers = list(match_eqs.columns)
            final_GM = pd.DataFrame(columns=GM_headers)
        
        final_GM = pd.concat([least_scaled,final_GM], sort=False)
        final_GM[' Horizontal-1 Acc. Filename'] = final_GM[
            ' Horizontal-1 Acc. Filename'].str.strip()

    final_GM = final_GM.reset_index()
    final_GM = final_GM.drop(columns=['index', 'scale_difference'])
    final_GM.columns = ['RSN', 'sf_average_spectral', 
                        'earthquake_name', 'lowest_frequency', 'filename']
    
    # select random GM from the list
    from random import randrange
    
    sf = 40.0
    
    # keep selecting until we have a GM with <20
    # assumes that there is a ground motion that can reach the S_1 without being scaled >20.0
    tries = 0
    while sf > 20.0:
        ind = randrange(len(final_GM.index))
        filename = str(final_GM['filename'][ind]) # ground motion name
        gm_name = filename.replace('.AT2', '') # remove extension from file name
        sf = float(final_GM['sf_average_spectral'][ind])  # scale factor used
        
        tries += 1
        if tries > len(final_GM.index):
            print('No GM found.')
            break
    
    return(gm_name, sf, target_average)
    
    
# a, b = scale_ground_motion()