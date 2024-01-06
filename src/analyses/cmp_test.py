############################################################################
#               Components testing file

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2023

# Description:  Working file to find component contents of structure

# Open issues:  (1) 

############################################################################
import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30

# and import pelicun classes and methods
from pelicun.assessment import Assessment

# import warnings
# warnings.filterwarnings('ignore')


#%% SDC function

def get_SDC(run_data):
    Sm1 = run_data.S_1
    Sd1 = Sm1*2/3
    if Sd1 < 0.135:
        cmp_name = 'fema_nqe_cmp_cat_ab.csv'
    elif Sd1 < 0.2:
        cmp_name = 'fema_nqe_cmp_cat_c.csv'
    else:
        cmp_name = 'fema_nqe_cmp_cat_def.csv'
    return(cmp_name)


#%%
# p90 low situations - calculator (values replaced in sheet)
'''
from scipy.stats import lognorm, norm
from scipy.optimize import curve_fit
f = lambda x,mu,sigma: lognorm(sigma,mu).cdf(x)
fn = lambda x, mu, sigma: norm(mu, sigma).cdf(x)

quantile_data = [6, 16.2, 27]
theta_n, beta_n = curve_fit(fn, np.log(quantile_data), [0.1, 0.5, 0.9])[0]
print(theta_n)
print(beta_n)
xx_pr = np.arange(0.001, 1.5, 0.001)
xx_pr_log = np.log(xx_pr)
p = fn(xx_pr_log, theta_n, beta_n)

import matplotlib.pyplot as plt
plt.close('all')

fig = plt.figure(figsize=(13, 10))
plt.plot(xx_pr, p)
plt.grid(True)
'''

#%% data prep function

# returns SDC-custom mean, std, and metadata of components
def nqe_sheets(run_data, nqe_dir='../../resource/loss/'):
    import pandas as pd
    sheet_name = get_SDC(run_data)
    
    nqe_data = pd.read_csv(nqe_dir + sheet_name)
    nqe_data.set_index('cmp', inplace=True)
    nqe_data = nqe_data.replace({'All Zero': 0}, regex=True)
    nqe_data = nqe_data.replace({'2 Points = 0': 0}, regex=True)
    nqe_data = nqe_data.replace({np.nan: 0})
    nqe_data['directional'] = nqe_data['directional'].replace(
        {'YES': True, 'NO': False})

    nqe_meta = nqe_data[[c for c in nqe_data if not (
        c.endswith('mean') or c.endswith('std'))]]
    nqe_mean = nqe_data[[c for c in nqe_data if c.endswith('mean')]]
    nqe_std = nqe_data[[c for c in nqe_data if c.endswith('std')]].apply(
        pd.to_numeric, errors='coerce')
    
    # unit conversion

    # goal: convert nqe sheet from FEMA units to PBEE units
    # also change PACT block division from FEMA to PBEE

    # this section should not be set on a slice
    # i will ignore
    pd.options.mode.chained_assignment = None  # default='warn'

    # convert chillers to single units (assumes small 75 ton chillers)
    # also assumes chillers only components using TN
    mask = nqe_meta['unit'].str.contains('TN')
    nqe_mean.loc[mask,:] = nqe_mean.loc[mask,:].div(75)
    nqe_meta.loc[mask, 'PACT_block'] = 'EA 1'
    nqe_meta = nqe_meta.replace({'TN': 'EA'})

    # convert AHUs to single units (assumes small 4000 cfm AHUs)
    # also assumes AHUs only components using CF
    mask = nqe_meta['unit'].str.contains('CF')
    nqe_mean.loc[mask,:] = nqe_mean.loc[mask,:].div(4000)
    nqe_meta.loc[mask, 'PACT_block'] = 'EA 1'
    nqe_meta = nqe_meta.replace({'CF': 'EA'})

    # convert large transformers from WT to EA (assumes 250e3 W = 250 kV = 1 EA)
    mask = nqe_meta['unit'].str.contains('WT')
    nqe_mean.loc[mask,:] = nqe_mean.loc[mask,:].div(250e3)

    # change all transformers block division to EA
    mask = nqe_meta['PACT_name'].str.contains('Transformer')
    nqe_meta.loc[mask, 'PACT_block'] = 'EA 1'
    nqe_meta = nqe_meta.replace({'WT': 'EA'})


    # distribution panels already in EA, but block division needs to change
    mask = nqe_meta['PACT_name'].str.contains('Distribution Panel')
    nqe_meta.loc[mask, 'PACT_block'] = 'EA 1'

    # convert low voltage switchgear to single units (assumes 225 AP per unit)
    # also assumes switchgear only components using AP
    mask = nqe_meta['unit'].str.contains('AP')
    nqe_mean.loc[mask,:] = nqe_mean.loc[mask,:].div(225)
    nqe_meta.loc[mask, 'PACT_block'] = 'EA 1'
    nqe_meta = nqe_meta.replace({'AP': 'EA'})

    # convert diesel generator to single units (assumes 250 kV per unit)
    mask = nqe_meta['PACT_name'].str.contains('Diesel generator')
    nqe_mean.loc[mask,:] = nqe_mean.loc[mask,:].div(250)
    nqe_meta.loc[mask, 'PACT_block'] = 'EA 1'
    nqe_meta.loc[mask, 'unit'] = 'EA'
    
    return nqe_meta, nqe_mean, nqe_std
    
#%% structural components

def get_structural_cmp_MF(run_info, metadata):
    import pandas as pd
    
    cmp_strct = pd.DataFrame(columns=['Component', 'Units', 'Location', 'Direction',
                              'Theta_0', 'Theta_1', 'Family', 
                              'Blocks', 'Comment'])
    
    n_bays = run_info['num_bays']
    n_stories = run_info['num_stories']
    
    # ft
    L_bay = run_info['L_bay']
    
    n_col_base = (n_bays+1)**2
    
    # TODO: store data as pickle to avoid this having to convert str to datatype
    from ast import literal_eval
    all_beams = literal_eval(run_info['beam'])
    all_cols = literal_eval(run_info['column'])
    
    # column base plates
    n_col_base = (n_bays+1)**2
    base_col_wt = float(all_cols[0].split('X',1)[1])
    if base_col_wt < 150.0:
        cur_cmp = 'B.10.31.011a'
    elif base_col_wt > 300.0:
        cur_cmp = 'B.10.31.011c'
    else:
        cur_cmp = 'B.10.31.011b'
        
    cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', '1', '1,2',
                         n_col_base, np.nan, np.nan,
                         n_col_base, metadata[cur_cmp]['Description']]], 
                                        columns=cmp_strct.columns), cmp_strct])
                                           
    # bolted shear tab gravity, assume 1 per every 10 ft span in one direction
    cur_cmp = 'B.10.31.001'
    num_grav_tabs_per_frame = (L_bay//10 - 1) * n_bays # girders
    num_frames = (n_bays+1)*2
    
    # non-MF connections
    n_cxn_per_reg_frame = (n_bays-1)*2
    n_reg_frames = (n_bays-1)*2
    
    # gravity girders + shear tabs from non-MF connections
    num_grav_tabs = ((num_grav_tabs_per_frame * num_frames) + 
                     (n_cxn_per_reg_frame * n_reg_frames))
    
    cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', 'all', '0',
                             num_grav_tabs, np.nan, np.nan,
                             num_grav_tabs, metadata[cur_cmp][
                                 'Description']]], 
                                        columns=cmp_strct.columns), cmp_strct])
    
    # assume one splice after every 3 floors
    n_splice = n_stories // (3+1)
    
    if n_splice > 0:
        for splice in range(n_splice):
            splice_floor_ind = (splice+1)*3
            splice_col_wt = float(all_cols[splice_floor_ind].split('X',1)[1])
            
            if splice_col_wt < 150.0:
                cur_cmp = 'B.10.31.021a'
            elif splice_col_wt > 300.0:
                cur_cmp = 'B.10.31.021c'
            else:
                cur_cmp = 'B.10.31.021b'
                
            cmp_strct = pd.concat(
                [pd.DataFrame([[cur_cmp, 'ea', splice_floor_ind+1, '1,2',
                                n_col_base, np.nan, np.nan, 
                                n_col_base, metadata[cur_cmp]['Description']]], 
                              columns=cmp_strct.columns), cmp_strct])               
    
    for fl_ind, beam_str in enumerate(all_beams):
        
        beam_depth = float(beam_str.split('X',1)[0].split('W',1)[1])
        
        # beam-one-side connections
        if beam_depth <= 27.0:
            cur_cmp = 'B.10.35.021'
        else:
            cur_cmp = 'B.10.35.022'
            
        # quantity is always 8 because 4 corner columns, 2 directions 
        cmp_strct = pd.concat(
            [pd.DataFrame([[cur_cmp, 'ea', fl_ind+1, '1,2',
                            8, np.nan, np.nan,
                            8, metadata[cur_cmp]['Description']]], 
                          columns=cmp_strct.columns), cmp_strct])
                                               
        # beam-both-side connections
        if beam_depth <= 27.0:
            cur_cmp = 'B.10.35.031'
        else:
            cur_cmp = 'B.10.35.032'
        
        # assumes 2 frames x 2 directions = 4
        n_cxn_interior = (n_bays-1)*4
        cmp_strct = pd.concat(
            [pd.DataFrame([[cur_cmp, 'ea', fl_ind+1, '1,2',
                            n_cxn_interior, np.nan, np.nan,
                            n_cxn_interior, metadata[cur_cmp]['Description']]], 
                          columns=cmp_strct.columns), cmp_strct])
    return(cmp_strct)

def get_structural_cmp_CBF(run_info, metadata, 
                           brace_dir='../../resource/'):
    
    import pandas as pd
    cmp_strct = pd.DataFrame(columns=['Component', 'Units', 'Location', 'Direction',
                              'Theta_0', 'Theta_1', 'Family', 
                              'Blocks', 'Comment'])
    
    brace_db = pd.read_csv(brace_dir+'braceShapes.csv',
                           index_col=None, header=0) 
    
    n_bays = run_info['num_bays']
    n_stories = run_info['num_stories']
    n_braced = max(int(round(n_bays/2.25)), 1)
    
    # ft
    L_bay = run_info['L_bay']
    
    n_col_base = (n_bays+1)**2
    
    from ast import literal_eval
    all_cols = literal_eval(run_info['column'])
    all_braces = literal_eval(run_info['brace'])
    
    # column base plates
    n_col_base = (n_bays+1)**2
    base_col_wt = float(all_cols[0].split('X',1)[1])
    if base_col_wt < 150.0:
        cur_cmp = 'B.10.31.011a'
    elif base_col_wt > 300.0:
        cur_cmp = 'B.10.31.011c'
    else:
        cur_cmp = 'B.10.31.011b'
        
    cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', '1', '1,2',
                         n_col_base, np.nan, np.nan,
                         n_col_base, metadata[cur_cmp]['Description']]], 
                                        columns=cmp_strct.columns), cmp_strct])
                                           
    # bolted shear tab gravity, assume 1 per every 10 ft span in one direction
    cur_cmp = 'B.10.31.001'
    num_grav_tabs_per_frame = (L_bay//10 - 1) * n_bays # girders
    num_frames = (n_bays+1)*2
    
    # shear tab at every column joints
    n_cxn_per_reg_frame = (n_bays)*2
    
    # gravity girders + shear tabs from non-MF connections
    num_grav_tabs = ((num_grav_tabs_per_frame * num_frames) + 
                     (n_cxn_per_reg_frame * num_frames))
    
    cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', 'all', '0',
                             num_grav_tabs, np.nan, np.nan,
                             num_grav_tabs, metadata[cur_cmp][
                                 'Description']]], 
                                        columns=cmp_strct.columns), cmp_strct])
    
    
    # assume one splice after every 3 floors
    n_splice = n_stories // (3+1)
    
    if n_splice > 0:
        for splice in range(n_splice):
            splice_floor_ind = (splice+1)*3
            splice_col_wt = float(all_cols[splice_floor_ind].split('X',1)[1])
            
            if splice_col_wt < 150.0:
                cur_cmp = 'B.10.31.021a'
            elif splice_col_wt > 300.0:
                cur_cmp = 'B.10.31.021c'
            else:
                cur_cmp = 'B.10.31.021b'
                
            cmp_strct = pd.concat(
                [pd.DataFrame([[cur_cmp, 'ea', splice_floor_ind+1, '1,2',
                                n_col_base, np.nan, np.nan, 
                                n_col_base, metadata[cur_cmp]['Description']]], 
                              columns=cmp_strct.columns), cmp_strct])
            
    for fl_ind, brace_str in enumerate(all_braces):
        
        cur_brace = brace_db.loc[brace_db['AISC_Manual_Label'] == brace_str]
        brace_wt = float(cur_brace['W'])
        
        if brace_wt < 40.0:
            cur_cmp = 'B.10.33.011a'
        elif brace_wt > 100.0:
            cur_cmp = 'B.10.33.011c'
        else:
            cur_cmp = 'B.10.33.011b'
        
        # n_bay_braced, two frames, two directions
        n_brace_cmp_bays = n_braced*2*2
        cmp_strct = pd.concat(
            [pd.DataFrame([[cur_cmp, 'ea', fl_ind+1, '1,2',
                            n_brace_cmp_bays, np.nan, np.nan,
                            n_brace_cmp_bays, metadata[cur_cmp]['Description']]], 
                          columns=cmp_strct.columns), cmp_strct])
        
    return(cmp_strct)
#%% nqe function

# estimate mean qty for floor area
# has not adjusted for 'quantity' column
def floor_mean(area_usage, mean_data):
    return mean_data * area_usage

# get std for floor (sum of lognormals)
def floor_std(area_usage, std_data, floor_mean):
    # get boolean if things are present, then multiply with stdev
    import numpy as np
    has_stuff = floor_mean.copy()
    has_stuff[has_stuff != 0] = 1
    
    # variance per floor
    var_present = np.square(std_data.values * has_stuff.values)
    
    # var_xy = var_x + var_y; std = sqrt(var)
    std_cmp = np.sqrt(np.sum(var_present, axis=1))
    
    return pd.Series(std_cmp, index=std_data.index)
    
def floor_qty_estimate(area_usage, mean_data, std_data, meta_data):
    fl_cmp_by_usage = floor_mean(area_usage, mean_data)
    fl_std = floor_std(area_usage, std_data, fl_cmp_by_usage)
    
    # sum across all usage and adjust for base quantity, then round up
    fl_cmp_qty = fl_cmp_by_usage.sum(axis=1) * meta_data['quantity']
    return(fl_cmp_by_usage, fl_cmp_qty, fl_std)

# function to remove (cmp, dir, loc) duplicates. assumes that only
# theta_0 and theta_1 changes
def remove_dupes(dupe_df):
    dupe_cmps = dupe_df.cmp.unique()
    
    clean_df = pd.DataFrame()
    import numpy as np
    for cmp in dupe_cmps:
        cmp_df = dupe_df[dupe_df.cmp == cmp]
        sum_means = cmp_df.Theta_0.sum()
        sum_blocks = cmp_df.Blocks.sum()
        srss_std = np.sqrt(np.square(cmp_df.Theta_1).sum())
        
        new_row = cmp_df.iloc[[0]].copy()
        new_row.Theta_0 = sum_means
        new_row.Blocks = sum_blocks
        new_row.Theta_1 = srss_std
        
        clean_df = pd.concat([clean_df, new_row], axis=0)
    return(clean_df)

def bldg_wide_cmp(roof_df):
    roof_cmps = roof_df.Component.unique()
    clean_df = pd.DataFrame()
    import numpy as np
    for cmp in roof_cmps:
        cmp_df = roof_df[roof_df.Component == cmp]
        sum_means = cmp_df.Theta_0.sum()
        sum_blocks = cmp_df.Blocks.sum()
        srss_std = np.sqrt(np.square(cmp_df.Theta_1).sum())
        
        new_row = cmp_df.iloc[[0]].copy()
        if (new_row['Comment'].str.contains('Elevator').any()):
            new_row.Location = 1
        else:
            new_row.Location = 'roof'
        new_row.Theta_0 = sum_means
        
        # blocks may be over-rounded here
        new_row.Blocks = sum_blocks
        new_row.Theta_1 = srss_std
        
        clean_df = pd.concat([clean_df, new_row], axis=0)
    return(clean_df)

def normative_quantity_estimation(run_info, usage, nqe_mean, nqe_std, nqe_meta,
                                  P58_metadata):
    floor_area = run_info.L_bldg**2 # sq ft
    
    cmp_marginal = pd.DataFrame()
    
    fema_units = nqe_meta['unit']
    nqe_meta[['pact_unit', 'pact_block_qty']] = nqe_meta['PACT_block'].str.split(
        ' ', n=1, expand=True)
    
    if not nqe_meta['pact_unit'].equals(fema_units):
        print('units not equal, check before block division')
    
    nqe_meta['pact_block_qty'] = pd.to_numeric(nqe_meta['pact_block_qty'])
    pact_units = fema_units.replace({'SF': 'ft2',
                                     'LF': 'ft',
                                     'EA': 'ea'})
    # perform floor estimation
    for fl, fl_usage in enumerate(bldg_usage):
        area_usage = np.array(fl_usage)*floor_area
        
        fl_cmp_by_cat, fl_cmp_total, fl_cmp_std = floor_qty_estimate(
            area_usage, nqe_mean, nqe_std, nqe_meta)
        
        fl_cmp_total.name = 'Theta_0'
        fl_cmp_std.name = 'Theta_1'
        
        loc_series = pd.Series([fl+1]).repeat(
            len(fl_cmp_total)).set_axis(fl_cmp_total.index)
        
        dir_map = {True:'1,2', False:'0'}
        dir_series = nqe_meta.directional.map(dir_map)
        
        has_stdev = fl_cmp_std != 0
        has_stdev.name = 'Family'
        family_map = {True:'lognormal', False:''}
        family_series = has_stdev.map(family_map)
        
        block_series = fl_cmp_total // nqe_meta['pact_block_qty']
        block_series.name = 'Blocks'
        
        fl_cmp_df = pd.concat([pact_units, loc_series, 
                               dir_series, fl_cmp_total, fl_cmp_std,
                               family_series, block_series, nqe_meta.PACT_name], 
                              axis=1)
        
        fl_cmp_df = fl_cmp_df[fl_cmp_df.Theta_0 != 0]
        
        fl_cmp_df = fl_cmp_df.reset_index()
        
        # combine duplicates, then remove duplicates from floor's list
        dupes = fl_cmp_df[fl_cmp_df.duplicated(
            'cmp', keep=False)].sort_values('cmp')
        combined_dupe_rows = remove_dupes(dupes)
        fl_cmp_df = fl_cmp_df[~fl_cmp_df['cmp'].isin(combined_dupe_rows['cmp'])]
        
        cmp_marginal = pd.concat([cmp_marginal, fl_cmp_df, combined_dupe_rows], 
                                 axis=0, ignore_index=True)
    
    cmp_marginal.columns = ['Component', 'Units', 'Location',
                            'Direction','Theta_0', 'Theta_1',
                            'Family', 'Blocks', 'Comment']
    
    # hardcoded roof list
    roof_stuff = ['Chiller', 'Air Handling Unit', 'Cooling Tower', 'HVAC Fan',
               'Elevator', 'Distribution Panel', 'Diesel generator', 
               'Motor Control', 'Transformer', 'roof']
    roof_cmp = cmp_marginal[
        cmp_marginal['Comment'].str.contains('|'.join(roof_stuff))]
    
    combined_roof_rows = bldg_wide_cmp(roof_cmp)
    cmp_marginal = cmp_marginal[
        ~cmp_marginal['Component'].isin(combined_roof_rows['Component'])]
    
    cmp_marginal = pd.concat([cmp_marginal, combined_roof_rows], 
                             axis=0, ignore_index=True)
    
    # hardcoded no-block list
    no_block_stuff = ['Chiller', 'Cooling Tower', 'Motor Control', 'stair',
                      'Elevator', 'Raised Access Floor', 'Switchgear']
    mask = cmp_marginal['Comment'].str.contains('|'.join(no_block_stuff))
    cmp_marginal.loc[mask, 'Blocks'] = ''
    
    # total loss cmps
    replace_df = pd.DataFrame([['excessiveRID', 'ea' , 'all', '1,2', 
                                '1', '', '', '', 'Excessive residual drift'],
                               ['irreparable', 'ea', 0, '1', 
                                '1', '', '', '', 'Irreparable building'],
                               ['collapse', 'ea', 0, '1',
                                '1', '', '', '', 'Collapsed building']
                               ], columns=cmp_marginal.columns)
    
    nsc_cmp = pd.concat([cmp_marginal, replace_df])
    
    # structural components
    superstructure = run_info['superstructure_system']
    if superstructure == 'MF':
        structural_cmp = get_structural_cmp_MF(run_info, P58_metadata)
    else:
        structural_cmp = get_structural_cmp_CBF(run_info, P58_metadata)
        
    total_cmps = pd.concat([structural_cmp, nsc_cmp], ignore_index=True)

    return(total_cmps)



#%% main

# run info
data = pd.read_csv('../../data/structural_db_conv.csv')
cbf_run = data.iloc[0]
mf_run = data.iloc[-1]

# get database
# initialize, no printing outputs, offset fixed with current components
PAL = Assessment({
    "PrintLog": False, 
    "Seed": 985,
    "Verbose": False,
    "DemandOffset": {"PFA": 0, "PFV": 0}
})

# generate structural components and join with NSCs
P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')

cbf_floors = cbf_run.num_stories
cbf_area = cbf_run.L_bldg**2 # sq ft

# lab, health, ed, res, office, retail, warehouse, hotel
fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
bldg_usage = [fl_usage]*cbf_floors

area_usage = np.array(fl_usage)*cbf_area

nqe_meta, nqe_mean, nqe_std = nqe_sheets(cbf_run)
cmp_1 = normative_quantity_estimation(cbf_run, bldg_usage, nqe_mean, nqe_std, 
                                      nqe_meta, P58_metadata)

mf_floors = mf_run.num_stories

# lab, health, ed, res, office, retail, warehouse, hotel
fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
bldg_usage = [fl_usage]*mf_floors
nqe_meta, nqe_mean, nqe_std = nqe_sheets(mf_run)
cmp_2 = normative_quantity_estimation(mf_run, bldg_usage, nqe_mean, nqe_std, 
                                      nqe_meta, P58_metadata)
# TODO: keep record of total cmp (groupby?)