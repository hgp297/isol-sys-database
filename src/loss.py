############################################################################
#               Generalized component object

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2024

# Description:  Object stores all information for building contents for Pelicun

# Open issues:  (1) 

############################################################################
## temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

# suppress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Loss_Analysis:
        
    # import attributes as building characteristics from pd.Series
    def __init__(self, edp_sheet):
        for key, value in edp_sheet.items():
            setattr(self, key, value)
            
    # SDC function
    def get_SDC(self):
        Sm1 = self.S_1
        Sd1 = Sm1*2/3
        if Sd1 < 0.135:
            cmp_name = 'fema_nqe_cmp_cat_ab.csv'
        elif Sd1 < 0.2:
            cmp_name = 'fema_nqe_cmp_cat_c.csv'
        else:
            cmp_name = 'fema_nqe_cmp_cat_def.csv'
        return(cmp_name)
            
    # returns SDC-custom mean, std, and metadata of components
    def nqe_sheets(self, nqe_dir='../resource/loss/'):
        import pandas as pd
        import numpy as np
        sheet_name = self.get_SDC()
        
        nqe_data = pd.read_csv(nqe_dir + sheet_name)
        nqe_data.set_index('cmp', inplace=True)
        nqe_data = nqe_data.replace({'All Zero': 0}, regex=True)
        nqe_data = nqe_data.replace({'2 Points = 0': 0}, regex=True)
        nqe_data = nqe_data.replace({np.nan: 0})
        nqe_data['directional'] = nqe_data['directional'].map(
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
        
        # Reduce block size for some excessive block count components
        # curtain walls
        mask = nqe_meta['PACT_name'].str.contains('Curtain Walls')
        nqe_meta.loc[mask, 'PACT_block'] = 'SF 500'
        
        # # steam piping
        # mask = nqe_meta['PACT_name'].str.contains('Steam Piping')
        # nqe_meta.loc[mask, 'PACT_block'] = 'LF 1000'
        
        # # waste piping
        # mask = nqe_meta['PACT_name'].str.contains('Waste Piping')
        # nqe_meta.loc[mask, 'PACT_block'] = 'LF 1000'
        
        # ceiling
        mask = nqe_meta['PACT_name'].str.contains('Raised Access Floor')
        nqe_meta.loc[mask, 'PACT_block'] = 'SF 1000'
        
        # ceiling
        mask = nqe_meta['PACT_name'].str.contains('Suspended Ceiling')
        nqe_meta.loc[mask, 'PACT_block'] = 'SF 1000'
        
        # Floor covering
        mask = nqe_meta['PACT_name'].str.contains('Generic Floor Covering')
        nqe_meta.loc[mask, 'PACT_block'] = 'SF 500'
        
        # Pendant lighting
        mask = nqe_meta['PACT_name'].str.contains('Independent Pendant Lighting')
        nqe_meta.loc[mask, 'PACT_block'] = 'EA 100'
        
        # Concrete tile roof
        mask = nqe_meta['PACT_name'].str.contains('Concrete tile roof')
        nqe_meta.loc[mask, 'PACT_block'] = 'SF 1000'
        
        # Bookcases
        mask = nqe_meta['PACT_name'].str.contains('Bookcase')
        nqe_meta.loc[mask, 'PACT_block'] = 'EA 10'
        
        self.meta_sheet = nqe_meta
        self.mean_sheet = nqe_mean
        self.std_sheet = nqe_std
    
    # structural components
    def get_structural_cmp_MF(self, metadata):
        import pandas as pd
        import numpy as np
        
        cmp_strct = pd.DataFrame(columns=['Component', 'Units', 'Location', 'Direction',
                                  'Theta_0', 'Theta_1', 'Family', 
                                  'Blocks', 'Comment'])
        
        n_bays = self.num_bays
        n_stories = self.num_stories
        
        # ft
        L_bay = self.L_bay
        
        n_col_base = (n_bays+1)**2
        
        # from ast import literal_eval
        # all_beams = literal_eval(self.beam)
        # all_cols = literal_eval(self.column)
        
        all_beams = self.beam
        all_cols = self.column
        
        # column base plates
        n_col_base = (n_bays+1)**2
        base_col_wt = float(all_cols[0].split('X',1)[1])
        if base_col_wt < 150.0:
            cur_cmp = 'B.10.31.011a'
        elif base_col_wt > 300.0:
            cur_cmp = 'B.10.31.011c'
        else:
            cur_cmp = 'B.10.31.011b'
            
        cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', '1', '0',
                             n_col_base, np.nan, np.nan,
                             n_col_base, metadata[cur_cmp]['Description']]], 
                                            columns=cmp_strct.columns), cmp_strct])
                                               
        # bolted shear tab gravity, assume 1 per every 10 ft span in one direction
        cur_cmp = 'B.10.31.001'
        num_grav_tabs_per_frame = (L_bay//10 - 1) * n_bays # girders
        n_side_connection = (n_bays*2)*2
        
        # non-MF connections
        n_cxn_per_reg_frame = (n_bays-1)*2 + 2
        n_reg_frames = (n_bays-1)*2
        
        # gravity girders + shear tabs from non-MF connections
        num_grav_tabs = ((num_grav_tabs_per_frame * n_side_connection) + 
                         (n_cxn_per_reg_frame * n_reg_frames))
        
        # use blocks of 10
        cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', 'all', '0',
                                 num_grav_tabs, np.nan, np.nan,
                                 num_grav_tabs//9, metadata[cur_cmp][
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

    def get_structural_cmp_CBF(self, metadata, 
                               brace_dir='../resource/'):
        
        import pandas as pd
        import numpy as np
        cmp_strct = pd.DataFrame(columns=['Component', 'Units', 'Location', 'Direction',
                                  'Theta_0', 'Theta_1', 'Family', 
                                  'Blocks', 'Comment'])
        
        brace_db = pd.read_csv(brace_dir+'braceShapes.csv',
                               index_col=None, header=0) 
        
        n_bays = self.num_bays
        n_stories = self.num_stories
        n_braced = max(int(round(n_bays/2.25)), 1)
        
        # ft
        L_bay = self.L_bay
        
        n_col_base = (n_bays+1)**2
        
        # from ast import literal_eval
        # all_cols = literal_eval(self.column)
        # all_braces = literal_eval(self.brace)
        
        all_cols = self.column
        all_braces = self.brace
        
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
        n_side_connection = (n_bays*2)*2
        
        # shear tab at every column joints
        n_cxn_per_reg_frame = (n_bays)*2
        
        # gravity girders + shear tabs from non-MF connections
        num_grav_tabs = ((num_grav_tabs_per_frame * n_side_connection) + 
                         (n_cxn_per_reg_frame * n_side_connection))
        
        # use blocks of 10
        cmp_strct = pd.concat([pd.DataFrame([[cur_cmp, 'ea', 'all', '0',
                                 num_grav_tabs, np.nan, np.nan,
                                 num_grav_tabs//9, metadata[cur_cmp][
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
                    [pd.DataFrame([[cur_cmp, 'ea', splice_floor_ind+1, '0',
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
        
    # nqe function
        
    def floor_qty_estimate(self, area_usage, mean_data, std_data, meta_data):
        import pandas as pd
        fl_cmp_by_usage = mean_data * area_usage
        
        import numpy as np
        has_stuff = fl_cmp_by_usage.copy()
        has_stuff[has_stuff != 0] = 1
        
        # variance per floor
        var_present = np.square(std_data.values * has_stuff.values)
        
        # var_xy = var_x + var_y; std = sqrt(var)
        std_cmp = np.sqrt(np.sum(var_present, axis=1))
        
        fl_std = pd.Series(std_cmp, index=std_data.index)
        
        # sum across all usage and adjust for base quantity, then round up
        fl_cmp_qty = fl_cmp_by_usage.sum(axis=1) * meta_data['quantity']
        return(fl_cmp_by_usage, fl_cmp_qty, fl_std)

    # function to remove (cmp, dir, loc) duplicates. assumes that only
    # theta_0 and theta_1 changes
    def remove_dupes(self, dupe_df):
        import pandas as pd
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

    def bldg_wide_cmp(self, roof_df):
        import pandas as pd
        
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
    
    def normative_quantity_estimation(self, usage, P58_metadata, brace_dir='../resource/'):
        floor_area = self.L_bldg**2 # sq ft
        import pandas as pd
        import numpy as np
        cmp_marginal = pd.DataFrame()
        
        nqe_mean = self.mean_sheet
        nqe_std = self.std_sheet
        nqe_meta = self.meta_sheet
        
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
        for fl, fl_usage in enumerate(usage):
            area_usage = np.array(fl_usage)*floor_area
            
            fl_cmp_by_cat, fl_cmp_total, fl_cmp_std = self.floor_qty_estimate(
                area_usage, nqe_mean, nqe_std, nqe_meta)
            
            fl_cmp_total.name = 'Theta_0'
            fl_cmp_std.name = 'Theta_1'
            
            loc_series = pd.Series([fl+1]).repeat(
                len(fl_cmp_total)).set_axis(fl_cmp_total.index)
            
            dir_map = {True:'1,2', False:'0'}
            dir_series = nqe_meta.directional.map(dir_map)
            
            has_stdev = fl_cmp_std != 0
            has_stdev.name = 'Family'
            family_map = {True:'lognormal', False:np.nan}
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
            combined_dupe_rows = self.remove_dupes(dupes)
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
        
        combined_roof_rows = self.bldg_wide_cmp(roof_cmp)
        cmp_marginal = cmp_marginal[
            ~cmp_marginal['Component'].isin(combined_roof_rows['Component'])]
        
        cmp_marginal = pd.concat([cmp_marginal, combined_roof_rows], 
                                 axis=0, ignore_index=True)
        
        # hardcoded no-block list
        no_block_stuff = ['Chiller', 'Cooling Tower', 'Motor Control', 'stair',
                          'Elevator', 'Switchgear']
        mask = cmp_marginal['Comment'].str.contains('|'.join(no_block_stuff))
        cmp_marginal.loc[mask, 'Blocks'] = cmp_marginal.loc[mask, 'Theta_0']//1.0
        
        # convert all 0 blocks to nan
        mask = cmp_marginal['Blocks'] == 0
        cmp_marginal.loc[mask, 'Blocks'] = np.nan
        
        # total loss cmps
        replace_df = pd.DataFrame([['excessiveRID', 'ea' , 'all', '1,2', 
                                    '1', np.nan, np.nan, np.nan, 'Excessive residual drift'],
                                   ['irreparable', 'ea', 0, '1', 
                                    '1', np.nan, np.nan, np.nan, 'Irreparable building'],
                                   ['collapse', 'ea', 0, '1',
                                    '1', np.nan, np.nan, np.nan, 'Collapsed building']
                                   ], columns=cmp_marginal.columns)
        
        nsc_cmp = pd.concat([cmp_marginal, replace_df])
        
        # structural components
        superstructure = self.superstructure_system
        if superstructure == 'MF':
            structural_cmp = self.get_structural_cmp_MF(P58_metadata)
        else:
            structural_cmp = self.get_structural_cmp_CBF(P58_metadata, brace_dir)
        
        # dtype conversion
        structural_cmp[['Theta_0']] = structural_cmp[['Theta_0']].apply(pd.to_numeric)
        structural_cmp[['Blocks']] = structural_cmp[['Blocks']].apply(pd.to_numeric)
        
        structural_cmp['Theta_1'] = 0
        
        from numpy import ceil
        nsc_cmp[['Theta_0']] = nsc_cmp[['Theta_0']].apply(pd.to_numeric)
        nsc_cmp[['Theta_0']] = ceil(nsc_cmp[['Theta_0']])
        nsc_cmp[['Blocks']] = nsc_cmp[['Blocks']].apply(pd.to_numeric)
        
        total_cmps = pd.concat([structural_cmp, nsc_cmp], ignore_index=True)
        self.components = total_cmps
        
    def process_EDP(self, df_edp=None):

        # from ast import literal_eval
        # PID = literal_eval(self.PID)
        # PFV = literal_eval(self.PFV)
        # PFA = literal_eval(self.PFA)
        
        if df_edp is None:
            PID = self.PID
            PFV = self.PFV
            PFA = self.PFA
            # max_isol_disp = self.max_isol_disp
            
            PID_names_1 = ['PID-'+str(fl+1)+'-1' for fl in range(len(PID))]
            PID_names_2 = ['PID-'+str(fl+1)+'-2' for fl in range(len(PID))]
            
            PFA_names_1 = ['PFA-'+str(fl+1)+'-1' for fl in range(len(PFA))]
            PFA_names_2 = ['PFA-'+str(fl+1)+'-2' for fl in range(len(PFA))]
            
            PFV_names_1 = ['PFV-'+str(fl+1)+'-1' for fl in range(len(PFV))]
            PFV_names_2 = ['PFV-'+str(fl+1)+'-2' for fl in range(len(PFV))]
            
            import pandas as pd
            all_edps = PFA + PFV + PID + PFA + PFV + PID
            all_names = (PFA_names_1 + PFV_names_1 + PID_names_1 +
                         PFA_names_2 + PFV_names_2 + PID_names_2)
            edp_df = pd.DataFrame([all_edps], columns = all_names)
            
            
            edp_df.loc['Units'] = ['g' if edp.startswith('PFA') else
                                   'inps' if edp.startswith('PFV') else
                                   'rad' for edp in all_names]
    
            edp_df["new"] = range(1,len(edp_df)+1)
            edp_df.loc[edp_df.index=='Units', 'new'] = 0
            edp_df = edp_df.sort_values("new").drop('new', axis=1)
    
            self.edp = edp_df
        # made for validation mode, results in edp item being a df
        else:
            PID = df_edp['PID'].iloc[0]
            PFV = df_edp['PFV'].iloc[0]
            PFA = df_edp['PFA'].iloc[0]
            
            PID_names_1 = ['PID-'+str(fl+1)+'-1' for fl in range(len(PID))]
            PID_names_2 = ['PID-'+str(fl+1)+'-2' for fl in range(len(PID))]
            
            PFA_names_1 = ['PFA-'+str(fl+1)+'-1' for fl in range(len(PFA))]
            PFA_names_2 = ['PFA-'+str(fl+1)+'-2' for fl in range(len(PFA))]
            
            PFV_names_1 = ['PFV-'+str(fl+1)+'-1' for fl in range(len(PFV))]
            PFV_names_2 = ['PFV-'+str(fl+1)+'-2' for fl in range(len(PFV))]
            
            all_names = (PFA_names_1 + PFV_names_1 + PID_names_1 +
                         PFA_names_2 + PFV_names_2 + PID_names_2)
            
            import pandas as pd
            pid_df = pd.DataFrame(df_edp['PID'].to_list())
            pfv_df = pd.DataFrame(df_edp['PFV'].to_list())
            pfa_df = pd.DataFrame(df_edp['PFA'].to_list())
            
            
            
            edp_df = pd.concat([pfa_df, pfv_df, pid_df, pfa_df, pfv_df, pid_df],
                          axis=1)
            edp_df.columns = all_names
            edp_df.loc['Units'] = ['g' if edp.startswith('PFA') else
                                   'inps' if edp.startswith('PFV') else
                                   'rad' for edp in all_names]
    
            edp_df["new"] = range(1,len(edp_df)+1)
            edp_df.loc[edp_df.index=='Units', 'new'] = 0
            edp_df = edp_df.sort_values("new").drop('new', axis=1)
    
            self.edp = edp_df
    
    def estimate_damage(self, mode='generate', custom_fragility_db=None):
        
        
        from pelicun.assessment import Assessment
        from pelicun.base import convert_to_MultiIndex
        import pandas as pd
        import numpy as np
        
        # prepare edp
        # TODO: fix EDPs for validation (make suitable for distribution per IDA level)
        # determine what to do for validation
        # validation is currently same as generate
        if mode=='generate':
            raw_demands = self.edp.transpose()
            raw_demands.columns = ['Units', 'Value']
            raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
            raw_demands.index.names = ['type','loc','dir']
        # if mode is validation, treat the dataset as a distribution
        elif mode=='validation':
            raw_demands = self.edp.transpose()
            raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
            raw_demands.index.names = ['type','loc','dir']
            
        # if mode is maximize, add high EDPs to maximize loss
        elif mode=='maximize':
            raw_demands = self.edp.transpose()
            raw_demands.columns = ['Units', 'Value']
            raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
            raw_demands.index.names = ['type','loc','dir']
            raw_demands.loc[raw_demands['Units'] == 'g', 'Value'] = 1e4
            raw_demands.loc[raw_demands['Units'] == 'rad', 'Value'] = 1e2
            raw_demands.loc[raw_demands['Units'] == 'inps', 'Value'] = 1e6
        
        # initialize, no printing outputs, offset fixed with current components
        PAL = Assessment({
            "PrintLog": False, 
            "Seed": 985,
            "Verbose": False,
            "DemandOffset": {"PFA": 0, "PFV": 0}
        })
        
        ###########################################################################
        # DEMANDS
        ###########################################################################
        # TODO: change validation? can do either deterministic or distro
        if mode=='validation':
            PAL.demand.load_sample(raw_demands.T)
            PAL.demand.calibrate_model(
                {
                    "ALL": {
                        "DistributionFamily": "lognormal"
                    }
                }
            )
        else:
            # specify deterministic demands
            demands = raw_demands
            demands.insert(1, 'Family',"deterministic")
            demands.rename(columns = {'Value': 'Theta_0'}, inplace=True)
            
            # prepare a correlation matrix that represents perfect correlation
            ndims = demands.shape[0]
            demand_types = demands.index 
        
            perfect_CORR = pd.DataFrame(
                np.ones((ndims, ndims)),
                columns = demand_types,
                index = demand_types)
        
            # load the demand model
            PAL.demand.load_model({'marginals': demands,
                                   'correlation': perfect_CORR})

        # generate demand sample
        n_sample = 1000
        PAL.demand.generate_sample({"SampleSize": n_sample})

        # extract the generated sample
        # Note that calling the save_sample() method is better than directly pulling the 
        # sample attribute from the demand object because the save_sample method converts
        # demand units back to the ones you specified when loading in the demands.
        demand_sample = PAL.demand.save_sample()


        # TODO: this is structure system dependent
        # get residual drift estimates 
        delta_y = 0.0075 # found from typical pushover curve for structure
        PID = demand_sample['PID']

        RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y}) 

        # and join them with the demand_sample
        demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

        # add spectral acceleration at fundamental period (BEARING EFFECTIVE PERIOD)
        # does Sa(T) characterize this well considering impact? for now, try using peak acceleration
        # demand_sample_ext[('SA_Tm',0,1)] = max(run_data['accMax0'],
        #                                        run_data['accMax1'],
        #                                        run_data['accMax2'],
        #                                        run_data['accMax3'])
        
        PID_all = self.PID
        if mode != 'maximize':
            demand_sample_ext[('SA_Tm',0,1)] = self.sa_tm
            demand_sample_ext[('PID_all',0,1)] = max(PID_all)
        else:
            demand_sample_ext[('SA_Tm',0,1)] = 1e4
            demand_sample_ext[('PID_all',0,1)] = 1e2
        
        # demand_sample_ext[('PID_all',0,1)] = demand_sample_ext[[('PID','1','1'),
        #                                                         ('PID','2','1'),
        #                                                         ('PID','3','1')]].max(axis=1)
        
        # add units to the data 
        demand_sample_ext.T.insert(0, 'Units',"")

        # PFA and SA are in "g" in this example, while PID and RID are "rad"
        demand_sample_ext.loc['Units', ['PFA', 'SA_Tm']] = 'g'
        demand_sample_ext.loc['Units',['PID', 'PID_all', 'RID']] = 'rad'
        demand_sample_ext.loc['Units',['PFV']] = 'inps'


        PAL.demand.load_sample(demand_sample_ext)
        
        # ###########################################################################
        # # COMPONENTS
        # ###########################################################################
        
        # generate structural components and join with NSCs
        
        cmp_marginals = self.components
        cmp_marginals = cmp_marginals.set_index('Component')
        cmp_marginals.index.names = ['Index']
        
        cmp_marginals[["Theta_0", "Theta_1", "Blocks"]] = cmp_marginals[[
            "Theta_0", "Theta_1", "Blocks"]].apply(pd.to_numeric)
        cmp_marginals['Location'] = cmp_marginals['Location'].astype(str)
        
        
        
        # review the damage model - in this example: fragility functions
        P58_data = PAL.get_default_data('damage_DB_FEMA_P58_2nd')

        # note that we drop the last three components here (excessiveRID, irreparable, and collapse) 
        # because they are not part of P58
        cmp_list = cmp_marginals.index.unique().values[:-3]

        P58_data_for_this_assessment = P58_data.loc[cmp_list,:].sort_values('Incomplete', ascending=False)
        
        # to make the convenience keywords work in the model, 
        # we need to specify the number of stories
        PAL.stories = len(PID_all)

        # now load the model into Pelicun
        PAL.asset.load_cmp_model({'marginals': cmp_marginals})
        
        # Generate the component quantity sample
        PAL.asset.generate_cmp_sample()

        # get the component quantity sample - again, use the save function to convert units
        cmp_sample = PAL.asset.save_cmp_sample()

        # load in some custom definitions for a couple of missing components
        incomplete_db = P58_data_for_this_assessment.loc[
            P58_data_for_this_assessment['Incomplete'] == 1].sort_index() 
        inc_names = incomplete_db.index.tolist()
        
        if custom_fragility_db is None:
            custom_fragility_db = pd.read_csv('../resource/loss/custom_component_fragilities.csv',
                                              header=[0,1], index_col=0)
            
        custom_fragility_db = custom_fragility_db.rename(
            columns=lambda x: '' if "Unnamed" in x else x, level=1)
        
        additional_fragility_db = pd.concat([custom_fragility_db, incomplete_db], axis=0)
        
        # if we have all components accounted for, drop the initial duplicate list
        if set(inc_names).issubset(custom_fragility_db.index.tolist()):
            additional_fragility_db = additional_fragility_db[additional_fragility_db['Incomplete'] != 1]
            
        mask = additional_fragility_db.index.isin(inc_names)
        additional_fragility_db = additional_fragility_db[mask]
        
        
        # TODO: change this section to the system-dependent drift values
        # add demand for the replacement criteria
        # irreparable damage
        # this is based on the default values in P58
        additional_fragility_db.loc[
            'excessiveRID', [('Demand','Directional'),
                            ('Demand','Offset'),
                            ('Demand','Type'), 
                            ('Demand','Unit')]] = [1, 
                                                    0, 
                                                    'Residual Interstory Drift Ratio',
                                                    'rad']   

        additional_fragility_db.loc[
            'excessiveRID', [('LS1','Family'),
                            ('LS1','Theta_0'),
                            ('LS1','Theta_1')]] = ['lognormal', 0.01, 0.3]   

        additional_fragility_db.loc[
            'irreparable', [('Demand','Directional'),
                            ('Demand','Offset'),
                            ('Demand','Type'), 
                            ('Demand','Unit')]] = [1,
                                                    0,
                                                    'Peak Spectral Acceleration|Tm',
                                                    'g']   


        # a very high capacity is assigned to avoid damage from demands
        # this will trigger on excessiveRID instead
        additional_fragility_db.loc[
            'irreparable', ('LS1','Theta_0')] = 1e10 

        

        # sa_judg = calculate_collapse_SaT1(run_data)

        # capacity is assigned based on the example in the FEMA P58 background documentation
        # additional_fragility_db.loc[
        #     'collapse', [('Demand','Directional'),
        #                     ('Demand','Offset'),
        #                     ('Demand','Type'), 
        #                     ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|Tm', 'g']   

        # # use judgment method, apply 0.6 variance (FEMA P58 ch. 6)
        # additional_fragility_db.loc[
        #     'collapse', [('LS1','Family'),
        #                   ('LS1','Theta_0'),
        #                   ('LS1','Theta_1')]] = ['lognormal', sa_judg, 0.6]  

        # collapse capacity is assumed lognormal distributed with 10% interstory drift
        # being the mean + 1 stdev percentile
        # Mean provided by Lee and Foutch (2001) for SMRF
        # Std from Yun and Hamburger (2002)
        
        # we can define a lognormal distribution that results in a PID of 10% having
        # 84% collapse rate (10% is the mean+1std.dev)
        # Yun and Hamburger has beta (logarithmic stdev) value of 0.3 for 
        # 3-story global collapse drift, lowered by 0.05 if nonlin dynamic anly
        from math import log, exp
        from scipy.stats import norm
        drift_mu_plus_std = 0.1
        inv_norm = norm.ppf(0.84)
        beta_drift = 0.25
        mean_log_drift = exp(log(drift_mu_plus_std) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
        additional_fragility_db.loc[
            'collapse', [('Demand','Directional'),
                            ('Demand','Offset'),
                            ('Demand','Type'), 
                            ('Demand','Unit')]] = [1, 0, 'Peak Interstory Drift Ratio|all', 'rad']   

        additional_fragility_db.loc[
            'collapse', [('LS1','Family'),
                          ('LS1','Theta_0'),
                          ('LS1','Theta_1')]] = ['lognormal', mean_log_drift, beta_drift]

        # We set the incomplete flag to 0 for the additional components
        additional_fragility_db['Incomplete'] = 0
        
        # if maximizing, drop the three replacement damage states
        if mode == 'maximize':
            additional_fragility_db = additional_fragility_db.drop(
                index=['collapse', 'excessiveRID', 'irreparable'])
        
        # load fragility data
        PAL.damage.load_damage_model([
            additional_fragility_db,  # This is the extra fragility data we've just created
            'PelicunDefault/damage_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
        ])
        
        ### 3.3.5 Damage Process
        # 
        # Damage processes are a powerful new feature in Pelicun 3. 
        # They are used to connect damages of different components in the performance model 
        # and they can be used to create complex cascading damage models.
        # 
        # The default FEMA P-58 damage process is farily simple. The process below can be interpreted as follows:
        # * If Damage State 1 (DS1) of the collapse component is triggered (i.e., the building collapsed), 
        # then damage for all other components should be cleared from the results. 
        # This considers that component damages (and their consequences) in FEMA P-58 are conditioned on no collapse.

        # * If Damage State 1 (DS1) of any of the excessiveRID components is triggered 
        # (i.e., the residual drifts are larger than the prescribed capacity on at least one floor),
        # then the irreparable component should be set to DS1.

        # FEMA P58 uses the following process:
        if mode != 'maximize':
            dmg_process = {
                "1_collapse": {
                    "DS1": "ALL_NA"
                },
                "2_excessiveRID": {
                    "DS1": "irreparable_DS1"
                }
            }
        else:
            dmg_process = None
        
        ###########################################################################
        # DAMAGE
        ###########################################################################
        
        
        print('Damage estimation...')
        # Now we can run the calculation
        PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100)
        
        # Damage estimates
        damage_sample = PAL.damage.save_sample()
        
        print('Damage estimation complete!')
        
        ###########################################################################
        # LOSS
        ###########################################################################
        
        # we need to prepend 'DMG-' to the component names to tell pelicun to look for the damage of these components
        drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
        drivers = drivers[:-3]+drivers[-2:]

        # we are looking at repair consequences in this example
        # the components in P58 have consequence models under the same name
        # this assumes that there are three omitted components: irreparable, excessiveRID, collapse
        loss_models = cmp_marginals.index.unique().tolist()[:-3]

        # We will define the replacement consequence in the following cell.
        loss_models+=['replacement',]*2

        # Assemble the DataFrame with the mapping information
        # The column name identifies the type of the consequence model.
        loss_map = pd.DataFrame(loss_models, columns=['Repair'], index=drivers)
        
        # load the consequence models
        P58_data = PAL.get_default_data('loss_repair_DB_FEMA_P58_2nd')

        # group E (filing cabinets, bookcases)
        incomplete_cmp = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([('Incomplete',''), 
                                                  ('Quantity','Unit'), 
                                                  ('DV', 'Unit'), 
                                                  ('DS1','Theta_0'),
                                                  ('DS1','Theta_1'),
                                                  ('DS1','Family'),]),
            index=pd.MultiIndex.from_tuples([('E.20.22.102a','Cost'), 
                                              ('E.20.22.102a','Time'),
                                              ('E.20.22.112a','Cost'), 
                                              ('E.20.22.112a','Time'),
                                              ('E.20.22.114b','Cost'), 
                                              ('E.20.22.114b','Time'),
                                              ('E.20.22.106b','Cost'), 
                                              ('E.20.22.106b','Time'),])
        )
        
        # bookcases (unanchored vs anchored)
        incomplete_cmp.loc[('E.20.22.102a', 'Cost')] = [0, '1 EA', 'USD_2011',
                                                      '190.0,150.0|1,5', 0.35, 'lognormal']
        incomplete_cmp.loc[('E.20.22.102a', 'Time')] = [0, '1 EA', 'worker_day',
                                                      0.02, 0.5, 'lognormal']
        
        incomplete_cmp.loc[('E.20.22.106b', 'Cost')] = [0, '1 EA', 'USD_2011',
                                                      '250.0,150.0|1,5', 0.35, 'lognormal']
        incomplete_cmp.loc[('E.20.22.106b', 'Time')] = [0, '1 EA', 'worker_day',
                                                      0.03, 0.5, 'lognormal']

        # filing cabinets (unanchored vs anchored)
        incomplete_cmp.loc[('E.20.22.112a', 'Cost')] = [0, '1 EA', 'USD_2011',
                                                      '110.0,70.0|1,5', 0.35, 'lognormal']
        incomplete_cmp.loc[('E.20.22.112a', 'Time')] = [0, '1 EA', 'worker_day',
                                                      0.02, 0.5, 'lognormal']
        
        incomplete_cmp.loc[('E.20.22.114b', 'Cost')] = [0, '1 EA', 'USD_2011',
                                                      '170.0,70.0|1,5', 0.35, 'lognormal']
        incomplete_cmp.loc[('E.20.22.114b', 'Time')] = [0, '1 EA', 'worker_day',
                                                      0.03, 0.5, 'lognormal']
        
        # if maximizing, drop the three replacement damage states
        if mode == 'maximize':
            loss_map = loss_map.drop(
                index=['DMG-irreparable', 'DMG-collapse'])

        # get the consequences used by this assessment
        # grab all loss map values that are present in P58_data
        P58_available = list(set(P58_data.index.get_level_values(0).values).intersection(
            loss_map['Repair'].values.tolist()))
        P58_data_for_this_assessment = P58_data.loc[P58_available,:]
        
        # this should be taken care of from incomplete_cmp
        P58_missing = set(loss_map['Repair'].values[:-2]) - set(P58_available)

        # initialize the dataframe
        additional_consequences = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([('Incomplete',''), 
                                                  ('Quantity','Unit'), 
                                                  ('DV', 'Unit'), 
                                                  ('DS1', 'Theta_0'),
                                                  ('DS1', 'Theta_1'),
                                                  ('DS1', 'Family')]),
            index=pd.MultiIndex.from_tuples([('replacement','Cost'), 
                                              ('replacement','Time')])
        )
        
        # add the data about replacement cost and time

        # TODO: find replacement cost estimate
        # use PACT
        # assume $250/sf
        # assume 40% of replacement cost is labor, $680/worker-day for SF Bay Area
        
        # assume $600/sf
        bldg_area = self.L_bldg**2 * (self.num_stories + 1)
        replacement_cost = 600.0*bldg_area
        
        # assume 2 years timeline
        # assume 1 worker per 1000 sf, but can work in parallel of 2 floors
        n_worker_series = bldg_area/1000
        n_worker_parallel = n_worker_series/2
        replacement_time = n_worker_parallel*365*2
        additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA',
                                                                'USD_2011',
                                                                replacement_cost,
                                                                0,
                                                                np.nan]
        additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA',
                                                                'worker_day',
                                                                replacement_time,
                                                                0,
                                                                np.nan]
        
        
    
        # Load the loss model to pelicun
        PAL.repair.load_model(
            [additional_consequences, incomplete_cmp,
              "PelicunDefault/loss_repair_DB_FEMA_P58_2nd.csv"], 
            loss_map, decision_variables=['Cost', 'Time'])
        
        # and run the calculations
        print('Loss estimation...')
        PAL.repair.calculate()
        print('Loss estimation done!')
        
        # loss estimates
        loss_sample = PAL.repair.save_sample()
        
        
        # group components and ensure that all components and replacement are present
        loss_by_cmp = loss_sample.groupby(level=[0, 2], axis=1).sum()['Cost']
        for cmp_grp in list(cmp_list):
            if cmp_grp not in list(loss_by_cmp.columns):
                loss_by_cmp[cmp_grp] = 0
                
        if mode == 'maximize':
            # summarize by groups
            loss_groups = pd.DataFrame()
            loss_groups['B'] = loss_by_cmp[[
                col for col in loss_by_cmp.columns if col.startswith('B')]].sum(axis=1)
            loss_groups['C'] = loss_by_cmp[[
                col for col in loss_by_cmp.columns if col.startswith('C')]].sum(axis=1)
            loss_groups['D'] = loss_by_cmp[[
                col for col in loss_by_cmp.columns if col.startswith('D')]].sum(axis=1)
            loss_groups['E'] = loss_by_cmp[[
                col for col in loss_by_cmp.columns if col.startswith('E')]].sum(axis=1)
            
            # this returns NaN if collapse/irreparable is 100%
            loss_groups = loss_groups.describe()
            
            # aggregate
            agg_DF = PAL.repair.aggregate_losses()
            
            collapse_freq = 0.0
            irreparable_freq = 0.0
            
            return(cmp_sample, damage_sample, loss_sample, loss_groups, agg_DF,
                    collapse_freq, irreparable_freq)
            
        # grab replacement cost and convert to instances, fill with zeros if needed
        replacement_instances = pd.DataFrame()
        try:
            replacement_instances['collapse'] = loss_by_cmp['collapse']/replacement_cost
        except KeyError:
            loss_by_cmp['collapse'] = 0
            replacement_instances['collapse'] = pd.DataFrame(np.zeros((n_sample, 1)))
        try:
            replacement_instances['irreparable'] = loss_by_cmp['irreparable']/replacement_cost
        except KeyError:
            loss_by_cmp['irreparable'] = 0
            replacement_instances['irreparable'] = pd.DataFrame(np.zeros((n_sample, 1)))
        replacement_instances = replacement_instances.astype(int)
                
        # summarize by groups
        loss_groups = pd.DataFrame()
        loss_groups['B'] = loss_by_cmp[[
            col for col in loss_by_cmp.columns if col.startswith('B')]].sum(axis=1)
        loss_groups['C'] = loss_by_cmp[[
            col for col in loss_by_cmp.columns if col.startswith('C')]].sum(axis=1)
        loss_groups['D'] = loss_by_cmp[[
            col for col in loss_by_cmp.columns if col.startswith('D')]].sum(axis=1)
        loss_groups['E'] = loss_by_cmp[[
            col for col in loss_by_cmp.columns if col.startswith('E')]].sum(axis=1)
        
        # only summarize repair cost from non-replacement cases
        loss_groups = loss_groups.loc[
            (replacement_instances['collapse'] == 0) & (replacement_instances['irreparable'] == 0)]
        
        collapse_freq = replacement_instances['collapse'].sum(axis=0)/n_sample
        irreparable_freq = replacement_instances['irreparable'].sum(axis=0)/n_sample
        
        # this returns NaN if collapse/irreparable is 100%
        loss_groups = loss_groups.describe()
        
        # aggregate
        agg_DF = PAL.repair.aggregate_losses()
        
        return(cmp_sample, damage_sample, loss_sample, loss_groups, agg_DF,
                collapse_freq, irreparable_freq)
    
#%% test
'''
# run info
import pandas as pd
import numpy as np

idx = pd.IndexSlice
pd.options.display.max_rows = 30

# and import pelicun classes and methods
from pelicun.assessment import Assessment

# get database
# initialize, no printing outputs, offset fixed with current components
PAL = Assessment({
    "PrintLog": False, 
    "Seed": 985,
    "Verbose": False,
    "DemandOffset": {"PFA": 0, "PFV": 0}
})

# generate structural components and join with NSCs
P58_metadata = PAL.get_default_metadata('loss_repair_DB_FEMA_P58_2nd')

# data = pd.read_csv('../data/tfp_mf_db.csv')
pickle_path = '../data/'
main_obj = pd.read_pickle(pickle_path+"tfp_mf_db.pickle")
data = main_obj.ops_analysis
run = data.iloc[1]


floors = run.num_stories
area = run.L_bldg**2 # sq ft

# lab, health, ed, res, office, retail, warehouse, hotel
fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
bldg_usage = [fl_usage]*floors

area_usage = np.array(fl_usage)*area

loss = Loss_Analysis(run)
loss.nqe_sheets()
loss.normative_quantity_estimation(bldg_usage, P58_metadata)


additional_frag_db = pd.read_csv('../resource/loss/custom_component_fragilities.csv',
                                  header=[0,1], index_col=0)
loss.process_EDP()
[cmp, dmg, loss, loss_cmp, agg, 
 collapse_rate, irr_rate] = loss.estimate_damage(
     custom_fragility_db=additional_frag_db, mode='maximize')

'''