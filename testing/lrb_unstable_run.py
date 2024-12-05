import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')

from building import Building
import json
input_path = '../src/inputs/'
run_case_str = 'mf_lrb_strict'
with open(input_path+run_case_str+'.in') as f: 
    data = f.read() 
design_dict = json.loads(data)

# use db to prepare all IDA runs, then grab the assigned row
from db import prepare_ida_util
ida_df = prepare_ida_util(design_dict)
row_num = 4
run = ida_df.iloc[row_num]

gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/'
from experiment import run_nlth
print('========= Run %d of %d ==========' % 
      (row_num+1, len(ida_df)))

bldg = Building(run)
bldg.model_frame()
bldg.apply_grav_load()

T_1 = bldg.run_eigen()

bldg.provide_damping(80, method='SP',
                                  zeta=[0.05], modes=[1])

dt = 0.005
ok = bldg.run_ground_motion(run.gm_selected, 
                        run.scale_factor*1.0, 
                        dt, T_end=60.0, data_dir='./output/')

#%%
from plot_structure import plot_dynamic
plot_dynamic(run, data_dir='./output/')