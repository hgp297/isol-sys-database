############################################################################
#               Figure generation (plotting, ML, inverse design)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: April 2024

# Description:  Main file which imports the structural database and starts the
# loss estimation

# Open issues:  (1) 

############################################################################
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from doe import GP

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
pd.options.mode.chained_assignment = None  

plt.close('all')

main_obj = pd.read_pickle("../../data/loss/structural_db_mixed_loss.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df_raw = main_obj.ops_analysis
df_raw = df_raw.reset_index(drop=True)

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 10].copy()

df = df.drop(columns=['index'])
# df = df_whole.head(100).copy()

df['max_drift'] = df.PID.apply(max)
df['log_drift'] = np.log(df['max_drift'])

df['max_velo'] = df.PFV.apply(max)
df['max_accel'] = df.PFA.apply(max)

df['T_ratio'] = df['T_m'] / df['T_fb']
df['T_ratio_e'] = df['T_m'] / df['T_fbe']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df['Bm'] = np.interp(df['zeta_e'], zetaRef, BmRef)

df['gap_ratio'] = (df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm']/df['Bm'])*df['T_m']**2)

df_loss = main_obj.loss_data

max_obj = pd.read_pickle("../../data/loss/structural_db_mixed_loss_max.pickle")
df_loss_max = max_obj.max_loss

#%%
# make a generalized 2D plotting grid, defaulted to gap and Ry
# grid is based on the bounds of input data
def make_2D_plotting_space(X, res, x_var='gap_ratio', y_var='RI', 
                           all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                           third_var_set = None, fourth_var_set = None,
                           x_bounds=None, y_bounds=None):
    
    if x_bounds == None:
        x_min = min(X[x_var])
        x_max = max(X[x_var])
    else:
        x_min = x_bounds[0]
        x_max = x_bounds[1]
    if y_bounds == None:
        y_min = min(X[y_var])
        y_max = max(X[y_var])
    else:
        y_min = y_bounds[0]
        y_max = y_bounds[1]
    xx, yy = np.meshgrid(np.linspace(x_min,
                                     x_max,
                                     res),
                         np.linspace(y_min,
                                     y_max,
                                     res))

    rem_vars = [i for i in all_vars if i not in [x_var, y_var]]
    third_var = rem_vars[0]
    fourth_var = rem_vars[-1]
       
    xx = xx
    yy = yy
    
    if third_var_set is None:
        third_var_val= X[third_var].median()
    else:
        third_var_val = third_var_set
    if fourth_var_set is None:
        fourth_var_val = X[fourth_var].median()
    else:
        fourth_var_val = fourth_var_set
    
    
    X_pl = pd.DataFrame({x_var:xx.ravel(),
                         y_var:yy.ravel(),
                         third_var:np.repeat(third_var_val,
                                             res*res),
                         fourth_var:np.repeat(fourth_var_val, 
                                              res*res)})
    X_plot = X_pl[all_vars]
                         
    return(X_plot)

# hard-coded
def make_design_space(res, zeta_fix=None):
    if zeta_fix is None:
        xx, yy, uu, vv = np.meshgrid(np.linspace(0.6, 1.5,
                                                 res),
                                     np.linspace(0.5, 2.25,
                                                 res),
                                     np.linspace(2.0, 5.0,
                                                 res),
                                     np.linspace(0.1, 0.25,
                                                 res))
    else:
        xx, yy, uu, vv = np.meshgrid(np.linspace(0.6, 1.5,
                                                 res),
                                     np.linspace(0.5, 2.25,
                                                 res),
                                     np.linspace(2.0, 5.0,
                                                 res),
                                     zeta_fix)    
                                 
    X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                         'RI':yy.ravel(),
                         'T_ratio':uu.ravel(),
                         'zeta_e':vv.ravel()})

    return(X_space)

#%% collapse fragility def
'''
import numpy as np
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
# collapse as a probability
from scipy.stats import lognorm
from math import log, exp


x = np.linspace(0, 0.15, 200)
mu = log(0.1)- 0.25*inv_norm
sigma = 0.25;

ln_dist = lognorm(s=sigma, scale=exp(mu))
p = ln_dist.cdf(np.array(x))

# plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.plot(x, p, label='Collapse', color='blue')

mu_irr = log(0.01)
ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
p_irr = ln_dist_irr.cdf(np.array(x))

ax.plot(x, p_irr, color='red', label='Irreparable')

axis_font = 20
subt_font = 18
xleft = 0.15
ax.set_ylim([0,1])
ax.set_xlim([0, xleft])
ax.set_ylabel('Limit state probability', fontsize=axis_font)
ax.set_xlabel('Drift ratio', fontsize=axis_font)

ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
ax.hlines(y=0.5, xmin=exp(mu), xmax=0.15, color='blue', linestyle=":")
ax.text(0.105, 0.52, r'PID = 0.078', fontsize=axis_font, color='blue')
ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

ax.vlines(x=0.1, ymin=0, ymax=0.84, color='blue', linestyle=":")
ax.hlines(y=0.84, xmin=0.1, xmax=xleft, color='blue', linestyle=":")
ax.text(0.105, 0.87, r'PID = 0.10', fontsize=axis_font, color='blue')
ax.plot([0.10], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

lower= ln_dist.ppf(0.16)
ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
ax.hlines(y=0.16, xmin=lower, xmax=xleft, color='blue', linestyle=":")
ax.text(0.105, 0.19, r'PID = 0.061', fontsize=axis_font, color='blue')
ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


ax.hlines(y=0.5, xmin=0.0, xmax=exp(mu_irr), color='red', linestyle=":")
lower = ln_dist_irr.ppf(0.16)
ax.hlines(y=0.16, xmin=0.0, xmax=lower, color='red', linestyle=":")
upper = ln_dist_irr.ppf(0.84)
ax.hlines(y=0.84, xmin=0.0, xmax=upper, color='red', linestyle=":")
ax.plot([lower], [0.16], marker='*', markersize=15, color="red", linestyle=":")
ax.plot([0.01], [0.5], marker='*', markersize=15, color="red", linestyle=":")
ax.plot([upper], [0.84], marker='*', markersize=15, color="red", linestyle=":")
ax.vlines(x=upper, ymin=0, ymax=0.84, color='red', linestyle=":")
ax.vlines(x=0.01, ymin=0, ymax=0.5, color='red', linestyle=":")
ax.vlines(x=lower, ymin=0, ymax=0.16, color='red', linestyle=":")

ax.text(0.005, 0.19, r'RID = 0.007', fontsize=axis_font, color='red')
ax.text(0.005, 0.87, r'RID = 0.013', fontsize=axis_font, color='red')
ax.text(0.005, 0.53, r'RID = 0.010', fontsize=axis_font, color='red')

ax.set_title('Moment frame fragility definition', fontsize=axis_font)
ax.grid()
label_size = 16
clabel_size = 12

ax.legend(fontsize=label_size, loc='upper center')
plt.show()

#%% collapse fragility def

import numpy as np
from scipy.stats import norm
inv_norm = norm.ppf(0.9)
# collapse as a probability
from scipy.stats import lognorm
from math import log, exp

x = np.linspace(0, 0.15, 200)
mu = log(0.05)- 0.55*inv_norm
sigma = 0.55

ln_dist = lognorm(s=sigma, scale=exp(mu))
p = ln_dist.cdf(np.array(x))

# plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.plot(x, p, label='Collapse', color='blue')

mu_irr = log(0.005)
ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
p_irr = ln_dist_irr.cdf(np.array(x))

ax.plot(x, p_irr, color='red', label='Irreparable')

axis_font = 20
subt_font = 18
xleft = 0.15
ax.set_ylim([0,1])
ax.set_xlim([0, xleft])
ax.set_ylabel('Limit state probability', fontsize=axis_font)
ax.set_xlabel('Drift ratio', fontsize=axis_font)

ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
ax.hlines(y=0.5, xmin=exp(mu), xmax=0.15, color='blue', linestyle=":")
ax.text(0.105, 0.52, r'PID = 0.0247', fontsize=axis_font, color='blue')
ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

ax.vlines(x=0.05, ymin=0, ymax=0.9, color='blue', linestyle=":")
ax.hlines(y=0.9, xmin=0.05, xmax=xleft, color='blue', linestyle=":")
ax.text(0.105, 0.92, r'PID = 0.05', fontsize=axis_font, color='blue')
ax.plot([0.05], [0.9], marker='*', markersize=15, color="blue", linestyle=":")

lower= ln_dist.ppf(0.1)
ax.vlines(x=lower, ymin=0, ymax=0.1, color='blue', linestyle=":")
ax.hlines(y=0.1, xmin=lower, xmax=xleft, color='blue', linestyle=":")
ax.text(0.105, 0.12, r'PID = 0.003', fontsize=axis_font, color='blue')
ax.plot([lower], [0.1], marker='*', markersize=15, color="blue", linestyle=":")


ax.hlines(y=0.5, xmin=0.0, xmax=exp(mu_irr), color='red', linestyle=":")
lower = ln_dist_irr.ppf(0.1)
ax.hlines(y=0.1, xmin=0.0, xmax=lower, color='red', linestyle=":")
upper = ln_dist_irr.ppf(0.9)
ax.hlines(y=0.9, xmin=0.0, xmax=upper, color='red', linestyle=":")
ax.plot([lower], [0.1], marker='*', markersize=15, color="red", linestyle=":")
ax.plot([0.005], [0.5], marker='*', markersize=15, color="red", linestyle=":")
ax.plot([upper], [0.9], marker='*', markersize=15, color="red", linestyle=":")
ax.vlines(x=upper, ymin=0, ymax=0.9, color='red', linestyle=":")
ax.vlines(x=0.005, ymin=0, ymax=0.5, color='red', linestyle=":")
ax.vlines(x=lower, ymin=0, ymax=0.1, color='red', linestyle=":")

# ax.text(0.005, 0.19, r'RID = 0.007', fontsize=axis_font, color='red')
# ax.text(0.005, 0.87, r'RID = 0.013', fontsize=axis_font, color='red')
ax.text(0.005, 0.53, r'RID = 0.005', fontsize=axis_font, color='red')

ax.set_title('Braced frame fragility definition', fontsize=axis_font)
ax.grid()
label_size = 16
clabel_size = 12

ax.legend(fontsize=label_size, loc='upper center')
plt.show()

'''
#%% normalize DVs and prepare all variables
df['bldg_area'] = df['L_bldg']**2 * (df['num_stories'] + 1)

df['replacement_cost'] = 600.0*(df['bldg_area'])
df['total_cmp_cost'] = df_loss_max['cost_50%']
df['cmp_replace_cost_ratio'] = df['total_cmp_cost']/df['replacement_cost']
df['median_cost_ratio'] = df_loss['cost_50%']/df['replacement_cost']
df['cmp_cost_ratio'] = df_loss['cost_50%']/df['total_cmp_cost']

# but working in parallel (2x faster)
df['replacement_time'] = df['bldg_area']/1000*365
df['total_cmp_time'] = df_loss_max['time_l_50%']
df['cmp_replace_time_ratio'] = df['total_cmp_time']/df['replacement_time']
df['median_time_ratio'] = df_loss['time_l_50%']/df['replacement_time']
df['cmp_time_ratio'] = df_loss['time_l_50%']/df['total_cmp_time']

df['replacement_freq'] = df_loss['replacement_freq']

df[['B_50%', 'C_50%', 'D_50%', 'E_50%']] = df_loss[['B_50%', 'C_50%', 'D_50%', 'E_50%']]

df['impacted'] = pd.to_numeric(df['impacted'])

cost_var = 'cmp_cost_ratio'
time_var = 'cmp_time_ratio'
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

mask = df['B_50%'].isnull()

df['B_50%'].loc[mask] = df_loss_max['B_50%'].loc[mask]
df['C_50%'].loc[mask] = df_loss_max['C_50%'].loc[mask]
df['D_50%'].loc[mask] = df_loss_max['D_50%'].loc[mask]
df['E_50%'].loc[mask] = df_loss_max['E_50%'].loc[mask]


#%% subsets

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']
df_cbf = df[df['superstructure_system'] == 'CBF']
df_mf = df[df['superstructure_system'] == 'MF']

#%% engineering data
'''
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
fig = plt.figure(figsize=(13, 8))

bins = pd.IntervalIndex.from_tuples([(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.5)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['gap_ratio'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 1)
import seaborn as sns
sns.stripplot(data=df, x="max_drift", y="bin", orient="h", alpha=0.8, size=5,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_drift", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)

ax.set_ylabel('$GR$ range', fontsize=axis_font)
ax.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
plt.xlim([0.0, 0.15])

#####
bins = pd.IntervalIndex.from_tuples([(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.25)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['RI'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 2)
import seaborn as sns
sns.stripplot(data=df, x="max_drift", y="bin", orient="h", size=5, alpha=0.8,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_drift", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$R_y$ range', fontsize=axis_font)
ax.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
plt.xlim([0.0, 0.15])



#####
bins = pd.IntervalIndex.from_tuples([(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['T_ratio'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 3)
import seaborn as sns
sns.stripplot(data=df, x="max_accel", y="bin", orient="h", size=5, alpha=0.8,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_accel", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$T_M/T_{fb}$ range', fontsize=axis_font)
ax.set_xlabel('Peak floor acceleration (g)', fontsize=axis_font)
plt.xlim([0.0, 5.0])

#####
bins = pd.IntervalIndex.from_tuples([(0.1, 0.14), (0.14, 0.18), (0.18, 0.22), (0.22, 0.25)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['zeta_e'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 4)
import seaborn as sns
sns.stripplot(data=df, x="max_velo", y="bin", orient="h", size=5, alpha=0.8,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_velo", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$\zeta_M$ range', fontsize=axis_font)
ax.set_xlabel('Peak floor velocity (in/s)', fontsize=axis_font)
plt.xlim([25, 125.0])
fig.tight_layout(h_pad=2.0)
plt.show()

#%% seaborn scatter with histogram: Structural data
def scatter_hist(x, y, c, alpha, ax, ax_histx, ax_histy, label=None):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    cmap = plt.cm.Blues
    ax.scatter(x, y, alpha=alpha, edgecolors='black', s=25, facecolors=c,
               label=label)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    
    if y.name == 'zeta_e':
        binwidth = 0.02
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bin_y = np.arange(-lim, lim + binwidth, binwidth)
    elif y.name == 'RI':
        binwidth = 0.15
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bin_y = np.arange(-lim, lim + binwidth, binwidth)
    else:
        bin_y = bins
        
    if x.name == 'Q':
        binwidth = 0.01
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bin_x = np.arange(-lim, lim + binwidth, binwidth)
    else:
        bin_x = bins
    ax_histx.hist(x, bins=bin_x, alpha = 0.5, weights=np.ones(len(x)) / len(x),
                  facecolor = c, edgecolor='navy', linewidth=0.5)
    ax_histy.hist(y, bins=bin_y, orientation='horizontal', alpha = 0.5, weights=np.ones(len(x)) / len(x),
                  facecolor = c, edgecolor='navy', linewidth=0.5)
    
df_cbf = df[df['superstructure_system'] == 'CBF']
df_mf = df[df['superstructure_system'] == 'MF']

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# Start with a square Figure.
fig = plt.figure(figsize=(13, 6), layout='constrained')

# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 4,  width_ratios=(5, 1, 5, 1), height_ratios=(1, 5),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0., hspace=0.)
# # Create the Axes.
# fig = plt.figure(figsize=(13, 10))
# ax1=fig.add_subplot(2, 2, 1)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_cbf['T_ratio'], df_cbf['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy,
             label='CBF')
scatter_hist(df_mf['T_ratio'], df_mf['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy,
             label='MF')
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_e$', fontsize=axis_font)
ax.set_xlim([1.0, 12.0])
ax.set_ylim([0.1, 0.25])
ax.legend(fontsize=label_size)

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_cbf['T_ratio'], df_cbf['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df_mf['T_ratio'], df_mf['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlim([1.0, 12.0])
ax.set_ylim([0.1, 0.25])

#####################

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']

# plt.close('all')
# Start with a square Figure.
fig = plt.figure(figsize=(13, 6), layout='constrained')

# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 4,  width_ratios=(5, 1, 5, 1), height_ratios=(1, 5),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0., hspace=0.)
# # Create the Axes.
# fig = plt.figure(figsize=(13, 10))
# ax1=fig.add_subplot(2, 2, 1)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and df_tfp.
scatter_hist(df_tfp['Q'], df_tfp['k_ratio'], 'navy', 0.9, ax, ax_histx, ax_histy,
             label='TFP')
scatter_hist(df_lrb['Q'], df_lrb['k_ratio'], 'orange', 0.3, ax, ax_histx, ax_histy,
             label='LRB')
ax.set_xlabel(r'$Q / W$', fontsize=axis_font)
ax.set_ylabel(r'$k_1 / k_2$', fontsize=axis_font)
ax.set_xlim([0.04, 0.125])
ax.set_ylim([4, 18])
ax.legend(fontsize=label_size)

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_tfp['Q'], df_tfp['k_ratio'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df_lrb['Q'], df_lrb['k_ratio'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'$Q / W$', fontsize=axis_font)
ax.set_ylabel(r'$k_1 / k_2$', fontsize=axis_font)
ax.set_xlim([0.04, 0.125])
ax.set_ylim([4, 18])


#%% 3d surf for replacement risk - structure
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(df_cbf[xvar], df_cbf[yvar], df_cbf['replacement_freq'], color=color[0],
           edgecolors='k', alpha = 0.7, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], df_mf['replacement_freq'], color=color[1],
           edgecolors='k', alpha = 0.7, label='MF')

ax.legend(fontsize=label_size)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)

#################################
ax=fig.add_subplot(1, 2, 2, projection='3d')

xvar = 'T_ratio'
yvar = 'zeta_e'

ax.scatter(df_cbf[xvar], df_cbf[yvar], df_cbf['replacement_freq'], color=color[0],
           edgecolors='k', alpha = 0.7, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], df_mf['replacement_freq'], color=color[1],
           edgecolors='k', alpha = 0.7, label='MF')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.tight_layout()

#%% 3d surf for replacement risk - bearing
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

#################################
xvar = 'Q'
yvar = 'k_ratio'

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(df_tfp[xvar], df_tfp[yvar], df_tfp['replacement_freq'], color=color[2],
           edgecolors='k', alpha = 0.7, label='TFP')
ax.scatter(df_lrb[xvar], df_lrb[yvar], df_lrb['replacement_freq'], color=color[3],
           edgecolors='k', alpha = 0.7, label='LRB')

ax.legend(fontsize=label_size)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel('$Q/W$', fontsize=axis_font)
ax.set_ylabel('$k_1/k_2$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)

#################################
ax=fig.add_subplot(1, 2, 2, projection='3d')

xvar = 'T_ratio'
yvar = 'zeta_e'

ax.scatter(df_tfp[xvar], df_tfp[yvar], df_tfp['replacement_freq'], color=color[2],
           edgecolors='k', alpha = 0.7, label='TFP')
ax.scatter(df_lrb[xvar], df_lrb[yvar], df_lrb['replacement_freq'], color=color[3],
           edgecolors='k', alpha = 0.7, label='LRB')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.tight_layout()

#%% 3d surf for repair cost - structure
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(df_cbf[xvar], df_cbf[yvar], df_cbf[cost_var], color=color[0],
           edgecolors='k', alpha = 0.7, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], df_mf[cost_var], color=color[1],
           edgecolors='k', alpha = 0.7, label='MF')

ax.legend(fontsize=label_size)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Replacement cost', fontsize=axis_font)

#################################
ax=fig.add_subplot(1, 2, 2, projection='3d')

xvar = 'T_ratio'
yvar = 'zeta_e'

ax.scatter(df_cbf[xvar], df_cbf[yvar], df_cbf[cost_var], color=color[0],
           edgecolors='k', alpha = 0.7, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], df_mf[cost_var], color=color[1],
           edgecolors='k', alpha = 0.7, label='MF')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Replacement cost', fontsize=axis_font)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.tight_layout()

#%% 3d surf for repair cost - bearing
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

#################################
xvar = 'Q'
yvar = 'k_ratio'

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(df_tfp[xvar], df_tfp[yvar], df_tfp[cost_var], color=color[2],
           edgecolors='k', alpha = 0.7, label='TFP')
ax.scatter(df_lrb[xvar], df_lrb[yvar], df_lrb[cost_var], color=color[3],
           edgecolors='k', alpha = 0.7, label='LRB')

ax.legend(fontsize=label_size)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel('$Q/W$', fontsize=axis_font)
ax.set_ylabel('$k_1/k_2$', fontsize=axis_font)
ax.set_zlabel('Replacement cost ratio', fontsize=axis_font)

#################################
ax=fig.add_subplot(1, 2, 2, projection='3d')

xvar = 'T_ratio'
yvar = 'zeta_e'

ax.scatter(df_tfp[xvar], df_tfp[yvar], df_tfp[cost_var], color=color[2],
           edgecolors='k', alpha = 0.7, label='TFP')
ax.scatter(df_lrb[xvar], df_lrb[yvar], df_lrb[cost_var], color=color[3],
           edgecolors='k', alpha = 0.7, label='LRB')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Replacement cost ratio', fontsize=axis_font)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.tight_layout()
'''
#%%  variable testing

print('========= stats for repair cost ==========')
from sklearn import preprocessing

X = df[['k_ratio', 'T_ratio', 'zeta_e', 'Q', 'RI', 'gap_ratio']]
y = df[cost_var].ravel()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.feature_selection import r_regression,f_regression

r_results = r_regression(X_scaled,y)
print("Pearson's R test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(["%.4f" % member for member in r_results])

f_statistic, p_values = f_regression(X_scaled, y)
print("F test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(["%.4f" % member for member in f_statistic])
print("P-values: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(["%.4f" % member for member in p_values])

print('========= stats for replacement_risk ==========')
from sklearn import preprocessing

X = df[['k_ratio', 'T_ratio', 'zeta_e', 'Q', 'RI', 'gap_ratio']]
y = df['replacement_freq'].ravel()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.feature_selection import r_regression,f_regression

r_results = r_regression(X_scaled,y)
print("Pearson's R test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(["%.4f" % member for member in r_results])

f_statistic, p_values = f_regression(X_scaled, y)
print("F test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(["%.4f" % member for member in f_statistic])
print("P-values: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(["%.4f" % member for member in p_values])

#%% ml training

covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_cost_hit = GP(df_hit)
mdl_cost_hit.set_covariates(covariate_list)
mdl_cost_hit.set_outcome(cost_var)
mdl_cost_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_cost_miss = GP(df_miss)
mdl_cost_miss.set_covariates(covariate_list)
mdl_cost_miss.set_outcome(cost_var)
mdl_cost_miss.test_train_split(0.2)

mdl_time_hit = GP(df_hit)
mdl_time_hit.set_covariates(covariate_list)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = GP(df_miss)
mdl_time_miss.set_covariates(covariate_list)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

mdl_repl_hit = GP(df_hit)
mdl_repl_hit.set_covariates(covariate_list)
mdl_repl_hit.set_outcome('replacement_freq')
mdl_repl_hit.test_train_split(0.2)

mdl_repl_miss = GP(df_miss)
mdl_repl_miss.set_covariates(covariate_list)
mdl_repl_miss.set_outcome('replacement_freq')
mdl_repl_miss.test_train_split(0.2)

mdl_unconditioned = GP(df)
mdl_unconditioned.set_covariates(covariate_list)
mdl_unconditioned.set_outcome(cost_var)
mdl_unconditioned.test_train_split(0.2)

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
               outcome='cost_50%', return_var=False):
        
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
                    hit_loss_mdl.predict(X).ravel(),
                    hit_prob)})
            
#        # get points that are predicted no impact from full dataset
#        df_mss = self.X[preds_imp == 0]
    
    # run miss model on this dataset
    expected_DV_miss = pd.DataFrame(
            {outcome_str:np.multiply(
                    miss_loss_mdl.predict(X).ravel(),
                    miss_prob)})
    
    expected_DV = expected_DV_hit + expected_DV_miss
    
    # self.median_loss_pred = pd.concat([loss_pred_hit,loss_pred_miss], 
    #                                   axis=0).sort_index(ascending=True)
    if return_var:
        pass
    else:
        return(expected_DV)
    
#%% impact prediction

print('========== Fitting impact classification (GPC or KLR) ============')

# prepare the problem
mdl_impact = GP(df)
mdl_impact.set_covariates(covariate_list)
mdl_impact.set_outcome('impacted', use_ravel=True)
mdl_impact.test_train_split(0.2)

mdl_impact.fit_gpc(kernel_name='rbf_ard')
mdl_impact.fit_svc(kernel_name='rbf')
mdl_impact.fit_kernel_logistic(kernel_name='rbf')

# predict the entire dataset
preds_imp = mdl_impact.gpc.predict(mdl_impact.X)
probs_imp = mdl_impact.gpc.predict_proba(mdl_impact.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl_impact.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

#%% Classification plot

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 22
title_font = 22
subt_font = 18
import matplotlib as mpl
label_size = 18
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# make grid and plot classification predictions

fig, ax = plt.subplots(1, 1, figsize=(9,7))

xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact.gpc.predict_proba(X_plot)[:,1]


# # kernel logistic impact prediction
# K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)
# probs_imp = mdl_impact.log_reg_kernel.predict_proba(K_space)
# Z = probs_imp[:,1]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_classif = Z.reshape(xx_pl.shape)

# ax1.imshow(
#         Z,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.Greys,
#     )

plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.Blues,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=np.linspace(0.1,1.0,num=10))
clabels = plt.clabel(cs, fontsize=clabel_size, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

ax.scatter(df_hit[xvar][:plt_density],
            df_hit[yvar][:plt_density],
            s=40, c='darkblue', marker='v', edgecolors='crimson', label='Impacted')

ax.scatter(df_miss[xvar][:plt_density],
            df_miss[yvar][:plt_density],
            s=40, c='azure', edgecolors='k', label='No impact')
plt.legend(fontsize=axis_font)

ax.set_xlim(0.3, 2.5)
ax.set_title(r'Impact likelihood: $R_y = 2.0$, $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$T_M/T_{fb}$', fontsize=axis_font)

fig.tight_layout()
plt.show()


####

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 22
title_font = 22
subt_font = 18
import matplotlib as mpl
label_size = 18
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# make grid and plot classification predictions

fig, ax = plt.subplots(1, 1, figsize=(9,7))

xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact.gpc.predict_proba(X_plot)[:,1]


# # kernel logistic impact prediction
# K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)
# probs_imp = mdl_impact.log_reg_kernel.predict_proba(K_space)
# Z = probs_imp[:,1]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_classif = Z.reshape(xx_pl.shape)

# ax1.imshow(
#         Z,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.Greys,
#     )

plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.Blues,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=np.linspace(0.1,1.0,num=10))
clabels = plt.clabel(cs, fontsize=clabel_size, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

ax.scatter(df_hit[xvar][:plt_density],
            df_hit[yvar][:plt_density],
            s=40, c='darkblue', marker='v', edgecolors='crimson', label='Impacted')

ax.scatter(df_miss[xvar][:plt_density],
            df_miss[yvar][:plt_density],
            s=40, c='azure', edgecolors='k', label='No impact')
plt.legend(fontsize=axis_font)

# ax.set_xlim(0.3, 2.5)
ax.set_title(r'Impact likelihood: $R_y = 2.0$, $GR = 1.0$', fontsize=title_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)

fig.tight_layout()
plt.show()

#%% fit regressions for impact / non-impact set

# Fit conditioned DVs using kernel ridge

print('========== Fitting regressions (kernel ridge) ============')

# fit impacted set
mdl_cost_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')

# fit no impact set
mdl_cost_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')


mdl_repl_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_repl_miss.fit_kernel_ridge(kernel_name='rbf')

'''
print('========== Fitting regressions (SVR) ============')

# fit impacted set
mdl_cost_hit.fit_svr()
mdl_time_hit.fit_svr()

# fit no impact set
mdl_cost_miss.fit_svr()
mdl_time_miss.fit_svr()


mdl_repl_hit.fit_svr()
mdl_repl_miss.fit_svr()
'''

print('========== Fitting regressions (GPR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_gpr(kernel_name='rbf_iso')
mdl_time_hit.fit_gpr(kernel_name='rbf_iso')

# fit no impact set
mdl_cost_miss.fit_gpr(kernel_name='rbf_iso')
mdl_time_miss.fit_gpr(kernel_name='rbf_iso')


mdl_repl_hit.fit_gpr(kernel_name='rbf_iso')
mdl_repl_miss.fit_gpr(kernel_name='rbf_iso')

mdl_unconditioned.fit_gpr(kernel_name='rbf_iso')

print('========== Fitting ordinary ridge (OR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_ols_ridge()
mdl_time_hit.fit_ols_ridge()

# fit no impact set
mdl_cost_miss.fit_ols_ridge()
mdl_time_miss.fit_ols_ridge()


mdl_repl_hit.fit_ols_ridge()
mdl_repl_miss.fit_ols_ridge()

#%% 2d contours for replacement

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_contour = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1)

plt_density = 200
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z_contour, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df[xvar][:plt_density], df[yvar][:plt_density], 
            c=df['replacement_freq'][:plt_density],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.3, 2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('$GR$', fontsize=axis_font)
plt.ylabel('$R_y$', fontsize=axis_font)
plt.grid(True)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_contour = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2)

plt_density = 200
# lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax.contour(xx_pl, yy_pl, Z_contour, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
ax.scatter(df[xvar][:plt_density], df[yvar][:plt_density], 
            c=df['replacement_freq'][:plt_density],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
# plt.xlim([2.0, 5.0])
# plt.ylim([0.1, 0.25])
plt.xlabel('$T_M/T_{fb}$', fontsize=axis_font)
plt.ylabel('$\zeta_M$', fontsize=axis_font)
plt.grid(True)
plt.show()

#%% 3d surf for replacement risk
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5)]

ax.scatter(df_sc[xvar], df_sc[yvar], df_sc['replacement_freq'], c=df_sc['replacement_freq'],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_cost_hit.gpr, mdl_cost_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

df_sc = df[(df['gap_ratio']<=1.2) & (df['gap_ratio']>=0.8)]

ax.scatter(df_sc[xvar], df_sc[yvar], df_sc['replacement_freq'], c=df_sc['replacement_freq'],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

df_t = df[(df['gap_ratio']<=1.2) & (df['gap_ratio']>=0.8) & 
           (df['T_ratio']>=3.0)]
fig.tight_layout()
plt.show()

#%% 3d surf for cost ratio
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome=cost_var)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df[cost_var], c=df[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_cost_hit.gpr, mdl_cost_miss.gpr, outcome=cost_var)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df[cost_var], c=df[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% dirty contours replacement time and cost


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax = fig.add_subplot(1, 2, 1)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

grid_cost = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_cost_hit.gpr,
                       mdl_cost_miss.gpr,
                       outcome=cost_var)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
clabels = ax.clabel(cs, fontsize=clabel_size)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

prob_list = [0.3, 0.2, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.4, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5) & 
           (df['zeta_e']<=0.2) & (df['zeta_e']>=0.13)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc[cost_var], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$c_r$",
                      fontsize=subt_font, title_fontsize=subt_font)

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax = fig.add_subplot(1, 2, 2)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

grid_time = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_time_hit.gpr,
                       mdl_time_miss.gpr,
                       outcome=time_var)

Z = np.array(grid_time)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
clabels = ax.clabel(cs, fontsize=clabel_size)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

prob_list = [0.3, 0.2, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.4, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5) & 
           (df['zeta_e']<=0.2) & (df['zeta_e']>=0.13)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc[cost_var], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$t_r$",
                      fontsize=subt_font, title_fontsize=subt_font)


fig.tight_layout()
plt.show()

#%% dirty contours (presenting replacement risk for EMI)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fig, ax = plt.subplots(1, 1, figsize=(9,7))
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

prob_target = 0.1
grid_repl = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_repl_hit.gpr,
                       mdl_repl_miss.gpr,
                       outcome='replacement_freq')

Z = np.array(grid_repl)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
clabels = ax.clabel(cs, fontsize=clabel_size)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

prob_list = [0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.4, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5) & 
           (df['zeta_e']<=0.2) & (df['zeta_e']>=0.13)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc['replacement_freq'], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$p_r$",
                      fontsize=subt_font, title_fontsize=subt_font)

fig.tight_layout()
plt.show()

#%% Testing the design space
import time

res_des = 20
X_space = make_design_space(res_des)
#K_space = mdl.get_kernel(X_space, kernel_name='rbf', gamma=gam)

# choice GPC
# HOWEVER, SVC is poorly calibrated for probablities
# consider using GP if computational resources allow, and GP looks good

# choice GPR for all prediction model for richness. HOWEVER must keep resolution low
t0 = time.time()
space_repair_cost = predict_DV(X_space, 
                               mdl_impact.gpc, 
                               mdl_cost_hit.gpr, 
                               mdl_cost_miss.gpr, 
                               outcome=cost_var)
tp = time.time() - t0
print("GPC-GPR repair cost prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                           tp))

t0 = time.time()
space_downtime = predict_DV(X_space,
                            mdl_impact.gpc,
                            mdl_time_hit.gpr,
                            mdl_time_miss.gpr,
                            outcome=time_var)
tp = time.time() - t0
print("GPC-GPR downtime prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

t0 = time.time()
space_repl = predict_DV(X_space,
                        mdl_impact.gpc,
                        mdl_repl_hit.gpr,
                        mdl_repl_miss.gpr,
                        outcome='replacement_freq')
tp = time.time() - t0
print("GPC-GPR replacement prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% Calculate upfront cost of data

# TODO: upfront cost calcs for CBFs and TFPs

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
#%% baseline predictions
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 3.0, 0.15]]),
                          columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])
baseline_repair_cost_pred = predict_DV(X_baseline,
                                      mdl_impact.gpc,
                                      mdl_cost_hit.kr,
                                      mdl_cost_miss.kr,
                                      outcome=cost_var)[cost_var+'_pred'].item()
baseline_downtime_pred = predict_DV(X_baseline,
                                      mdl_impact.gpc,
                                      mdl_time_hit.kr,
                                      mdl_time_miss.kr,
                                      outcome=time_var)[time_var+'_pred'].item()


baseline_repl_risk_pred = predict_DV(X_baseline,
                                      mdl_impact.gpc,
                                      mdl_repl_hit.kr,
                                      mdl_repl_miss.kr,
                                      outcome='replacement_freq')['replacement_freq_pred'].item()

uncut_baseline_cost = mdl_unconditioned.gpr.predict(X_baseline)
    
