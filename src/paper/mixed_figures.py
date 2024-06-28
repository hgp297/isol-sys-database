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

#%% collapse fragility def
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

#%% normalize DVs and prepare all variables
df['bldg_area'] = df['L_bldg']**2 * (df['num_stories'] + 1)

df['replacement_cost'] = 600.0*(df['bldg_area'])
df['total_cmp_cost'] = df_loss_max['cost_50%']
df['cmp_replace_cost_ratio'] = df['total_cmp_cost']/df['replacement_cost']
df['median_cost_ratio'] = df_loss['cost_50%']/df['replacement_cost']
df['cmp_cost_ratio'] = df_loss['cost_50%']/df['total_cmp_cost']

# , but working in parallel (2x faster)
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

#%% engineering data
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

plt.close('all')
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
plt.close('all')

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
plt.close('all')

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
plt.close('all')

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
plt.close('all')

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

#%%  variable testing

print('========= stats for repair cost ==========')
from sklearn import preprocessing

X = df[['k_ratio', 'T_ratio_e', 'zeta_e', 'Q', 'RI', 'gap_ratio']]
y = df[cost_var].ravel()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.feature_selection import r_regression,f_regression

r_results = r_regression(X_scaled,y)
print("Pearson's R test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(r_results)


f_statistic, p_values = f_regression(X_scaled, y)
f_results = r_regression(X,y)
print("F test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print("F-statistics:", f_statistic)
print("P-values:", p_values)

print('========= stats for replacement_risk ==========')
from sklearn import preprocessing

X = df[['k_ratio', 'T_ratio_e', 'zeta_e', 'Q', 'RI', 'gap_ratio']]
y = df['replacement_freq'].ravel()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.feature_selection import r_regression,f_regression

r_results = r_regression(X_scaled,y)
print("Pearson's R test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print(r_results)


f_statistic, p_values = f_regression(X_scaled, y)
f_results = r_regression(X,y)
print("F test: k_ratio, T_ratio, zeta, Q, Ry, GR")
print("F-statistics:", f_statistic)
print("P-values:", p_values)