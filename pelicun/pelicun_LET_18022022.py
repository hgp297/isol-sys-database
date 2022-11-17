#!/usr/bin/env
# coding: utf-8

# ![alt text](Figures/pelicun_LET.png "Live Expert Tips - Pelicun3 in Jupyter")

# **<p style="text-align: center;">Live Expert Tips    |    February 18, 2022    |    Adam Zsarnóczay</p>**
# 
# This notebook is published under Live Expert Tips 2022 February in the Pelicun Examples project in DesignSafe (project number PRJ-3411)

# # Context

# Pelicun is one of many tools developed by the SimCenter. Learn more about our tools at the [SimCenter Website](https://simcenter.designsafe-ci.org/)
# 
# ![alt text](Figures/SimCenter_website.png "SimCenter Website")

# Pelicun 2.6 is still the stable version. Pelicun 3.0 is in beta now and expected to become stable in a few months
# 
# ![alt text](Figures/pelicun_docs.png "Pelicun documentation")

# # Introduction to Pelicun

# Pelicun is part of the SimCenter application framework
# 
# ![alt text](Figures/SimCenter_App_Framework.png "SimCenter Application Framework")

# Pelicun provides a unifying performance assessment engine that aims to integrate all popular methods across hazards, asset types and resolutions
# 
# ![alt text](Figures/Pelicun_framework.png "Pelicun Framework")

# # Setting up the environment





# In[4]:


# import helpful packages for numerical analysis
import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 100

# and for plotting
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import plotly.io as pio
pio.renderers.default='browser'

import seaborn as sns

# and import pelicun classes and methods
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment


# # Detailed calculation

# ## Demands

# ![alt text](Figures/pelicun_LET.png "Live Expert Tips - Pelicun3 in Jupyter")

# ### Load demand distribution data

# Most of this example, including the demand distribution data is based on the example building featured in FEMA P58. That example is described in more detail in the background documentation:
# 
# <img src="Figures/P58_background_doc.png" alt="FEMA P58 background documentation" width="400"/>

# In[5]:


raw_demands = pd.read_csv('./demand_data.csv', index_col=0)
raw_demands


# Pelicun uses SimCenter's naming convention for demands:
# - The first number represents the event_ID. This can be used to differentiate between multiple stripes of an analysis, or multiple consecutive events in a main-shock - aftershock sequence, for example. Pelicun does not use the first number internally.
# - The type of the demand identifies the EDP or IM. The following options are available:
#     * 'Story Drift Ratio' :             'PID',
#     * 'Peak Interstory Drift Ratio':    'PID',
#     * 'Roof Drift Ratio' :              'PRD',
#     * 'Peak Roof Drift Ratio' :         'PRD',
#     * 'Damageable Wall Drift' :         'DWD',
#     * 'Racking Drift Ratio' :           'RDR',
#     * 'Peak Floor Acceleration' :       'PFA',
#     * 'Peak Floor Velocity' :           'PFV',
#     * 'Peak Gust Wind Speed' :          'PWS',
#     * 'Peak Inundation Height' :        'PIH',
#     * 'Peak Ground Acceleration' :      'PGA',
#     * 'Peak Ground Velocity' :          'PGV',
#     * 'Spectral Acceleration' :         'SA',
#     * 'Spectral Velocity' :             'SV',
#     * 'Spectral Displacement' :         'SD',
#     * 'Peak Spectral Acceleration' :    'SA',
#     * 'Peak Spectral Velocity' :        'SV',
#     * 'Peak Spectral Displacement' :    'SD',
#     * 'Permanent Ground Deformation' :  'PGD',
#     * 'Mega Drift Ratio' :              'PMD',
#     * 'Residual Drift Ratio' :          'RID',
#     * 'Residual Interstory Drift Ratio':'RID'
# - The third part is an integer the defines the location where the demand was recorded. In buildings, locations are typically floors, but in other assets, locations could reference any other part of the structure.
# - The last part is an integer the defines the direction of the demand. Typically 1 stands for horizontal X and 2 for horizontal Y, but any other numbering convention can be used.
# 
# The location and direction numbers need to be in line with the component definitions presented later.

# In[7]:


# convert index to MultiIndex to make it easier to slice the data
raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
raw_demands
raw_demands.index.names = ['stripe','type','loc','dir']
raw_demands.tail(30)


# ### Prepare input for pelicun

# In[8]:


# now prepare the demand information in the format pelicun can consume it as a set of marginal distributions

# we'll use stripe 3 for this example
stripe = '3'
stripe_demands = raw_demands.loc[stripe,:]

# units - - - - - - - - - - - - - - - - - - - - - - - -  
stripe_demands.insert(0, 'Units',"")

# PFA is in "g" in this example, while PID is "rad"
stripe_demands.loc['PFA','Units'] = 'g'
stripe_demands.loc['PID','Units'] = 'rad'

# distribution family  - - - - - - - - - - - - - - - - -  
stripe_demands.insert(1, 'Family',"")

# we assume lognormal distribution for all demand marginals
stripe_demands['Family'] = 'lognormal'

# distribution parameters  - - - - - - - - - - - - - - -
# pelicun uses generic parameter names to handle various distributions within the same data structure
# we need to rename the parameter columns as follows:
# median -> theta_0
# log_std -> theta_1
stripe_demands.rename(columns = {'median': 'Theta_0'}, inplace=True)
stripe_demands.rename(columns = {'log_std': 'Theta_1'}, inplace=True)

stripe_demands


# In[9]:


# let's plot these demands to perform a sanity check before the analysis


# In[10]:


fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('<b>Peak Interstory Drift ratio</b><br> ', 
                                    '<b>Peak Floor Acceleration</b><br> '), 
                    shared_yaxes=True,
                    horizontal_spacing=0.05,
                    vertical_spacing=0.05)

for demand_i, demand_type in enumerate(["PID", "PFA"]):
    
    if demand_type == 'PID':
        offset = -0.5
    else:
        offset = 0.0
    
    for d_i, (dir_, d_color) in enumerate(zip([1,2], ['blue', 'red'])):
            
        result_name = f'{demand_type} dir {dir_}'

        params = stripe_demands.loc[idx[demand_type, :, str(dir_)], ['Theta_0', 'Theta_1']]     
        params.index = params.index.get_level_values(1).astype(float)

        # plot +- 2 log std
        for mul, m_dash in zip([1, 2], ['dash', 'dot']):

            if mul == 1:
                continue

            for sign in [-1, 1]:
                fig.add_trace(go.Scatter(
                    x = np.exp(np.log(params['Theta_0'].values) + params['Theta_1'].values*sign*mul),
                    y = params.index+offset,
                    hovertext = result_name + " median +/- 2logstd",
                    name = result_name + " median +/- 2logstd",
                    mode='lines+markers',
                    line=dict(color=d_color, dash=m_dash, width=0.5),   
                    marker=dict(size=4/mul),
                    showlegend=False,
                ), row=1, col=demand_i+1,)

        # plot the medians
        fig.add_trace(go.Scatter(
            x = params['Theta_0'].values,
            y = params.index+offset,
            hovertext = result_name + ' median',
            name = result_name + ' median',
            mode='lines+markers',
            line=dict(color=d_color, width=1.0),
            marker=dict(size=8),
            showlegend=False,
        ),row=1, col=demand_i+1,)

        if d_i == 0:

            shared_ax_props = dict(
                showgrid = True,
                linecolor = 'black',
                gridwidth = 0.05,
                gridcolor = 'rgb(192,192,192)',
            )

            if demand_type == 'PID':
                fig.update_xaxes(title_text='drift ratio', 
                                 range=[0,0.05], row=1, col=demand_i+1,
                                 **shared_ax_props)

            elif demand_type == 'PFA':
                fig.update_xaxes(title_text='acceleration [g]', 
                                 range=[0,1.0], row=1, col=demand_i+1,
                                 **shared_ax_props)

            if demand_i == 0:
                fig.update_yaxes(title_text= f"story", 
                                 range=[0,4], row=1, col=demand_i+1, **shared_ax_props)
            else:
                fig.update_yaxes(range=[0,4], row=1, col=demand_i+1, **shared_ax_props)
    
fig.update_layout(
    title = f'intensity level {stripe} ~ 475 yr return period',
    height=500,
    width=900,
    plot_bgcolor = 'white'
)

fig.show()


# ### Sample demand distribution

# In[11]:


# initialize a pelicun Assessment
PAL = Assessment({"PrintLog": True, "Seed": 415,})

# load the demand model
PAL.demand.load_model({'marginals': stripe_demands})


# In[12]:


# choose a sample size for the analysis
sample_size = 10000

PAL.demand.generate_sample({"SampleSize": sample_size})


# ### Extend sample

# In[13]:


# this call to a save function is needed to perform the unit conversions
demand_sample = PAL.demand.save_sample()

demand_sample.head()


# In[14]:


# We'll need residual drift and the Sa(T=1.13s) sample for the assessment

# Residual drifts could come from nonlinear analysis, but they are often not available or not robust enough
# pelicun provides a convenience function to convert PID to PRD; we'll use that now
delta_y = 0.0035 # yield drift ratio based on FEMA P58 Vol 2 4.7.3 - this leads to excessive irreparable damage
delta_y = 0.0075 # we use this higher drift based on FEMA P58 Vol 1 Table C-2
PID = demand_sample['PID']

RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y})
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

# The Sa(T) can be easily added as a new column to the demand sample. 
# Note that Sa values are identical across all realizations because we are running the analysis for one stripe.
# We assign the Sa values to direction 1
Sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756] # based on FEMA P58 background documentation
demand_sample_ext[('SA_1.13',0,1)] = Sa_vals[int(stripe)-1]

demand_sample_ext.describe().T


# In[15]:


# plot two demands from the sample

demands = ['PID-1-1', 'RID-1-1']

fig = go.Figure()


fig.add_trace(go.Scatter(
    x = demand_sample_ext.loc[:, tuple(demands[0].split('-'))],
    y = demand_sample_ext.loc[:, tuple(demands[1].split('-'))],
    hovertext = demand_sample_ext.index,
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.025)
))

shared_ax_props = dict(
    showgrid = True,
    linecolor = 'black',
    gridwidth = 0.05,
    gridcolor = 'rgb(192,192,192)',
    type = 'log'
)

if 'PFA' in demands[0]:
    fig.update_xaxes(title_text=f'acceleration [g]<br>{demands[0]}', 
                     range=np.log10([0.001,1.5]),
                     **shared_ax_props)

else:
    fig.update_xaxes(title_text=f'drift ratio<br>{demands[0]}', 
                     range=np.log10([0.001,0.1]),
                     **shared_ax_props)
    
if 'PFA' in demands[1]:
    fig.update_yaxes(title_text=f'{demands[1]}<br>acceleration [g]', 
                     range=np.log10([0.0001,1.5]),
                     **shared_ax_props)
    
else:
    fig.update_yaxes(title_text=f'{demands[1]}<br>drift ratio', 
                     range=np.log10([0.0001,0.1]),
                     **shared_ax_props)
    
    
fig.update_layout(
    title = f'demand sample',
    height=600,
    width=650,
    plot_bgcolor = 'white'
)

fig.show()


# In[16]:


# load the extended sample to pelicun

# add units to the data 
demand_sample_ext.T.insert(0, 'Units',"")

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
demand_sample_ext.loc['Units',['PID', 'RID']] = 'rad'

demand_sample_ext

PAL.demand.load_sample(demand_sample_ext)


# In[17]:


# demands are ready, we can move on to damage calculation


# ## Damage

# ![alt text](Figures/pelicun_LET.png "Live Expert Tips - Pelicun3 in Jupyter")

# ### Component configuration
# * This could be prepared in Excel and saved in a csv file or it could be prepared as part of this script. We are using a csv file here for the sake of brevity
# * The table below is an efficient way to assign component quantities (Theta_0) to Performance Groups (PG). A PG is a group of components at a given floor (location) and direction that is affected by the same demand (EDP) values. Each row in the table can assign more than one PG.
# * Zero ("0") stands for Not Applicable in the location and direction column. It is used in location to represent components with a general effect that cannot be linked to a particular floor (e.g., collapse). In directions, it is used to assign non-directional components.
# * The index refers to the component ID in FEMA P58, but it can be any arbitrary string that has a corresponding entry in the fragility database.
# * Blocks are the number of independent units within a Performance Group.
# * Any units can be used that are compatible with the component type. (e.g., ft2, inch2, m2 are all valid)
# * The last three components are custom fragilities that we will define below

# In[18]:


# First, we need a component configuration
cmp_marginals = pd.read_csv('./CMP_marginals.csv', index_col=0)

cmp_marginals


# In[19]:


# we can load the model into pelicun now

# to make the convenience keywords work in the model, we need to specify the number of stories
PAL.stories = 4

# now load the model
PAL.asset.load_cmp_model({'marginals': cmp_marginals})


# Notice in the table below that we could have assigned uncertain component quantities by adding a "Family" and "Theta_1" column to describe their distribution. Truncation limits allow for bounded component quantity distributions that is especially useful when the distribution family is "normal".
# 
# Our input in this example describes a deterministic configuration by leaving those fields blank.

# In[20]:


# let's take a look at the generated marginal parameters

PAL.asset.cmp_marginal_params.loc['B.10.41.002a',:]


# In[21]:


# now, let's generate the component quantity sample - in this case identical values for every realization
PAL.asset.generate_cmp_sample(sample_size)


# In[22]:


# get the component quantity sample - again, use the save function to convert units
cmp_sample = PAL.asset.save_cmp_sample()


# In[23]:


# as expected - we don't have any uncertainty in quantities
# we could edit this sample here and load it back to pelicun similarly to how we did with the demands
cmp_sample


# ### Fragility data
# Pelicun comes with fragility data, including the P58 fragility functions - we will take a look at those first.
# 
# - Pelicun uses the following terminology for damage models:
#     * Fragility functions describe Limit States (LS) that are triggered when a controlling Demand exceeds the Capacity of the component.
#     * The controlling Demand can be Offset in terms of location (e.g., ceilings use acceleration from the floor slab above the floor)
#     * Demand units need to be compatible with the demand (e.g., g, mps2, ftps2 are all okay for accelerations)
#     * The Capacity of the component can be deterministic or probabilistic. A deterministic capacity only requires the assignment of Theta_0 to the limit state. A probabilistic capacity is described by the Family and Theta_1 (i.e., the second parameter) of the capacity distribution.
#     * DamageStateWeights are used to assign more than one mutually exclusive Damage States to a Limit State. Using more than one Damage States allows to have several unique consequences later.
# - The Incomplete flag identifies components that require additional information from the user. Notice that more than a quarter of the components in FEMA P58 have incomplete fragility definitions.
# - We are working on a web-based damage and loss library that will provide a convenient overview of the available fragility and consequence data.

# In[24]:


# the next step is to load the damage model - i.e., fragility functions

P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')

P58_data.head(3)

print(P58_data['Incomplete'].sum(),' incomplete component fragility definitions')


# Let's focus on the incomplete column and check which of the components we want to use have incomplete damage models

# In[25]:


# note that we drop the last two components here (irreparable and collapse) because they are not part of P58
cmp_list = cmp_marginals.index.unique().values[:-3]

P58_data_for_this_assessment = P58_data.loc[cmp_list,:].sort_values('Incomplete', ascending=False)

P58_data_for_this_assessment


# Now take those components that are incomplete, and add the missing information
# 
# Note that the numbers below are just reasonable placeholders. This step would require substantial work from the engineer to review these components and assign the missing values.

# In[26]:


additional_fragility_db = P58_data_for_this_assessment.loc[
    P58_data_for_this_assessment['Incomplete'] == 1].sort_index() 

# D2022.013a, 023a, 023b - Heating, hot water piping and bracing
# dispersion values are missing, we use 0.5
additional_fragility_db.loc[['D.20.22.013a','D.20.22.023a','D.20.22.023b'],
                            [('LS1','Theta_1'),('LS2','Theta_1')]] = 0.5

# D2031.013b - Sanitary Waste piping
# dispersion values are missing, we use 0.5
additional_fragility_db.loc['D.20.31.013b',('LS1','Theta_1')] = 0.5

# D2061.013b - Steam piping
# dispersion values are missing, we use 0.5
additional_fragility_db.loc['D.20.61.013b',('LS1','Theta_1')] = 0.5

# D3031.013i - Chiller
# use a placeholder of 1.5|0.5
additional_fragility_db.loc['D.30.31.013i',('LS1','Theta_0')] = 1.5 #g
additional_fragility_db.loc['D.30.31.013i',('LS1','Theta_1')] = 0.5

# D3031.023i - Cooling Tower
# use a placeholder of 1.5|0.5
additional_fragility_db.loc['D.30.31.023i',('LS1','Theta_0')] = 1.5 #g
additional_fragility_db.loc['D.30.31.023i',('LS1','Theta_1')] = 0.5

# D3052.013i - Air Handling Unit
# use a placeholder of 1.5|0.5
additional_fragility_db.loc['D.30.52.013i',('LS1','Theta_0')] = 1.5 #g
additional_fragility_db.loc['D.30.52.013i',('LS1','Theta_1')] = 0.5

# additional_fragility_db


# Besides the P58 component fragilities, we need to add three new models:
# * excessiveRID is used to monitor residual drifts on every floor in every direction and check if they exceed the capacity assigned to irreparable damage.
# * irreparable is a global limit state that is triggered by having at least one excessive RID and leads to the replacement of the building. It uses a deterministic, placeholder capacity that is sufficiently high so that it will never get triggered by the controlling demand.
# * collapse represents the global collapse limit state that is modeled with a collapse fragility function
# 
# See the description of damage processes below on how excessiveRID and collapse can influence other damages

# In[27]:


# irreparable damage
# this is based on the default values in P58
additional_fragility_db.loc[
    'excessiveRID', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Residual Interstory Drift Ratio', 'rad']   

additional_fragility_db.loc[
    'excessiveRID', [('LS1','Family'),
                    ('LS1','Theta_0'),
                    ('LS1','Theta_1')]] = ['lognormal', 0.01, 0.3]   

additional_fragility_db.loc[
    'irreparable', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']   


# a very high capacity is assigned to avoid damage from demands
additional_fragility_db.loc[
    'irreparable', ('LS1','Theta_0')] = 1e10 

# collapse
# capacity is assigned based on the example in the FEMA P58 background documentation
additional_fragility_db.loc[
    'collapse', [('Demand','Directional'),
                 ('Demand','Offset'),
                 ('Demand','Type'), 
                 ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']   


additional_fragility_db.loc[
    'collapse', [('LS1','Family'),
                 ('LS1','Theta_0'),
                 ('LS1','Theta_1')]] = ['lognormal', 1.35, 0.5]  

# Now we can set the incomplete flag to 0 for the additional components
additional_fragility_db['Incomplete'] = 0

additional_fragility_db


# In[28]:


# now we can load the damage model to pelicun
# note that if there are identical components in the listed sources, 
# the first occurrence is preserved - so, start with the custom data 
# when listing sources and include the default databases in the end

PAL.damage.load_damage_model([
    additional_fragility_db,  # This is the extra fragility data we've just created
    'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
])


# ### Damage Process
# 
# Damage processes are really powerful and can be used to create complex cascading damage models
# 
# The process below means: 
# 
# * If DS1 of the collapse fragility is triggered (i.e., the building collapsed), then the damage for all other components should not be evaluated
# * If excessiveRID is triggered (i.e., the residual drifts are larger than the prescribed capacity), then the irreparable damage state should be triggered

# In[29]:


# we need to define a damage process that represents the logic of FEMA P58

# FEMA P58 uses the following logic:
dmg_process = {
    "1_collapse": {
        "DS1": "ALL_NA"
    },
    "2_excessiveRID": {
        "DS1": "irreparable_DS1"
    }
}

dmg_process
# ### Damage calculation

# In[30]:


# Now we can run the calculation
#PAL.damage.calculate(sample_size, dmg_process=dmg_process)
PAL.damage.calculate(dmg_process=dmg_process)


# In[31]:


# let's take a look at the results
damage_sample = PAL.damage.save_sample()

damage_sample


# In[32]:


#/damage_sample.describe().loc['max',:]

damage_sample.describe().T.loc['C.10.11.001a',:]


# ## Losses - repair consequences

# ![alt text](Figures/pelicun_LET.png "Live Expert Tips - Pelicun3 in Jupyter")

# ### Consequence mapping to damages
# 
# Consequences are decoupled from fragilities in pelicun to enforce and encourgae a modular approach to performance assessment.
# 
# The map that we prepare below describes which type of damage leads to which type of consequences. With FEMA P58 this is quite simple because the names of the fragility and consequence models are identical - note that we would have the option to link different ones though. Also, several fragilities in P58 have identical consequences that leads to redundant data. 
# 
# We plan to introduce a database that is a more concise and streamlined version of the one provided in FEMA P58 and encourage researchers to extend it by providing data to the incomplete components.

# In[33]:


# let us prepare the map based on the component list

# all of the drivers are damage quantities (as opposed to EDP or IM intensities)
# so we need to prepend 'DMG-' to the component names to tell pelicun to look for the damage of these components
drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
drivers = drivers[:-3]+drivers[-2:]

# we are looking at repair consequences in this example
# the components in P58 have consequence models under the same name
loss_models = cmp_marginals.index.unique().tolist()[:-3]

# Both irreparable damage and collapse lead to the replacement of the building. 
# Hence, we can use the same consequence model for both types of damages
# We will define the replacement consequence in the following cell.
loss_models+=['replacement',]*2

# Assemble the DataFrame with the mapping information
# The column name identifies the type of the consequence model.
loss_map = pd.DataFrame(loss_models, columns=['BldgRepair'], index=drivers)

loss_map


# ### Consequence data

# Now we need to check if the repair consequence models for the components in this building are complete in FEMA P58. 27 components in P58 only have damage models and do not have repair consequence models at all. All of the other models are complete. This example only includes components with complete consequence information.
# 
# Note that we have both cost and time consequences listed below. 
# 
# Consequences are organized around Damage States using the same structure to assign probabilistic values as before. The only exception is Theta_0 where the "val1,val2|q1,q2" option is available to assign a multilinear median function instead of a single median value. q1 and q2 are the controlling quantities, while val1 and val2 are the corresponding consequence median values. This type of assignment defines the sloped function used in FEMA P58 to consider economies of scale. Note that the number of values is not limited at 2, you can assign a multilinear function with as many values as necessary for your component.

# In[34]:


# Let us load the consequence models first
P58_data = PAL.get_default_data('bldg_repair_DB_FEMA_P58_2nd')

cmp_list = cmp_marginals.index.unique().values[:-3]

P58_data_for_this_assessment = P58_data.loc[cmp_list,:]

print(P58_data_for_this_assessment['Incomplete'].sum(), ' components have incomplete consequence models assigned.')

P58_data_for_this_assessment


# Now we need to define the replacement consequence.
# 
# Note that
# - We efficiently use the same consequence for the collapse and irreparable damages
# - We could consider uncertainty in the replacement cost/time with this approach. We are not going to do that now for the sake of simplicity

# In[35]:


# initialize the dataframe
additional_consequences = pd.DataFrame(
    columns = pd.MultiIndex.from_tuples([
        ('Incomplete',''), ('Quantity','Unit'), ('DV', 'Unit'), ('DS1', 'Theta_0')]),
    index=pd.MultiIndex.from_tuples([
        ('replacement','Cost'), ('replacement','Time')])
)

additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA', 'USD_2011', 21600000]
additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA', 'worker_day', 8400]  
# both values are based on the example in the background documentation of FEMA P58 
# replacement time: 400 days x 0.001 worker/ft2 x 21600 ft2

additional_consequences


# In[36]:


# Now we can load the loss model to pelicun
PAL.bldg_repair.load_model(
    [additional_consequences,
     "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
    loss_map)


# ### Loss calculation

# In[37]:


# and run the calculations
PAL.bldg_repair.calculate()


# In[38]:


# let's take a look at the losses
loss_sample = PAL.bldg_repair.sample

loss_sample


# In[39]:


loss_sample.groupby(level=[0,1,2], axis=1).sum().describe()


# Aggregating losses is not always trivial - e.g., repair times have sequential and parallel options

# In[40]:


agg_DF = PAL.bldg_repair.aggregate_losses()

agg_DF.describe()


# In[41]:


sns.set(font_scale=2)
sns.set_style('whitegrid')

agg_DF.describe([0.01,0.1,0.5,0.9,0.99]).T

# get only the results that describe repairable damage
agg_DF_ = agg_DF.loc[agg_DF['repair_cost'] < 2.0e7]

print('\nProbability of experiencing repairable damage:', agg_DF_.shape[0] / agg_DF.shape[0])

sns.histplot(data=agg_DF_, x='repair_cost')

sns.jointplot(data=agg_DF_/20, x=('repair_time','parallel'), y=('repair_time','sequential'),
              xlim=[0,100], ylim=[0,200], alpha=0.1, height=10)


# # Concise version

# In[42]:


#parameters
sample_size = 10000
delta_y = 0.0075
stripe = '3'


# ## Pre-processing

# In[43]:


# prepare demand input
raw_demands = convert_to_MultiIndex(pd.read_csv('demand_data.csv', index_col=0), axis=0)

# prepare the demand input for pelicun
stripe_demands = raw_demands.loc[stripe,:]

# units - - - - - - - - - - - - - - - - - - - - - - - -  
stripe_demands.insert(0, 'Units',"")
stripe_demands.loc['PFA','Units'] = 'g'
stripe_demands.loc['PID','Units'] = 'rad'

# distribution family  - - - - - - - - - - - - - - - - -  
stripe_demands.insert(1, 'Family',"")
stripe_demands['Family'] = 'lognormal'

# distribution parameters  - - - - - - - - - - - - - - -
stripe_demands.rename(columns = {'median': 'Theta_0'}, inplace=True)
stripe_demands.rename(columns = {'log_std': 'Theta_1'}, inplace=True)

# prepare additional fragility and consequence data ahead of time
cmp_marginals = pd.read_csv('CMP_marginals_LET.csv', index_col=0)

# add missing data to P58 damage model
P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')
cmp_list = cmp_marginals.index.unique().values[:-3]

# now take those components that are incomplete, and add the missing information
additional_fragility_db = P58_data.loc[cmp_list,:].loc[P58_data.loc[cmp_list,'Incomplete'] == 1].sort_index()

# D2022.013a, 023a, 023b - Heating, hot water piping and bracing
# dispersion values are missing, we use 0.5
additional_fragility_db.loc[['D2022.013a','D2022.023a','D2022.023b'],[('LS1','Theta_1'),('LS2','Theta_1')]] = 0.5

# D2031.013b - Sanitary Waste piping
# dispersion values are missing, we use 0.5
additional_fragility_db.loc['D2031.013b',('LS1','Theta_1')] = 0.5

# D2061.013b - Steam piping
# dispersion values are missing, we use 0.5
additional_fragility_db.loc['D2061.013b',('LS1','Theta_1')] = 0.5

# D3031.013i - Chiller
# use a placeholder of 3.0|0.5
additional_fragility_db.loc['D3031.013i',('LS1','Theta_0')] = 3.0
additional_fragility_db.loc['D3031.013i',('LS1','Theta_1')] = 0.5

# D3031.023i - Cooling Tower
# use a placeholder of 3.0|0.5
additional_fragility_db.loc['D3031.023i',('LS1','Theta_0')] = 3.0
additional_fragility_db.loc['D3031.023i',('LS1','Theta_1')] = 0.5

# D3052.013i - Air Handling Unit
# use a placeholder of 3.0|0.5
additional_fragility_db.loc['D3052.013i',('LS1','Theta_0')] = 3.0
additional_fragility_db.loc['D3052.013i',('LS1','Theta_1')] = 0.5

# prepare the extra damage models for collapse and irreparable damage
additional_fragility_db.loc[
    'excessiveRID', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Residual Interstory Drift Ratio', 'rad']   

additional_fragility_db.loc[
    'excessiveRID', [('LS1','Family'),
                    ('LS1','Theta_0'),
                    ('LS1','Theta_1')]] = ['lognormal', 0.01, 0.3]   

additional_fragility_db.loc[
    'irreparable', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']   

additional_fragility_db.loc[
    'irreparable', ('LS1','Theta_0')] = 1e10

additional_fragility_db.loc[
    'collapse', [('Demand','Directional'),
                 ('Demand','Offset'),
                 ('Demand','Type'), 
                 ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']   

additional_fragility_db.loc[
    'collapse', [('LS1','Family'),
                 ('LS1','Theta_0'),
                 ('LS1','Theta_1')]] = ['lognormal', 1.35, 0.5]  

# Now we can set the incomplete flag to 0 for these components
additional_fragility_db['Incomplete'] = 0

# create the additional consequence models
additional_consequences = pd.DataFrame(
    columns = pd.MultiIndex.from_tuples([
        ('Incomplete',''), ('Quantity','Unit'), ('DV', 'Unit'), ('DS1', 'Theta_0')]),
    index=pd.MultiIndex.from_tuples([
        ('replacement','Cost'), ('replacement','Time')])
)

additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA', 'US$_2011', 21600000]
additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA', 'worker_day', 8400]


# ## Demands

# In[44]:


# initialize a pelicun Assessment
PAL = Assessment({"PrintLog": True, "Seed": 415,})

# load the demand model
PAL.demand.load_model({'marginals': stripe_demands})

# generate samples
PAL.demand.generate_sample({"SampleSize": sample_size})

# add residual drift and Sa
demand_sample = PAL.demand.save_sample()

RID = PAL.demand.estimate_RID(demand_sample['PID'], {'yield_drift': delta_y})
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

Sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756]
demand_sample_ext[('SA_1.13',0,1)] = Sa_vals[int(stripe)-1]

# add units to the data 
demand_sample_ext.T.insert(0, 'Units',"")

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
demand_sample_ext.loc['Units',['PID', 'RID']] = 'rad'

PAL.demand.load_sample(demand_sample_ext)


# ## Damage

# In[45]:


# specify number of stories
PAL.stories = 4

# load component definitions
cmp_marginals = pd.read_csv('CMP_marginals_LET.csv', index_col=0)
PAL.asset.load_cmp_model({'marginals': cmp_marginals})

# generate sample
PAL.asset.generate_cmp_sample(sample_size)

# load the models into pelicun
PAL.damage.load_damage_model([
    additional_fragility_db,  # This is the extra fragility data we've just created
    'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
])

# prescribe the damage process
dmg_process = {
    "1_collapse": {
        "DS1": "ALL_NA"
    },
    "2_excessiveRID": {
        "DS1": "irreparable_DS1"
    }
}

# calculate damages
PAL.damage.calculate(sample_size, dmg_process=dmg_process)


# ## Losses

# In[46]:


# create the loss map
drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
drivers = drivers[:-3]+drivers[-2:]

loss_models = cmp_marginals.index.unique().tolist()[:-3] +['replacement',]*2

loss_map = pd.DataFrame(loss_models, columns=['BldgRepair'], index=drivers)

# load the loss model
PAL.bldg_repair.load_model(
    [additional_consequences,
     "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
    loss_map)

# perform the calculation
PAL.bldg_repair.calculate(sample_size)

# get the aggregate losses
agg_DF = PAL.bldg_repair.aggregate_losses()

agg_DF


# # Acknowledgments
# 
# This material is based upon work supported by the National Science Foundation under Grants No. 1612843 & 2131111. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

# In[ ]:




