import sys
import numpy as np
import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30
import pprint
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment


PAL = Assessment({
    "PrintLog": True, 
    "Seed": 415,
    "Verbose": False,
})

PAL.demand.load_sample('demand_data.csv')
PAL.demand.calibrate_model(
    {
        "ALL": {
            "DistributionFamily": "lognormal"
        },
        "PID": {
            "DistributionFamily": "lognormal",
            "TruncateLower": "",
            "TruncateUpper": "0.06"
        }
    }
)
# jump to the def of calibrate_model for info on the configuration dict

# all the rest is the same >>>>>>>>>>>>>>>>>>>>>

# In[11]:


# choose a sample size for the analysis
sample_size = 10000

# generate demand sample
PAL.demand.generate_sample({"SampleSize": sample_size})

# extract the generated sample
# Note that calling the save_sample() method is better than directly pulling the 
# sample attribute from the demand object because the save_sample method converts
# demand units back to the ones you specified when loading in the demands.
demand_sample = PAL.demand.save_sample()

demand_sample.head()


# ### 3.2.4 Extend the sample
# 
# The damage and loss models we use later in this example need residual drift and spectral acceleration [Sa(T=1.13s)] information for each realizations. The residual drifts are used to consider irreparable damage to the building; the spectral accelerations are used to evaluate the likelihood of collapse.
# 
# **Residual drifts*
# 
# Residual drifts could come from nonlinear analysis, but they are often not available or not robust enough. Pelicun provides a convenience method to convert PID to RID and we use that function in this example. Currently, the method implements the procedure recommended in FEMA P-58, but it is designed to support multiple approaches for inferring RID from available demand information. 
# 
# The FEMA P-58 RID calculation is based on the yield drift ratio. There are conflicting data in FEMA P-58 on the yield drift ratio that should be applied for this building: 
# * According to Vol 2 4.7.3, $\Delta_y = 0.0035$ , but this value leads to excessive irreparable drift likelihood that does not match the results in the background documentation.
# * According to Vol 1 Table C-2, $\Delta_y = 0.0075$ , which leads to results that are more in line with those in the background documentation.
# 
# We use the second option below. Note that we get a different set of residual drift estimates for every floor of the building.
# 
# **Spectral acceleration**
# 
# The Sa(T) can be easily added as a new column to the demand sample. Note that Sa values are identical across all realizations because we are running the analysis for one stripe that has a particular Sa(T) assigned to it. We assign the Sa values to direction 1 and we will make sure to have the collapse fragility defined as a directional component (see Damage/Fragility data) to avoid scaling these spectral accelerations with the nondirectional scale factor.
# 
# The list below provides Sa values for each stripe from the analysis - the values are from the background documentation referenced in the Introduction. 

# In[12]:


# get residual drift estimates 
delta_y = 0.0075
PID = demand_sample['PID']

RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y}) * 0.0000001

# and join them with the demand_sample
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

# add spectral acceleration
Sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756]
demand_sample_ext[('SA_1.13',0,1)] = 0.000000001

demand_sample_ext.describe().T


# The plot below illustrates that the relationship between a PID and RID variable is not multivariate lognormal. This underlines the importance of generating the sample for such additional demand types realization-by-realization rather than adding a marginal RID to the initial set and asking Pelicun to sample RIDs from a multivariate lognormal distribution.
# 
# You can use the plot below to display the joint distribution of any two demand variables

# In[13]:


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



# ### 3.2.5 Load Demand Samples
# 
# The script below adds unit information to the sample data and loads it to Pelicun. 
# 
# Note that you could skip the first part of this demand calculation and prepare a demand sample entirely by yourself. That allows you to consider any kind of distributions and any kind of correlation structure between the demands. As long as you have the final list of realizations formatted according to the conventions explained above, you should be able to load it directly to Pelicun.

# In[14]:


# add units to the data 
demand_sample_ext.T.insert(0, 'Units',"")

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
demand_sample_ext.loc['Units',['PID', 'RID']] = 'rad'


PAL.demand.load_sample(demand_sample_ext)


# This concludes the Demand section. The demand sample is ready, we can move on to damage calculation

# ## 3.3 Damage
# 
# Damage simulation requires an asset model, fragility data, and a damage process that describes dependencies between damages in the system. We will look at each of these in detail below.
# 
# ### 3.3.1 Define asset model
# 
# The asset model assigns components to the building and defines where they are and how much of each component is at each location. 
# 
# The asset model can consider uncertainties in the types of components assigned and in their quantities. This example does not introduce those uncertainties for the sake of brevity, but they are discussed in other examples. For this example, the component types and their quantities are identical in all realizations.
# 
# Given this deterministic approach, we can take advantage of a convenience method in Pelicun for defining the asset model. We can prepare a table (see the printed data below) where each row identifies a component and assigns some quantity of it to a set of locations and directions. Such a table can be prepared in Excel or in a text editor and saved in a CSV file - like we did in this example, see CMP_marginals.csv - or it could be prepared as part of this script. Storing these models in a CSV file facilitates sharing the basic inputs of an analysis with other researchers.
# 
# The tabular information is organized as follows:
# * Each row in the table can assign component quantities (Theta_0) to one or more Performance Groups (PG). A PG is a group of components at a given floor (location) and direction that is affected by the same demand (EDP or IM) values.
# * The quantity defined under Theta_0 is assigned to each location and direction listed. For example, the first row in the table below assigns 2.0 of B.10.41.001a to the third and fourth floors in directions 1 and 2. That is, it creates 4 Performance Groups, each with 2 of these components in it.
# * Zero ("0") is reserved for "Not Applicable" use cases in the location and direction column. As a location, it represents components with a general effect that cannot be linked to a particular floor (e.g., collapse). In directions, it is used to identify non-directional components.
# * The index in this example refers to the component ID in FEMA P58, but it can be any arbitrary string that has a corresponding entry in the applied fragility database (see the Fragility data section below for more information).
# * Blocks are the number of independent units within a Performance Group. By default (i.e., when the provided value is missing or NaN), each PG is assumed to have one block which means that all of the components assigned to it will have the same behavior. FEMA P-58 identifies unit sizes for its components. We used these sizes to determine the number of indepdendent blocks for each PG. See, for example, B.20.22.031 that has a 30 ft2 unit size in FEMA P-58. We used a large number of blocks to capture that each of those curtain wall elements can get damaged indepdendently of the others.
# * Component quantities (Theta_0) can be provided in any units compatible with the component type. (e.g., ft2, inch2, m2 are all valid)
# * The last three components use custom fragilities that are not part of the component database in FEMA P-58. We use these to consider irreparable damage and collapse probability. We will define the corresponding fragility and consequence functions in later sections of this example.
# * The Comment column is not used by Pelicun, any text is acceptable there.

# In[15]:


# load the component configuration
cmp_marginals = pd.read_csv('CMP_marginals.csv', index_col=0)

print("...")
cmp_marginals.tail(10)


# In[16]:


# to make the convenience keywords work in the model, 
# we need to specify the number of stories
PAL.stories = 4

# now load the model into Pelicun
PAL.asset.load_cmp_model({'marginals': cmp_marginals})


# Note that we could have assigned uncertain component quantities by adding a "Family" and "Theta_1", "Theta_2" columns to describe their distribution. Additional "TruncateLower" and "TruncateUpper" columns allow for bounded component quantity distributions that is especially useful when the distribution family is supported below zero values.
# 
# Our input in this example describes a deterministic configuration resulting in the fairly simple table shown below.

# In[17]:


# let's take a look at the generated marginal parameters
PAL.asset.cmp_marginal_params.loc['B.10.41.002a',:]


# ### 3.3.2 Sample asset distribution
# 
# In this example, the quantities are identical for every realization. We still need to generate a component quantity sample because the calculations in Pelicun expect an array of component quantity realizations. The sample size for the component quantities is automatically inferred from the demand sample. If such a sample is not available, you need to provide a sample size as the first argument of the generate_cmp_sample method.
# 
# The table below shows the statistics for each Performance Group's quantities. Notice the zero standard deviation and that the minimum and maximum values are identical - this confirms that the quantities are deterministic.
# 
# We could edit this sample and load the edited version back to Pelicun like we did for the Demands earlier.

# In[18]:


# Generate the component quantity sample
PAL.asset.generate_cmp_sample()

# get the component quantity sample - again, use the save function to convert units
cmp_sample = PAL.asset.save_cmp_sample()

cmp_sample.describe()


# ### 3.3.3 Define component fragilities
# 
# Pelicun comes with fragility data, including the FEMA P-58 component fragility functions. We will start with taking a look at those data first.
# 
# Pelicun uses the following terminology for fragility data:
# - Each Component has a number of pre-defined Limit States (LS) that are triggered when a controlling Demand exceeds the Capacity of the component. 
# - The type of controlling Demand can be any of the demand types supported by the tool - see the list of types in the Demands section of this example.
# - Units of the controlling Demand can be chosen freely, as long as they are compatible with the demand type (e.g., g, mps2, ftps2 are all acceptable for accelerations, but inch and m are not)
# - The controlling Demand can be Offset in terms of location (e.g., ceilings use acceleration from the floor slab above the floor) by providing a non-zero integer in the Offset column.
# - The Capacity of a component can be either deterministic or probabilistic. A deterministic capacity only requires the assignment of Theta_0 to the limit state. A probabilistic capacity is described by a Fragility function. Fragility functions use Theta_0 as well as the Family and Theta_1 (i.e., the second parameter) to define a distribution function for the random capacity variable.
# - When a Limit State is triggered, the Component can end up in one or more Damage States. DamageStateWeights are used to assign more than one mutually exclusive Damage States to a Limit State. Using more than one Damage States allows us to recognize multiple possible damages and assign unique consequences to each damage in the loss modeling step.
# - The Incomplete flag identifies components that require additional information from the user. More than a quarter of the components in FEMA P-58 have incomplete fragility definitions. If the user does not provide the missing information, Pelicun provides a warning message and skips Incomplete components in the analysis.
# 
# The SimCenter is working on a web-based damage and loss library that will provide a convenient overview of the available fragility and consequence data. Until then, the get_default_data method allows you to pull any of the default fragility datasets from Pelicun and review/edit/reload the data.

# In[19]:


# review the damage model - in this example: fragility functions
P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')


print(P58_data['Incomplete'].sum(),' incomplete component fragility definitions')


# Let's focus on the incomplete column and check which of the components we want to use have incomplete damage models. We do this by filtering the component database and only keeping those components that are part of our asset model and have incomplete definitions.

# In[20]:


# note that we drop the last three components here (excessiveRID, irreparable, and collapse) 
# because they are not part of P58
cmp_list = cmp_marginals.index.unique().values[:-3]

P58_data_for_this_assessment = P58_data.loc[cmp_list,:].sort_values('Incomplete', ascending=False)

additional_fragility_db = P58_data_for_this_assessment.loc[
    P58_data_for_this_assessment['Incomplete'] == 1].sort_index() 

additional_fragility_db


# The component database bundled with Pelicun includes a CSV file and a JSON file for each dataset. The CSV file contains the data required to perform the calculations; the JSON file provides additional metadata for each component. The get_default_metadata method in Pelicun provides convenient access to this metadata. Below we demonstrate how to pull in the data on the first incomplete component. The metadata in this example are directly from FEMA P-58.

# In[21]:


P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')

pprint.pprint(P58_metadata['D.20.22.013a'])


# We need to add the missing information to the incomplete components.
# 
# Note that the numbers below are just reasonable placeholders. This step would require substantial work from the engineer to review these components and assign the missing values. Such work is out of the scope of this example.
# 
# The table below shows the completed fragility information.

# In[22]:


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

# We can set the incomplete flag to 0 for these components
additional_fragility_db['Incomplete'] = 0

additional_fragility_db


# Now we need to add three new components:
# * **excessiveRID** is used to monitor residual drifts on every floor in every direction and check if they exceed the capacity assigned to irreparable damage.
# * **irreparable** is a global limit state that is triggered by having at least one excessive RID and leads to the replacement of the building. This triggering requires one component to affect another and it is handled in the Damage Process section below. For its individual damage evaluation, this component uses a deterministic, placeholder capacity that is sufficiently high so that it will never get triggered by the controlling demand.
# * **collapse** represents the global collapse limit state that is modeled with a collapse fragility function and uses spectral acceleration at the dominant vibration period as the demand. Multiple collapse modes could be considered by assigning a set of Damage State weights to the collapse component.
# 
# The script in this cell creates the table shown below. We could also create such information in a CSV file and load it to the notebook.

# In[23]:



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

# We set the incomplete flag to 0 for the additional components
additional_fragility_db['Incomplete'] = 0

additional_fragility_db.tail(3)


# ### 3.3.4 Load component fragility data
# 
# Now that we have the fragility data completed and available for all components in the asset model, we can load the data to the damage model in Pelicun.
# 
# When providing custom data, you can directly provide a DataFrame like we do in this example (additional_fragility_db), or you can provide a path to a CSV file that is structured like the table we prepared above.
# 
# Default databases are loaded using the keyword "PelicunDefault" in the path and then providing the name of the database. The PelicunDefault keyword is automatically replaced with the path to the default component data directory. 
# 
# Note that there are identical components in the listed sources. The additional_fragility_db contains the additional global components (e.g., collapse) and the ones that are incomplete in FEMA P-58. The latter ones are also listed in the default FEMA P-58 database. Such conflicts are resolved by preserving the first occurrence of every component. Hence, always start with the custom data when listing sources and add default databases in the end.

# In[24]:


PAL.damage.load_damage_model([
    additional_fragility_db,  # This is the extra fragility data we've just created
    'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
])


# ### 3.3.5 Damage Process
# 
# Damage processes are a powerful new feature in Pelicun 3. They are used to connect damages of different components in the performance model and they can be used to create complex cascading damage models.
# 
# The default FEMA P-58 damage process is farily simple. The process below can be interpreted as follows:
# * If Damage State 1 (DS1) of the collapse component is triggered (i.e., the building collapsed), then damage for all other components should be cleared from the results. This considers that component damages (and their consequences) in FEMA P-58 are conditioned on no collapse.
# * If Damage State 1 (DS1) of any of the excessiveRID components is triggered (i.e., the residual drifts are larger than the prescribed capacity on at least one floor), then the irreparable component should be set to DS1.

# In[25]:


# FEMA P58 uses the following process:
dmg_process = {
    "1_collapse": {
        "DS1": "ALL_NA"
    },
    "2_excessiveRID": {
        "DS1": "irreparable_DS1"
    }
}


# ### 3.3.6 Damage calculation
# 
# Damage calculation in Pelicun requires 
# - a pre-assigned set of component fragilities;
# - a pre-assigned sample of component quantities; 
# - a pre-assigned sample of demands;
# - and an (optional) damage process
# 
# The sample size for the damage calculation is automatically inferred from the demand sample size.
# 
# **Expected Runtime & Best Practices**
# 
# The output below shows the total number of Performance Groups (121) and Component Blocks (1736). The number of component blocks is a good proxy for the size of the problem. Damage calculation is the most demanding part of the performance assessment workflow. The runtime for damage calculations in Pelicun scales approximately linearly with the number of component blocks above 500 blocks and somewhat better than linearly with the sample size above 10000 samples. Below 10000 sample size and 500 blocks, the overhead takes a substantial part of the approximately few second calculation time. Below 1000 sample size and 100 blocks, these variables have little effect on the runtime.
# 
# Pelicun can handle failry large problems but it is ideal to make sure both the intermediate data and the results fit in the RAM of the system. Internal calculations are automatically disaggregated to 1000-block batches at a time to avoid memory-related issues. This might still be too large of a batch if the number of samples is more than 10,000. You can manually adjust the batch size using the block_batch_size argument in the calculate method below. We recommend using only 100-block batches when running a sample size of 100,000. Even larger sample sizes coupled with a complex model probably benefit from running in batches across the sample. Contact the SimCenter if you are interested in such large problems and we are happy to provide support.
# 
# Results are stored at a Performance Group (rather than Component Block) resolution to allow users to run larger problems. The size of the output data is proportional to the number of Performance Groups x number of active Damage States per PG x sample size. Modern computers with 64-bit memory addressing and 4+ GB of RAM should be able to handle problems with up to 10,000 performance groups and a sample size of 10,000. This limit shall be sufficient for even the most complex and high resolution models of a single building - note in the next cell that the size of the results from this calculation (121 PG x 10,000 realizations) is just 30 MB.

# In[26]:


# Now we can run the calculation
PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100) #- for large calculations


# ### 3.3.7 Damage estimates
# 
# Below, we extract the damage sample from Pelicun and show a few example plots to illustrate how rich information this data provides about the damages in the building

# In[27]:


damage_sample = PAL.damage.save_sample()

print("Size of damage results: ", sys.getsizeof(damage_sample)/1024/1024, "MB")


# **Damage statistics of a component type**
# 
# The table printed below shows the mean, standard deviation, minimum, 10th, 50th, and 90th percentile, and maximum quantity of the given component in each damage state across various locations and directions in the building.

# In[28]:


component = 'B.20.22.031'
damage_sample.describe([0.1, 0.5, 0.9]).T.loc[component,:].head(30)


# In[29]:


dmg_plot = damage_sample.loc[:, component].groupby(level=[0,2], axis=1).sum().T

px.bar(x=dmg_plot.index.get_level_values(1), y=dmg_plot.mean(axis=1), 
       color=dmg_plot.index.get_level_values(0),
       barmode='group',
       labels={
           'x':'Damage State',
           'y':'Component Quantity [ft2]',
           'color': 'Floor'
       },
       title=f'Mean Quantities of component {component} in each Damage State',
       height=500
      )


# In[30]:


dmg_plot = (damage_sample.loc[:, component].loc[:,idx[:,:,'2']] / 
            damage_sample.loc[:, component].groupby(level=[0,1], axis=1).sum()).T

px.bar(x=dmg_plot.index.get_level_values(0), y=(dmg_plot>0.5).mean(axis=1), 
       color=dmg_plot.index.get_level_values(1),
       barmode='group',
       labels={
           'x':'Floor',
           'y':'Probability',
           'color': 'Direction'
       },
       title=f'Probability of having more than 50% of component {component} in DS2',
       height=500
      )


# In[31]:


dmg_plot = (damage_sample.loc[:, component].loc[:,idx[:,:,'2']].groupby(level=[0], axis=1).sum() / 
            damage_sample.loc[:, component].groupby(level=[0], axis=1).sum()).T      

px.scatter(x=dmg_plot.loc['1'], y=dmg_plot.loc['2'], 
           color=dmg_plot.loc['3'],
           opacity=0.1,
           color_continuous_scale = px.colors.diverging.Portland,
           marginal_x ='histogram', marginal_y='histogram',
           labels={
               'x':'Proportion in DS2 in Floor 1',
               'y':'Proportion in DS2 in Floor 2',
               'color': 'Proportion in<br>DS2 in Floor 3'
           },
           title=f'Correlation between component {component} damages across three floors',
           height=600, width=750)


# In[32]:


dmg_plot = 1.0 - (damage_sample.groupby(level=[0,3], axis=1).sum() / 
                  damage_sample.groupby(level=[0], axis=1).sum()
                 ).loc[:, idx[:,'0']]
 
dmg_plot = dmg_plot.iloc[:,:-3]
dmg_plot.columns = dmg_plot.columns.get_level_values(0) 
    
px.box(y=np.tile(dmg_plot.columns, dmg_plot.shape[0]), 
       x=dmg_plot.values.flatten(), 
       color = [c[0] for c in dmg_plot.columns]*dmg_plot.shape[0],
       orientation = 'h',
       labels={
           'x':'Proportion in DS1 or higher',
           'y':'Component ID',
           'color': 'Component Group'
       },
       title=f'Range of normalized damaged quantities by component type',
       height=1500)


# In[33]:


# print('Probability of collapse: ', damage_sample[('collapse','0','1','1')].mean())
# print()
# print('Probability of irreparable damage: ', damage_sample[('irreparable','0','1','1')].mean())


# ## 3.4 Losses - repair consequences
# 
# Loss simulation is an umbrella term that can include the simulation of various types of consequences. In this example we focus on repair cost and repair time consequences. Pelicun provides a flexible framework that can be expanded with any aribtrary decision variable. Let us know if you need a particular decision variable for your work that would be good to support in Pelicun.
# 
# Losses can be either based on consequence functions controlled by the quantity of damages, or based on loss functions controlled by demand intensity. Pelicun supports both approaches and they can be mixed within the same analysis; in this example we use consequence functions following the FEMA P-58 methodology.
# 
# Loss simulation requires a demand/damage sample, consequence/loss function data, and a mapping that links the demand/damage components to the consequence/loss functions. The damage sample in this example is already available from the previous section. We will show below how to prepare the mapping matrix and how to load the consequence functions.
# 
# ### 3.4.1 Consequence mapping to damages
# 
# Consequences are decoupled from damages in pelicun to enforce and encourgae a modular approach to performance assessment.
# 
# The map that we prepare below describes which type of damage leads to which type of consequence. With FEMA P-58 this is quite straightforward because the IDs of the fragility and consequence data are identical - note that we would have the option to link different ones though. Also, several fragilities in P58 have identical consequences and the approach in Pelicun will allow us to remove such redundancy in future datasets.  We plan to introduce a database that is a more concise and streamlined version of the one provided in FEMA P58 and encourage researchers to extend it by providing data to the incomplete components.
# 
# The mapping is defined by a table (see the example below). Each row has a demand/damage ID and a list of consequence IDs, one for each type of decision variable. Here, we are looking at building repair consequences only, hence, there is only one column with consequence IDs. The IDs of FEMA P-58 consequence functions are identical to the name of the components they are assigned to. Damage sample IDs in the index of the table are preceded by 'DMG', while demand sample IDs would be preceded by 'DEM'.
# 
# Notice that besides the typical FEMA P-58 IDs, the table also includes 'DMG-collapse' and 'DMG-irreparable' to capture the consequences of those events. Both irreparable damage and collapse lead to the replacement of the building. Consequently, we can use the same consequence model (called 'replacement') for both types of damages. We will define what the replacement consequence is in the next section.

# In[34]:


# let us prepare the map based on the component list

# we need to prepend 'DMG-' to the component names to tell pelicun to look for the damage of these components
drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
drivers = drivers[:-3]+drivers[-2:]

# we are looking at repair consequences in this example
# the components in P58 have consequence models under the same name
loss_models = cmp_marginals.index.unique().tolist()[:-3]

# We will define the replacement consequence in the following cell.
loss_models+=['replacement',]*2

# Assemble the DataFrame with the mapping information
# The column name identifies the type of the consequence model.
loss_map = pd.DataFrame(loss_models, columns=['BldgRepair'], index=drivers)

loss_map


# ### 3.4.2 Define component consequence data
# 
# Pelicun comes with consequence data, including the FEMA P-58 component consequence functions. We will start with taking a look at those data first.
# 
# Pelicun uses the following terminology for consequence data:
# - Each Component has a number of pre-defined Damage States (DS)
# - The quantity of each Component in each DS in various locations and direction in the building is provided as a damage sample.
# - The index of the consequence data table can be hierarchical and list several consequence types that belong to the same group. For example, the repair consequences here include 'Cost' and 'Time'; injury consequences include injuries of various severity. Each row in the table corresponds to a combination of a component and a consequence type.
# - Consequences in each damage state can be:
#     * Deterministic: use only the 'Theta_1' column
#     * Probabilistic: provide information on the 'Family', 'Theta_0' and 'Theta_1' to describe the distribution family and its two parameters.
# - The first parameter of the distribution (Theta_0) can be either a scalar or a function of the quantity of damage. This applies to both deterministic and probabilistic cases. When Theta_0 is a function of the quantity of damage, two series of numbers are expected, separated by a '|' character. The two series are used to construct a multilinear function - the first set of numbers are the Theta_0 values, the second set are the corresponding quantities. The functions are assumed to be constant below the minimum and above the maximum quantities.
# - The LongLeadTime column is currently informational only - it does not affect the calculation.
# - The DV-Unit column (see the right side of the table below) defines the unit of the outputs for each consequence function - i.e., the unit of the Theta_0 values.
# - The Quantity-Unit column defines the unit of the damage/demand quantity. This allows mixing fragility and consequence functions that use different units - as long as the units are compatible, Pelicun takes care of the conversions automatically.
# - The Incomplete column is 1 if some of the data is missing from a row.
# 
# The SimCenter is working on a web-based damage and loss library that will provide a convenient overview of the available fragility and consequence data. Until then, the get_default_data method allows you to pull any of the default consequence datasets from Pelicun and review/edit/reload the data.
# 
# After pulling the data, first, we need to check if the repair consequence functions for the components in this building are complete in FEMA P-58. 27 components in FEMA P-58 only have damage models and do not have repair consequence models at all. All of the other models are complete. As you can see from the message below, this example only includes components with complete consequence information.

# In[35]:


# load the consequence models
P58_data = PAL.get_default_data('bldg_repair_DB_FEMA_P58_2nd')

# get the consequences used by this assessment
P58_data_for_this_assessment = P58_data.loc[loss_map['BldgRepair'].values[:-2],:]

print(P58_data_for_this_assessment['Incomplete'].sum(), ' components have incomplete consequence models assigned.')



# **Adding custom consequence functions**
# 
# Now we need to define the replacement consequence for the collapse and irreparable damage cases.
# 
# The FEMA P-58 background documentation provides the \$21.6 million as replacement cost and 400 days as replacement time. The second edition of FEMA P-58 introduced worker-days as the unit of replacement time; hence, we need a replacement time in worker-days. We show two options below to estimate that value:
# - We can use the assumption of 0.001 worker/ft2 from FEMA P-58 multiplied by the floor area of the building to get the average number of workers on a typical day. The total number of worker-days is the product of the 400 days of construction and this average number of workers. Using the plan area of the building for this calculation assumes that one floor is being worked on at a time - this provides a lower bound of the number of workers: 21600 x 0.001 = 21.6. The upper bound of workers is determined by using the gross area for the calculation: 86400 x 0.001 = 86.4. Consequently, the replacement time will be between 8,640 and 34,560 worker-days.
# - The other approach is taking the replacement cost, assuming a ratio that is spent on labor (0.3-0.5 is a reasonable estimate) and dividing that labor cost with the daily cost of a worker (FEMA P-58 estimates \$680 in 2011 USD for the SF Bay Area which we will apply to this site in Los Angeles). This calculation yields 9,529 - 15,882 worker-days depending on the labor ratio chosen.
# 
# Given the above estimates, we use 12,500 worker-days for this example. 
# 
# Note that
# - We efficiently use the same consequence for the collapse and irreparable damages
# - We could consider uncertainty in the replacement cost/time with this approach. We are not going to do that now for the sake of simplicity

# In[36]:


# initialize the dataframe
additional_consequences = pd.DataFrame(
    columns = pd.MultiIndex.from_tuples([('Incomplete',''), 
                                         ('Quantity','Unit'), 
                                         ('DV', 'Unit'), 
                                         ('DS1', 'Theta_0')]),
    index=pd.MultiIndex.from_tuples([('replacement','Cost'), 
                                     ('replacement','Time')])
)

# add the data about replacement cost and time
additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA', 'USD_2011', 21600000]
additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA', 'worker_day', 12500]  

additional_consequences


# ### 3.4.3 Load component consequence data
# 
# Now that we have the consequence data completed and available for all components in the damage sample, we can load the data to the loss model in Pelicun.
# 
# When providing custom data, you can directly provide a DataFrame like we do in this example (additional_consequences), or you can provide a path to a CSV file that is structured like the table we prepared above.
# 
# Default databases are loaded using the keyword "PelicunDefault" in the path and then providing the name of the database. The PelicunDefault keyword is automatically replaced with the path to the default component data directory. 
# 
# If there were identical components in the listed sources, Pelicun always preserves the first occurrence of a component. Hence, always start with the custom data when listing sources and add default databases in the end.

# In[37]:


# Load the loss model to pelicun
PAL.bldg_repair.load_model(
    [additional_consequences,
     "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
    loss_map)


# ### 3.4.4 Loss calculation
# 
# Loss calculation in Pelicun requires 
# - a pre-assigned set of component consequence functions;
# - a pre-assigned sample of demands and/or damages;
# - and a loss mapping matrix
# 
# The sample size for the loss calculation is automatically inferred from the demand/damage sample size.

# In[38]:


# and run the calculations
PAL.bldg_repair.calculate()


# ### 3.4.5 Loss estimates
# 
# **Repair cost of individual components and groups of components**
# 
# Below, we extract the loss sample from Pelicun and show a few example plots to illustrate how rich information this data provides about the repair consequences in the building

# In[39]:


loss_sample = PAL.bldg_repair.sample

print("Size of repair cost & time results: ", sys.getsizeof(loss_sample)/1024/1024, "MB")


# In[40]:


loss_sample['COST']['B.20.22.031'].groupby(level=[0,2,3],axis=1).sum().describe([0.1, 0.5, 0.9]).T


# In[41]:


loss_plot = loss_sample.groupby(level=[0, 2], axis=1).sum()['COST'].iloc[:, :-2]

# we add 100 to the loss values to avoid having issues with zeros when creating a log plot
loss_plot += 100

px.box(y=np.tile(loss_plot.columns, loss_plot.shape[0]), 
       x=loss_plot.values.flatten(), 
       color = [c[0] for c in loss_plot.columns]*loss_plot.shape[0],
       orientation = 'h',
       labels={
           'x':'Aggregate repair cost [2011 USD]',
           'y':'Component ID',
           'color': 'Component Group'
       },
       title=f'Range of repair cost realizations by component type',
       log_x=True,
       height=1500)


# In[42]:


loss_plot = loss_sample['COST'].groupby('loc', axis=1).sum().describe([0.1,0.5,0.9]).iloc[:, 1:]

fig = px.pie(values=loss_plot.loc['mean'],
       names=[f'floor {c}' if int(c)<5 else 'roof' for c in loss_plot.columns],
       title='Contribution of each floor to the average non-collapse repair costs',
       height=500,
       hole=0.4
      )

fig.update_traces(textinfo='percent+label')


# In[43]:


loss_plot = loss_sample['COST'].groupby(level=[1], axis=1).sum()

loss_plot['repairable'] = loss_plot.iloc[:,:-2].sum(axis=1)
loss_plot = loss_plot.iloc[:,-3:]

px.bar(x=loss_plot.columns, 
       y=loss_plot.describe().loc['mean'],
       labels={
           'x':'Damage scenario',
           'y':'Average repair cost'
       },
       title=f'Contribution to average losses from the three possible damage scenarios',
       height=400
      )


# **Aggregate losses**
# 
# Aggregating losses for repair costs is straightforward, but repair times are less trivial. Pelicun adopts the method from FEMA P-58 and provides two bounding values for aggregate repair times:
# - **parallel** assumes that repairs are conducted in parallel across locations. In each location, repairs are assumed to be sequential. This translates to aggregating component repair times by location and choosing the longest resulting aggregate value across locations.
# - **sequential** assumes repairs are performed sequentially across locations and within each location. This translates to aggregating component repair times across the entire building.
# 
# The parallel option is considered a lower bound and the sequential is an upper bound of the real repair time. Pelicun automatically calculates both options for all (i.e., not only FEMA P-58) analyses.

# In[44]:


agg_DF = PAL.bldg_repair.aggregate_losses()

agg_DF.describe([0.1, 0.5, 0.9])


# In[45]:


# filter only the repairable cases
agg_DF_plot = agg_DF.loc[agg_DF['repair_cost'] < 2e7]
px.scatter(x=agg_DF_plot[('repair_time','sequential')],
           y=agg_DF_plot[('repair_time','parallel')], 
           opacity=0.1,
           marginal_x ='histogram', marginal_y='histogram',
           labels={
               'x':'Sequential repair time [worker-days]',
               'y':'Parallel repair time [worker-days]',
           },
           title=f'Two bounds of repair time conditioned on repairable damage',
           height=750, width=750)


