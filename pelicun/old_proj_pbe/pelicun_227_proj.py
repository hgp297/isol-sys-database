import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 100

# and for plotting
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from pelicun.assessment import Assessment

raw_demands = pd.read_csv('demand_data.csv', index_col=0)
raw_demands.tail(30)