############################################################################
#               Generalized bearing object

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: March 2023

# Description:  Object stores all information about about a bearing. Primarily
# used for plotting and troubleshooting

# Open issues:  (1) 

############################################################################

# class takes a pandas Series (row) and creates an object that holds
# design information

class Bearing:
        
    # import attributes as building characteristics from pd.Series
    def __init__(self, design):
        for key, value in design.items():
            setattr(self, key, value)
            
    def plot_backbone(self):
        bearing_type = self.isolation_system
        
        if bearing_type == 'TFP':
            self.plot_TFP_backbone(self)
        else:
            self.plot_LRB_backbone(self)
            
    def plot_TFP_backbone(self):
        import matplotlib.pyplot as plt
        
    def plot_LRB_backbone(self):
        import matplotlib.pyplot as plt