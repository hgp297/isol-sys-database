############################################################################
#               Generalized component object

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2024

# Description:  Object stores all information for building contents for Pelicun

# Open issues:  (1) 

############################################################################

class Loss_Analysis:
        
    # import attributes as building characteristics from pd.Series
    def __init__(self, edp_sheet):
        for key, value in edp_sheet.items():
            setattr(self, key, value)