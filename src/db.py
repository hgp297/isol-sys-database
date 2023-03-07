############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Test file

# Open issues:  

############################################################################

from structure_database import Database

main_obj = Database(1000)

main_obj.design_bearings(filter_designs=True)

main_obj.design_structure()

test_mf = main_obj.mf_designs.iloc[0]
test_cbf = main_obj.tfp_designs.iloc[3]

# test build one building (MF, TFP only)
from building import Building
mf_bldg = Building(test_mf)
mf_bldg.model_frame()