############################################################################
#               Building object

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2023

# Description:  Object stores all information about building for Opensees

# Open issues:  (1) 

############################################################################

class Building(dict):
    
    def __init__(self, *args, **kwargs):
        super(Building, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
test = {'S_1': 1.0, 'T_m': 3.5, 'type': 'MF'}
bldg = Building(test)