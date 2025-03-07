
# -*- coding: utf-8 -*-
"""
Config class for the python truss model

@author: gabeapaza, modified by roshan94
"""
class Config:
    
    def __init__(self, sidenum):
        #sidenum = 3  # 3 | 4 | 5 | 6
        self.sidenum_nvar_map = {2: 6, 3: 30, 4: 108, 5: 280, 6: 600, 7: 1134, 8: 1960, 9: 3168, 10: 4860, 11: 7150, 12: 10164, 13: 14040, 14: 18928, 15: 24990, 16: 32400, 17: 41344, 18: 52020, 19: 64638, 20: 79420}
        self.num_vars = self.sidenum_nvar_map[sidenum]

    # def get_num_vars(self):
    #     return self.num_vars

































