import numpy as np
def get_bbox_from_mask(mask):
    vu = np.where(mask)
    if(len(vu[0])>0):
        return np.array([np.min(vu[0]),np.min(vu[1]),np.max(vu[0]),np.max(vu[1])],np.int)
    else:
        return np.zeros((4),np.int)
