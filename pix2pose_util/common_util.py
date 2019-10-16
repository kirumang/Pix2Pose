import numpy as np
import cv2
from scipy import ndimage

def get_bbox_from_mask(mask):
    vu = np.where(mask)
    if(len(vu[0])>0):
        return np.array([np.min(vu[0]),np.min(vu[1]),np.max(vu[0]),np.max(vu[1])],np.int)
    else:
        return np.zeros((4),np.int)

##ICP related
def getXYZ(depth,fx,fy,cx,cy,bbox=np.array([0])):
     #get x,y,z coordinate in mm dimension
    uv_table = np.zeros((depth.shape[0],depth.shape[1],2),dtype=np.int16)
    column = np.arange(0,depth.shape[0])
    uv_table[:,:,1] = np.arange(0,depth.shape[1]) - cx #x-c_x (u)
    uv_table[:,:,0] = column[:,np.newaxis] - cy #y-c_y (v)

    if(bbox.shape[0]==1):
         xyz=np.zeros((depth.shape[0],depth.shape[1],3)) #x,y,z
         xyz[:,:,0] = uv_table[:,:,1]*depth*1/fx
         xyz[:,:,1] = uv_table[:,:,0]*depth*1/fy
         xyz[:,:,2] = depth
    else: #when boundry region is given
         xyz=np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3)) #x,y,z
         xyz[:,:,0] = uv_table[bbox[0]:bbox[2],bbox[1]:bbox[3],1]*depth[bbox[0]:bbox[2],bbox[1]:bbox[3]]*1/fx
         xyz[:,:,1] = uv_table[bbox[0]:bbox[2],bbox[1]:bbox[3],0]*depth[bbox[0]:bbox[2],bbox[1]:bbox[3]]*1/fy
         xyz[:,:,2] = depth[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    return xyz

def get_normal(depth_refine,fx=-1,fy=-1,cx=-1,cy=-1,bbox=np.array([0]),refine=True):
    '''
    fast normal computation
    '''
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]
    centerX=cx
    centerY=cy
    constant_x = 1/fx
    constant_y = 1/fy

    if(refine):
        depth_refine = np.nan_to_num(depth_refine)
        mask = np.zeros_like(depth_refine).astype(np.uint8)
        mask[depth_refine==0]=1
        depth_refine = depth_refine.astype(np.float32)
        depth_refine = cv2.inpaint(depth_refine,mask,2,cv2.INPAINT_NS)
        depth_refine = depth_refine.astype(np.float)
        depth_refine = ndimage.gaussian_filter(depth_refine,2)

    uv_table = np.zeros((res_y,res_x,2),dtype=np.int16)
    column = np.arange(0,res_y)
    uv_table[:,:,1] = np.arange(0,res_x) - centerX #x-c_x (u)
    uv_table[:,:,0] = column[:,np.newaxis] - centerY #y-c_y (v)

    if(bbox.shape[0]==4):
        uv_table = uv_table[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        v_x = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
        v_y = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
        normals = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
        depth_refine=depth_refine[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    else:
        v_x = np.zeros((res_y,res_x,3))
        v_y = np.zeros((res_y,res_x,3))
        normals = np.zeros((res_y,res_x,3))
    
    uv_table_sign= np.copy(uv_table)
    uv_table=np.abs(np.copy(uv_table))

    
    dig=np.gradient(depth_refine,2,edge_order=2)
    v_y[:,:,0]=uv_table_sign[:,:,1]*constant_x*dig[0]
    v_y[:,:,1]=depth_refine*constant_y+(uv_table_sign[:,:,0]*constant_y)*dig[0]
    v_y[:,:,2]=dig[0]

    v_x[:,:,0]=depth_refine*constant_x+uv_table_sign[:,:,1]*constant_x*dig[1]
    v_x[:,:,1]=uv_table_sign[:,:,0]*constant_y*dig[1]
    v_x[:,:,2]=dig[1]

    cross = np.cross(v_x.reshape(-1,3),v_y.reshape(-1,3))
    norm = np.expand_dims(np.linalg.norm(cross,axis=1),axis=1)
    norm[norm==0]=1
    cross = cross/norm
    if(bbox.shape[0]==4):
        cross =cross.reshape((bbox[2]-bbox[0],bbox[3]-bbox[1],3))
    else:
        cross =cross.reshape(res_y,res_x,3)
    cross= np.nan_to_num(cross)
    return cross