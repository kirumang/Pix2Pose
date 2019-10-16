
import os,sys
sys.path.append(".")
sys.path.append("./bop_toolkit")


import math
from plyfile import PlyData, PlyElement
import numpy as np
import itertools
from tools import bop_io
from bop_toolkit_lib import inout,dataset_params
from numpy.lib import recfunctions

def get_xyz_max(fn_read):
    plydata = PlyData.read(fn_read)
    #x,y,z : embbedding to RGB
    x_ct = np.mean(plydata.elements[0].data['x'])    
    x_abs = np.max(np.abs(plydata.elements[0].data['x']-x_ct))    
    y_ct = np.mean(plydata.elements[0].data['y'])    
    y_abs = np.max(np.abs(plydata.elements[0].data['y']-y_ct))   
    
    z_ct = np.mean(plydata.elements[0].data['z'])    
    z_abs = np.max(np.abs(plydata.elements[0].data['z']-z_ct))   
    
    return x_abs,y_abs,z_abs,x_ct,y_ct,z_ct
    


def convert_unique(fn_read,fn_write,center_x=True,center_y=True,center_z=True):
    plydata = PlyData.read(fn_read)

    #x,y,z : embbedding to RGB
    x_ct = np.mean(plydata.elements[0].data['x'])    
    if not(center_x):
        x_ct=0
    x_abs = np.max(np.abs(plydata.elements[0].data['x']-x_ct))
    
    y_ct = np.mean(plydata.elements[0].data['y'])    
    if not(center_y):
        y_ct=0
    y_abs = np.max(np.abs(plydata.elements[0].data['y']-y_ct))    
    
    z_ct = np.mean(plydata.elements[0].data['z'])    
    if not(center_z):
        z_ct=0
    z_abs = np.max(np.abs(plydata.elements[0].data['z']-z_ct))    
    n_vert = plydata.elements[0].data['x'].shape[0]
   
    for i in range(n_vert):
        r=(plydata.elements[0].data['x'][i]-x_ct)/x_abs #-1 to 1
        r = (r+1)/2 #0 to 2 -> 0 to 1        
        g=(plydata.elements[0].data['y'][i]-y_ct)/y_abs
        g = (g+1)/2
        b=(plydata.elements[0].data['z'][i]-z_ct)/z_abs
        b = (b+1)/2
        #if b> 1: b=1
        #if b<0: b=0
        plydata.elements[0].data['red'][i]=r*255
        plydata.elements[0].data['green'][i]=g*255
        plydata.elements[0].data['blue'][i]=b*255
    plydata.write(fn_write)        
    return x_abs,y_abs,z_abs,x_ct,y_ct,z_ct

def rmfield( a, *fieldnames_to_remove ):
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]

if(len(sys.argv)<2):
    print("python3 tools/2_1_ply_file_to_3d_coord_model.py [cfg_fn] [dataset_name]")

cfg_fn =sys.argv[1]
cfg = inout.load_json(cfg_fn)

dataset = sys.argv[2]
bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global = bop_io.get_dataset(cfg,dataset)


if not(os.path.exists(bop_dir + "/models_xyz/")):
    os.makedirs(bop_dir + "/models_xyz/")
norm_factor = bop_dir+"/models_xyz/"+"norm_factor.json"
param={}


for m_id,model_ply in enumerate(model_plys):
    model_id = model_ids[m_id]
    m_info = model_info['{}'.format(model_id)]
    keys = m_info.keys()
    sym_continous = [0,0,0,0,0,0]
    center_x = center_y = center_z = True    
    #if('symmetries_discrete' in keys): #use this when objects are centered already
    #    center_x = center_y = center_z = False
    #    print("keep origins of the object when it has symmetric poses")    
    fn_read = model_ply
    fname = model_ply.split("/")[-1]
    obj_id = int(fname[4:-4])
    fn_write = bop_dir + "/models_xyz/" + fname    
    x_abs,y_abs,z_abs,x_ct,y_ct,z_ct = convert_unique(fn_read,fn_write,center_x=center_x,center_y=center_y,center_z=center_z)
    print(obj_id,x_abs,y_abs,z_abs,x_ct,y_ct,z_ct)
    param[int(obj_id)]={'x_scale':float(x_abs),'y_scale':float(y_abs),'z_scale':float(z_abs),'x_ct':float(x_ct),'y_ct':float(y_ct),'z_ct':float(z_ct)}

inout.save_json(norm_factor,param)
