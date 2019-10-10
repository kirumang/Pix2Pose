'''
Render training images of datasets that do not contain traininig images as default
HB, ITODD, YCB-V
Samples poses are the same from the one that has close poses

'''
import yaml
import skimage
from skimage import io
from skimage.transform import resize,rotate
import os,sys

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  
sys.path.append("./bop_toolkit")

from rendering import utils as renderutil
from rendering.renderer import Renderer
from rendering.model import Model3D

import matplotlib.pyplot as plt
import transforms3d as tf3d
import numpy as np
import time

from bop_toolkit_lib import inout
from tools import bop_io
import copy

#YCB(have to check) - > LMO
#HB, ITODD -> T-LESS
ref_gt =inout.load_scene_gt(os.path.join("/home/kiru/media/hdd/bop/tless/train_render_reconst/000001","scene_gt.json"))
ref_camera=inout.load_scene_camera(os.path.join("/home/kiru/media/hdd/bop/tless/train_render_reconst/000001","scene_camera.json"))
#bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global = bop_io.get_dataset('hb',train=True)
#bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global = bop_io.get_dataset('itodd',train=True)
bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global = bop_io.get_dataset('ycbv',train=True)

im_width,im_height =cam_param_global['im_size'] 
camK=cam_param_global['K']
cam_K_list = np.array(camK).reshape(-1)

ren = Renderer((im_width, im_height), camK)
source_dir = bop_dir + "/train"
if not(os.path.exists(source_dir)):os.makedirs(source_dir)

for i in range(len(model_plys)):
    target_dir = source_dir+"/{:06d}".format(model_ids[i])
    if not(os.path.exists(target_dir)):os.makedirs(target_dir)
    if not(os.path.exists(target_dir+"/rgb")):os.makedirs(target_dir+"/rgb")
    if not(os.path.exists(target_dir+"/depth")):os.makedirs(target_dir+"/depth")
    if not(os.path.exists(target_dir+"/mask")):os.makedirs(target_dir+"/mask")
    new_gt =copy.deepcopy(ref_gt)
    new_camera=copy.deepcopy(ref_camera)
    scene_gt = os.path.join(target_dir,"scene_gt.json")            
    scene_camera = os.path.join(target_dir,"scene_camera.json")   

    ply_fn =  model_plys[i]
    obj_model = Model3D()
    obj_model.load(ply_fn, scale=0.001)
    
    for im_id in range(len(ref_gt)):
        rgb_fn = os.path.join(target_dir+"/rgb","{:06d}.png".format(im_id))
        depth_fn= os.path.join(target_dir+"/depth","{:06d}.png".format(im_id))
        mask_fn  = os.path.join(target_dir+"/mask","{:06d}.png".format(im_id))

        rot = ref_gt[im_id][0]['cam_R_m2c']
        tra = ref_gt[im_id][0]['cam_t_m2c']/1000
        
        
        tf = np.eye(4)
        tf[:3,:3]=rot
        tf[:3,3]=tra[:,0]

        ren.clear()
        ren.draw_model(obj_model, tf)
        img_r, depth = ren.finish()
        img_r = img_r[:,:,::-1]
        mask =depth>0
        inout.save_im(rgb_fn,(img_r*255).astype(np.uint8))
        inout.save_im(mask_fn,mask.astype(np.uint8)*255)
        
        new_gt[im_id][0]['obj_bb']=[0,0,0,0]
        new_gt[im_id][0]['obj_id']=int(model_ids[i])
        new_camera[im_id]['cam_K']=np.array(camK)
        new_camera[im_id]['depth_scale']=float(1)
        #inout.save_depth(depth_fn,depth*65535) #we don't need detph for training (use only for ICP/inference)     
        
    inout.save_scene_gt(scene_gt,new_gt)
    inout.save_scene_camera(scene_camera,new_camera)