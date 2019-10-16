import yaml
import skimage
from skimage import io
from skimage.transform import resize,rotate
import os,sys

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR) 
sys.path.append("./bop_toolkit")



from rendering import utils as renderutil
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D

import matplotlib.pyplot as plt
import transforms3d as tf3d
import numpy as np
import time
import cv2

from bop_toolkit_lib import inout,dataset_params
from tools import bop_io

def get_sympose(rot_pose,sym):
    rotation_lock=False
    if(np.sum(sym)>0): #continous symmetric
        axis_order='s'
        multiply=[]
        for axis_id,axis in enumerate(['x','y','z']):
            if(sym[axis_id]==1):
                axis_order+=axis
                multiply.append(0)
        for axis_id,axis in enumerate(['x','y','z']):
            if(sym[axis_id]==0):
                axis_order+=axis
                multiply.append(1)

        axis_1,axis_2,axis_3 =tf3d.euler.mat2euler(rot_pose,axis_order)
        axis_1 = axis_1*multiply[0]
        axis_2 = axis_2*multiply[1]
        axis_3 = axis_3*multiply[2]            
        rot_pose =tf3d.euler.euler2mat(axis_1,axis_2,axis_3,axis_order) #
        sym_axis_tr = np.matmul(rot_pose,np.array([sym[:3]]).T).T[0]
        z_axis = np.array([0,0,1])
        #if symmetric axis is pallell to the camera z-axis, lock the rotaion augmentation
        inner = np.abs(np.sum(sym_axis_tr*z_axis))
        if(inner>0.8):
            rotation_lock=True #lock the in-plane rotation                            

    return rot_pose,rotation_lock
def get_rendering(obj_model,rot_pose,tra_pose, ren):
    ren.clear()
    M=np.eye(4)
    M[:3,:3]=rot_pose
    M[:3,3]=tra_pose
    ren.draw_model(obj_model, M)
    img_r, depth_rend = ren.finish()
    img_r = img_r[:,:,::-1] *255    
    vu_valid = np.where(depth_rend>0)
    bbox_gt = np.array([np.min(vu_valid[0]),np.min(vu_valid[1]),np.max(vu_valid[0]),np.max(vu_valid[1])])
    return img_r,depth_rend,bbox_gt
def augment_inplane_gen(xyz_id,img,img_r,depth_rend,mask,isYCB=False,step=10):
    depth_mask = (depth_rend>0).astype(np.float)
    for rot in np.arange(step,360,step):
        xyz_fn =  os.path.join(xyz_dir,"{:06d}_{:03d}.npy".format(xyz_id,rot))
        img_r_rot = rotate((img_r/255).astype(np.float32), rot,resize=True,cval=0)*255
        img_rot = rotate((img/255).astype(np.float32), rot,resize=True,cval=0.5)*255
        depth_rot = rotate(depth_mask, rot,resize=True)

        if(isYCB):
            mask_rot = rotate(mask.astype(np.float32), rot,resize=True)
        vu_box = np.where(depth_rot>0)
        bbox_t = np.array([np.min(vu_box[0]),np.min(vu_box[1]),np.max(vu_box[0]),np.max(vu_box[1])])
        if(isYCB):img_npy = np.zeros((bbox_t[2]-bbox_t[0],bbox_t[3]-bbox_t[1],7),np.uint8)
        else:img_npy = np.zeros((bbox_t[2]-bbox_t[0],bbox_t[3]-bbox_t[1],6),np.uint8)
        img_npy[:,:,:3]=[128,128,128]
        img_npy[:,:,:3]=img_rot[bbox_t[0]:bbox_t[2],bbox_t[1]:bbox_t[3]]
        img_npy[:,:,3:6]=img_r_rot[bbox_t[0]:bbox_t[2],bbox_t[1]:bbox_t[3]]
        if(isYCB):
            img_npy[:,:,6]=mask_rot[bbox_t[0]:bbox_t[2],bbox_t[1]:bbox_t[3]]
        
        data=img_npy
        max_axis=max(data.shape[0],data.shape[1])
        if(max_axis>128):                
                scale = 128.0/max_axis 
                new_shape=np.array([data.shape[0]*scale+0.5,data.shape[1]*scale+0.5]).astype(np.int) 
                new_data = np.zeros((new_shape[0],new_shape[1],data.shape[2]),data.dtype)
                new_data[:,:,:3] = resize( (data[:,:,:3]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                new_data[:,:,3:6] = resize( (data[:,:,3:6]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                if(data.shape[2]>6):
                    new_data[:,:,6:] = (resize(data[:,:,6:],(new_shape[0],new_shape[1]))>0.5).astype(np.uint8)
        else:
            new_data=data
        np.save(xyz_fn,new_data)

augment_inplane=30 
if len(sys.argv)<3:
    print("rendering 3d coordinate images using a converted ply file, format of 6D pose challange(http://cmp.felk.cvut.cz/sixd/challenge_2017/) can be used")
    print("python3 tools/2_2_render_pix2pose_training.py [cfg_fn] [dataset_name]")    
else:
    cfg_fn = sys.argv[1] #"cfg/cfg_bop2019.json"
    cfg = inout.load_json(cfg_fn)

    dataset=sys.argv[2]
    bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,\
        depth_files,mask_files,gts,cam_param_global,scene_cam =\
             bop_io.get_dataset(cfg,dataset,incl_param=True)
    
    xyz_target_dir = bop_dir+"/train_xyz"
    im_width,im_height =cam_param_global['im_size'] 
    cam_K = cam_param_global['K']
    #check if the image dimension is the same
    rgb_fn = rgb_files[0]
    img_temp = inout.load_im(rgb_fn)
    if(img_temp.shape[0]!=im_height or img_temp.shape[1]!=im_width):
        print("the size of training images is different from test images")
        im_height = img_temp.shape[0]
        im_width = img_temp.shape[1]
     
    ren = Renderer((im_width,im_height),cam_K)

t_model=-1
for m_id,model_id in enumerate(model_ids):
        if(t_model!=-1 and model_id!=t_model):
            continue
        m_info = model_info['{}'.format(model_id)]
        ply_fn =bop_dir+"/models_xyz/obj_{:06d}.ply".format(int(model_id))
        obj_model = Model3D()
        obj_model.load(ply_fn, scale=0.001)
        keys = m_info.keys()
        sym_continous = [0,0,0,0,0,0]
        if('symmetries_discrete' in keys):
            print(model_id,"is symmetric_discrete")
            print("During the training, discrete transform will be properly handled by transformer loss")
        if('symmetries_continuous' in keys):
            print(model_id,"is symmetric_continous")
            print("During the rendering, rotations w.r.t to the symmetric axis will be ignored")
            sym_continous[:3] = m_info['symmetries_continuous'][0]['axis']
            sym_continous[3:]= m_info['symmetries_continuous'][0]['offset']
            print("Symmetric axis(x,y,z):", sym_continous[:3])
        xyz_dir =xyz_target_dir+"/{:02d}".format(int(model_id))
        if not(os.path.exists(xyz_dir)):
            os.makedirs(xyz_dir)
        xyz_id = 0
        for img_id in range(len(rgb_files)):
            gt = (gts[img_id])[0]
            obj_id = int(gt['obj_id'])
            if(obj_id !=int(model_id)):
                continue            
            xyz_fn = os.path.join(xyz_dir,"{:06d}.npy".format(xyz_id))
            
            rgb_fn = rgb_files[img_id]    
            if(dataset not in ['hb','ycbv','itodd']):
                #for dataset that has gvien training images.
                cam_K = np.array(scene_cam[img_id]["cam_K"]).reshape(3,3)
            ren.set_cam(cam_K)
            tra_pose = np.array((gt['cam_t_m2c']/1000))[:,0]
            rot_pose = np.array(gt['cam_R_m2c']).reshape(3,3)
            
            mask = inout.load_im(mask_files[img_id])>0            
            rot_pose,rotation_lock = get_sympose(rot_pose,sym_continous)
            img_r,depth_rend,bbox_gt = get_rendering(obj_model,rot_pose,tra_pose,ren)
            
            img = inout.load_im(rgb_fn)

            img[depth_rend==0]=[128,128,128]
            data = np.zeros((bbox_gt[2]-bbox_gt[0],bbox_gt[3]-bbox_gt[1],6),np.uint8)
            data[:,:,:3]=img[bbox_gt[0]:bbox_gt[2],bbox_gt[1]:bbox_gt[3]]
            data[:,:,3:6]=img_r[bbox_gt[0]:bbox_gt[2],bbox_gt[1]:bbox_gt[3]]
            max_axis=max(data.shape[0],data.shape[1])
            if(max_axis>128):
                #resize to 128 
                scale = 128.0/max_axis #128/200, 200->128
                new_shape=np.array([data.shape[0]*scale+0.5,data.shape[1]*scale+0.5]).astype(np.int) 
                new_data = np.zeros((new_shape[0],new_shape[1],data.shape[2]),data.dtype)
                new_data[:,:,:3] = resize( (data[:,:,:3]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                new_data[:,:,3:6] = resize( (data[:,:,3:6]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                if(data.shape[2]>6):
                    new_data[:,:,6:] = (resize(data[:,:,6:],(new_shape[0],new_shape[1]))>0.5).astype(np.uint8)
            else:
                new_data= data
            np.save(xyz_fn,new_data)
            
            if(augment_inplane>0 and not(rotation_lock)):
                augment_inplane_gen(xyz_id,img,img_r,depth_rend,mask,isYCB=False,step=augment_inplane)
            xyz_id+=1
        if(dataset == "ycbv"):
            #for ycbv, exceptionally extract patches
            #from cluttered training images. 
            print("using real images for training")
            train_real_dir = bop_dir+"/train_real"
            for folder in sorted(os.listdir(train_real_dir)):
                sub_dir = os.path.join(train_real_dir,folder)
                print("Processing:",sub_dir)
                scene_gt_fn =os.path.join(sub_dir,"scene_gt.json")
                scene_gts = inout.load_scene_gt(scene_gt_fn)
                scene_gt_info = inout.load_json(os.path.join(sub_dir,"scene_gt_info.json"))
                scene_cam_fn = os.path.join(sub_dir,"scene_camera.json")
                scene_cam_real = inout.load_scene_camera(scene_cam_fn)
                scene_keys = scene_gts.keys()
                n_scene =len(scene_keys)
                for img_id in np.arange(1,n_scene,100):
                    im_id = img_id
                    rgb_fn = os.path.join(sub_dir+"/rgb","{:06d}.png".format(im_id))
                    print(rgb_fn)
                    gts_real = scene_gts[im_id]
                    scene_info = scene_gt_info["{}".format(im_id)]
                    for gt_id in range(len(gts_real)):
                        gt = gts_real[gt_id]
                        obj_id = int(gt['obj_id'])
                        bbox = scene_info[gt_id]['bbox_obj']
                        visib_fract = scene_info[gt_id]['visib_fract']
                        #skip objects in the boundary
                        if(obj_id !=int(model_id) or visib_fract<0.5):
                            continue            
                        if( (bbox[0]+bbox[2]) > (im_width-10) or  (bbox[1]+bbox[3]) > (im_height-10) ):
                            continue                        
                        if ( (bbox[0] < 10) or  (bbox[1]<10)):
                            continue                        
                        mask_img_fn = os.path.join(os.path.join(sub_dir,"mask_visib"),"{:06d}_{:06d}.png".format(im_id,gt_id))
                        mask_img = io.imread(mask_img_fn)
                        mask_obj = mask_img>0
                        
                        xyz_fn = os.path.join(xyz_dir,"{:06d}.npy".format(xyz_id))
                        rgb_fn = rgb_fn    
                        cam_K = np.array(scene_cam_real[img_id]["cam_K"]).reshape(3,3)
                        ren.set_cam(cam_K)
                        tra_pose = np.array((gt['cam_t_m2c']/1000))[:,0]
                        rot_pose = np.array(gt['cam_R_m2c']).reshape(3,3)                        
                        
                        img = inout.load_im(rgb_fn)               
                        mask = inout.load_im(mask_files[img_id])>0
                        rot_pose,rotation_lock = get_sympose(rot_pose,sym_continous)
                        img_r,depth_rend,bbox_gt = get_rendering(obj_model,rot_pose,tra_pose,ren)
                        img[depth_rend==0]=[128,128,128]
                        if(bbox_gt[2]-bbox_gt[0]==0 or bbox_gt[3]-bbox_gt[1]==0):
                            continue
                        if(bbox_gt[2]-bbox_gt[0]<10) and (bbox_gt[3]-bbox_gt[1]<10):
                            continue
                        #for YCB real images -> visible mask added
                        img_npy = np.zeros((bbox_gt[2]-bbox_gt[0],bbox_gt[3]-bbox_gt[1],7),np.uint8)
                        img_npy[:,:,:3]=img[bbox_gt[0]:bbox_gt[2],bbox_gt[1]:bbox_gt[3]]
                        img_npy[:,:,3:6]=img_r[bbox_gt[0]:bbox_gt[2],bbox_gt[1]:bbox_gt[3]]
                        img_npy[:,:,6]=mask[bbox_gt[0]:bbox_gt[2],bbox_gt[1]:bbox_gt[3]]
                        data=img_npy
                        max_axis=max(data.shape[0],data.shape[1])
                        if(max_axis>128):
                            #resize to 128 
                            scale = 128.0/max_axis #128/200, 200->128
                            new_shape=np.array([data.shape[0]*scale+0.5,data.shape[1]*scale+0.5]).astype(np.int) 
                            new_data = np.zeros((new_shape[0],new_shape[1],data.shape[2]),data.dtype)
                            new_data[:,:,:3] = resize( (data[:,:,:3]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                            new_data[:,:,3:6] = resize( (data[:,:,3:6]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                            if(data.shape[2]>6):
                                new_data[:,:,6:] = (resize(data[:,:,6:],(new_shape[0],new_shape[1]))>0.5).astype(np.uint8)
                        else:
                            new_data=img_npy
                        
                        np.save(xyz_fn,new_data)
                        if(augment_inplane>0 and not(rotation_lock)):
                            augment_inplane_gen(xyz_id,img,img_r,depth_rend,mask,isYCB=True,step=augment_inplane)
                        xyz_id+=1
