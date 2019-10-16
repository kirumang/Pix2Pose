#List all the patches, files
import os,sys
import numpy as np
import tensorflow as tf
from skimage.filters import gaussian
import yaml
import skimage
from skimage import io
from skimage.transform import resize,rotate
from imgaug import augmenters as iaa
import random
from skimage.transform import rescale, rotate

sys.path.append(".")
sys.path.append("./bop_toolkit")
from bop_toolkit_lib import inout,dataset_params
from tools import bop_io
import warnings
warnings.filterwarnings("ignore")
import cv2
import csv
from pix2pose_util.common_util import get_bbox_from_mask

def get_random_background(im_height,im_width,backfiles):
    back_fn = backfiles[int(random.random()*(len(backfiles)-1))]
    back_img = cv2.imread(back_dir+"/"+back_fn)
    img_syn = np.zeros( (im_height,im_width,3))                    

    if back_img.ndim != 3:
        back_img = skimage.color.gray2rgb(back_img)
    back_v = min(back_img.shape[0],img_syn.shape[0])
    back_u = min(back_img.shape[1],img_syn.shape[1])
    img_syn[:back_v,:back_u]=back_img[:back_v,:back_u]/255

    if(img_syn.shape[0]>back_img.shape[0]):
        width = min(img_syn.shape[0]-back_v,back_img.shape[0])
        img_syn[back_v:back_v+width,:back_u]=back_img[:width,:back_u]/255
    if(img_syn.shape[1]>back_img.shape[1]):
        height = min(img_syn.shape[1]-back_u,back_img.shape[1])
        img_syn[:back_v,back_u:back_u+height]=back_img[:back_v,:height]/255                                  
    return img_syn


cfg_fn = sys.argv[1]
cfg = inout.load_json(cfg_fn)

if(len(sys.argv)!=4 and len(sys.argv)!=3):
    print("usage: python3 tools/1_1_scene_gen_for_detection.py [cfg_fn] [dataset] [mask=1(yes)/0(no),default:1]")
    sys.exit(0)
    
dataset = sys.argv[2]
back_dir = cfg['background_imgs_for_training']
backfiles = os.listdir(back_dir)

genMask=True
if(len(sys.argv)==3):
    if(int(sys.argv[3])==0):genMask=False
n_image=200000

bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global = bop_io.get_dataset(cfg,dataset)

label_file = open(os.path.join(bop_dir,"retinanet_label.csv"), 'w')  #gt file for training keras-retinanet
l_writer = csv.writer(label_file,delimiter=",")

csvfile = open(os.path.join(bop_dir,"retinanet_gt.csv"), 'w')  #gt file for training keras-retinanet
writer = csv.writer(csvfile,delimiter=",")

for m_idx,m_id in enumerate(model_ids):
    l_writer.writerow([m_id,m_idx])
label_file.close()    

if(dataset!='ycbv'):
    print("Loading...", dataset)
    im_width,im_height =cam_param_global['im_size'] 

    test_params = dataset_params.get_split_params(bop_dir,dataset,"test")
    if(test_params['depth_range'] is not None):
        mean_depth = np.mean(test_params['depth_range'])/1000
        max_depth = test_params['depth_range'][1]/1000
    elif(dataset=="itodd"):
        mean_depth = (0.601+1.102)/2
        max_depth = 1.102

    else:
        mean_depth = 1 #1default depth= 1
        max_depth = 3

    #build map from class_id(used in Detection pipeline) to actual obj_id (in GT)
    #detection of i+1 label means detetction of model number: model_ids[i]  
    crop_dir = bop_dir+"/train_crop"
    cropmask_dir = bop_dir+"/train_cropmask"
    target_dir = bop_dir+"/train_detect"
    xyz_dir = bop_dir+"/train_crop_xyz"

    if not(os.path.exists(crop_dir)):os.makedirs(crop_dir)
    if not(os.path.exists(target_dir)):os.makedirs(target_dir)
    if not(os.path.exists(target_dir+"/mask")):os.makedirs(target_dir+"/mask")

    if not(os.path.exists(xyz_dir)):os.makedirs(xyz_dir)        
    aug_color= iaa.Sequential([
                                    iaa.WithChannels(0, iaa.Add((-15, 15))),
                                    iaa.WithChannels(1, iaa.Add((-15, 15))),
                                    iaa.WithChannels(2, iaa.Add((-15, 15))),
                                    iaa.ContrastNormalization((0.8, 1.3)),
                                    iaa.Multiply((0.8, 1.2),per_channel=0.5),
                                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                                    iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
                                    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
                            ], random_order=True)

    ## First cropping the training images (RGB files, mask_files)
    model_map = model_ids.tolist()
    n_models = len(model_plys)

    model_idx = np.zeros((n_models,int(len(rgb_files)/n_models*2)),np.int)
    model_maxinst =np.zeros((n_models),np.int) 

    crop_fns = []
    crop_masks = []
    z_tras=[]
    overall_idx =0
    for img_id in range(len(rgb_files)):
        rgb_fn = rgb_files[img_id]    
        gt= gts[img_id][0]
        obj_id = int(gt['obj_id'])
        z_tra = (gt['cam_t_m2c']/1000)[2,0]
        z_tras.append(z_tra)
        filename = rgb_fn.split("/")[-1]
        if not(os.path.exists(crop_dir+"/{:02d}".format(obj_id))):os.makedirs(crop_dir+"/{:02d}".format(obj_id))
        if not(os.path.exists(cropmask_dir+"/{:02d}".format(obj_id))):os.makedirs(cropmask_dir+"/{:02d}".format(obj_id))
        crop_fn = os.path.join(crop_dir+"/{:02d}".format(obj_id),filename)
        cropmask_fn = os.path.join(cropmask_dir+"/{:02d}".format(obj_id),filename)
        if not(os.path.exists(crop_fn)):        
            img = inout.load_im(rgb_fn)               
            mask = inout.load_im(mask_files[img_id])>0
            vu_valid = np.where(mask)
            bbox = np.array([np.min(vu_valid[0]),np.min(vu_valid[1]),np.max(vu_valid[0]),np.max(vu_valid[1])])
            crop_img = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3),np.uint8)
            img = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            crop_img[mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]]=img[mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]]
            inout.save_im(crop_fn,crop_img)
            inout.save_im(cropmask_fn,mask[bbox[0]:bbox[2],bbox[1]:bbox[3]].astype(np.uint8)*255)
        crop_fns.append(crop_fn)
        crop_masks.append(cropmask_fn)
        obj_idx = model_map.index(obj_id)
        
        instance_id = model_maxinst[obj_idx]
        model_idx[obj_idx,instance_id]=overall_idx    
        model_maxinst[obj_idx]+=1
        overall_idx+=1
    z_tra_mean = np.mean(z_tras)
    mean_scale = z_tra_mean/mean_depth # 0.5 / 1
    mean_sigma = 0.5*mean_scale 
            
    max_inst=0
    ids=np.zeros((n_models),np.int)
    obj_idxes=[]
    for i in range(n_models):
        dummy = np.arange(model_maxinst[i])
        np.random.shuffle(dummy)
        obj_idxes.append(dummy)
        
    for s_id in np.arange(n_image): 
        output_fn= target_dir+"/{:06d}.png".format(s_id)
        mask_fn= target_dir+"/mask/{:06d}.npy".format(s_id)
        if(os.path.exists(output_fn)):
            continue
        if(s_id%1000==0):
            print("{}: {:06d}/{:06d} - ({:.3f}%)".format(dataset,s_id,n_image,s_id/n_image*100))
        obj_order = np.arange(n_models)
        np.random.shuffle(obj_order)    
        img_syn = get_random_background(im_height,im_width,backfiles)
        mask_visible = []
        mask_integ = np.zeros((im_height,im_width),np.int)-1
        class_gt = np.zeros( (n_models),np.int)
        
        n_obj_in_a_scene = random.randint(5,20)
        vs=np.random.randint(im_height-100,size=n_obj_in_a_scene)  
        us=np.random.randint(im_width-100,size=n_obj_in_a_scene)  
        
        for order in range(n_obj_in_a_scene):
            obj_id = obj_order[order%obj_order.shape[0]]
            idx = model_idx[obj_id,obj_idxes[obj_id][ids[obj_id]]]
            patch = inout.load_im(crop_fns[idx])        
            patch_mask = inout.load_im(crop_masks[idx])        
            ids[obj_id]+=1
            if(ids[obj_id>=model_maxinst[obj_id]]) or ids[obj_id]+1>=len(obj_idxes):
                np.random.shuffle(obj_idxes[obj_id])
                ids[obj_id]=0
                
            r_scale = min(max(0.1,random.gauss(mean_scale,mean_sigma)),1.5)
            r_rotate = random.random()*360-180
            patch = rescale(patch.astype(np.float32)/255,scale=r_scale)
            patch = rotate(patch,angle=r_rotate,resize=True)
            
            patch_mask = rescale(patch_mask.astype(np.float32)/255,scale=r_scale)
            patch_mask = rotate(patch_mask,angle=r_rotate,resize=True)
            patch_mask = patch_mask>0.5
            #random occlusion
            vu_mask = np.where(patch_mask)
            if(len(vu_mask[0]>0)):
                bbox =np.array([ np.min(vu_mask[0]), np.min(vu_mask[1]), np.max(vu_mask[0]),np.max(vu_mask[1])] )
                h =bbox[2]-bbox[0]
                w =bbox[3]-bbox[1]
                h_aug = int( (random.random()*0.5+0.0)*h) 
                w_aug = int( (random.random()*0.5+0.0)*w) 
                bbox_ct_v_t = int((bbox[0]+bbox[2])/2 )
                bbox_ct_u_t = int((bbox[1]+bbox[3])/2 )
                ratio = 0.5
                d_pos_v = int(bbox_ct_v_t+(random.random()*ratio*2-ratio)*h) 
                d_pos_u = int(bbox_ct_u_t+(random.random()*ratio*2-ratio)*w) 
                if(h_aug > 0 and w_aug>0):
                    patch_mask[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=0

            mask_visible.append(np.sum(patch_mask))
            patch_aug = aug_color.augment_image( (patch*255).astype(np.uint8) )/255
            delta_v = min(vs[order],im_width-patch_aug.shape[0])
            delta_u = min(us[order],im_height-patch_aug.shape[1])

            mask = np.zeros((img_syn.shape[0],img_syn.shape[1]),bool)
            #range adjustment
            range_v = min(delta_v+patch_aug.shape[0],im_height)
            range_u = min(delta_u+patch_aug.shape[1],im_width)
            patch_mask = patch_mask[:range_v-delta_v,:range_u-delta_u]
            patch_aug = patch_aug[:range_v-delta_v,:range_u-delta_u]
            
            #skip bad bbox (wrong scale)
            if(delta_v < 0 or delta_u<0):
                continue
            mask[delta_v:range_v,delta_u:range_u]=patch_mask
            
            #put the augmented patch to the current image set
            mask_integ[mask]=order
            img_syn[mask]=patch_aug[patch_mask]

            p= np.gradient(mask.astype(float))
            boundary = np.logical_or(p[0]>0,p[1]>0)
            sigma = random.random()*2
            boundary = gaussian(boundary.astype(float),sigma=sigma)>0
            img_blurred = gaussian(img_syn,sigma=sigma)
            img_syn[boundary] = img_blurred[boundary]

        n_inst=0
        mask_gt = np.zeros((img_syn.shape[0],img_syn.shape[1]),np.int8)
        mask_gt[:,:]=-1
        n_obj_order= obj_order.shape[0]
        for order in range(n_obj_in_a_scene):
            obj_id = obj_order[order%n_obj_order]
            mask_temp = (mask_integ==order)
            if(mask_visible[order]>0):
                visible_ratio = np.sum(mask_temp) /mask_visible[order]
                if(visible_ratio>0.3):
                    mask_gt[mask_temp]= n_obj_order*int(order/n_obj_order) + (obj_id)
                    fn_temp = "train_detect/{:06d}.png".format(s_id)
                    bbox_m = get_bbox_from_mask(mask_temp)
                    writer.writerow([fn_temp,bbox_m[1],bbox_m[0],bbox_m[3],bbox_m[2],obj_id+1]) 

        mask_gt[:,:]+=1 #convert mask values into real obj_id values (starts from 1...)     
        
        img_syn = np.maximum(np.minimum(img_syn,1),0)
        if(dataset=="itodd"):
            #make the image to the gray scale
            img_int= np.sum(img_syn,axis=2)/3
            img_syn[:,:,0]=img_int
            img_syn[:,:,1]=img_int
            img_syn[:,:,2]=img_int

        io.imsave(output_fn,(img_syn*255).astype(np.uint8))
        
        if(genMask):
            np.save(mask_fn,mask_gt)
        
        for i in range(n_models):
            if(ids[i] >=model_maxinst[i]):
                ids[i]=0
                np.random.shuffle(obj_idxes[i])
    
else:
    #for ycbv, no additinal augmentation is necessary. (cluttered/occluded objects already)
    target_dir = bop_dir+"/train_detect"
    im_width= 640
    im_height =480
    train_syn_dir = bop_dir+"/train_synt"
    train_real_dir = bop_dir+"/train_real"
    if not(os.path.exists(target_dir)): os.makedirs(target_dir)
    if not(os.path.exists(target_dir+"/mask")): os.makedirs(target_dir+"/mask")
    s_id=0
    for train_type in ["syn","real"]:
        if(train_type=="syn"): train_dir = train_syn_dir
        else: train_dir = train_real_dir

        for folder in os.listdir(train_dir):
            sub_dir = os.path.join(train_dir,folder)
            print("Processing:",sub_dir)
            scene_gt_fn =os.path.join(sub_dir,"scene_gt.json")
            scene_gts = inout.load_scene_gt(scene_gt_fn)
            for img_id in sorted(scene_gts.keys()):
                im_id = int(img_id)
                rgb_fn = os.path.join(sub_dir+"/rgb","{:06d}.png".format(im_id))
                gts = scene_gts[im_id]
                mask_gt = np.zeros((im_height,im_width),np.int8)
                mask_gt[:,:]=0
                for gt_id,gt in enumerate(gts):
                    mask_img_fn = os.path.join(os.path.join(sub_dir,"mask_visib"),"{:06d}_{:06d}.png".format(im_id,gt_id))
                    mask_img = io.imread(mask_img_fn)
                    obj_id = gt['obj_id'] #real 1~
                    mask_obj = mask_img>0
                    mask_gt[mask_obj] = obj_id                      
                    fn_temp = "train_detect/{:06d}.png".format(s_id)
                    bbox_m = get_bbox_from_mask(mask_obj)
                    writer.writerow([fn_temp,bbox_m[1],bbox_m[0],bbox_m[3],bbox_m[2],obj_id+1]) 


                output_fn= target_dir+"/{:06d}.png".format(s_id)
                mask_fn= target_dir+"/mask/{:06d}.npy".format(s_id)
                if(train_type=="syn"): #put background images
                    rgb = cv2.imread(rgb_fn)   
                    if(rgb is None)                         :
                        continue
                    img_syn = get_random_background(im_height,im_width,backfiles)
                    img_syn[mask_gt>0] = rgb[mask_gt>0,:3]
                    cv2.imwrite(output_fn,img_syn)
                    if(genMask): np.save(mask_fn,mask_gt)
                else:
                    os.symlink(rgb_fn,output_fn) 
                    if(genMask): np.save(mask_fn,mask_gt)
                s_id+=1
    
