import os,sys
from math import radians
import csv
import cv2
from matplotlib import pyplot as plt
import time
import random
import numpy as np
import transforms3d as tf3d
from skimage.transform import resize 

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

if(len(sys.argv)!=4):
    print("python3 tools/5_evaluation_bop_basic.py [gpu_id] [cfg file] [dataset_name]")
    sys.exit()
    
gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
gpu_rendering=False
import tensorflow as tf
from bop_toolkit_lib import inout
from tools import bop_io
from pix2pose_util import data_io as dataio
from pix2pose_model import ae_model as ae
from pix2pose_model import recognition as recog
from pix2pose_util.common_util import getXYZ,get_normal,get_bbox_from_mask

if(gpu_rendering):
    from rendering.gpu_render import cuda_renderer
else:
    from rendering import utils as renderutil
    from rendering.renderer_xyz import Renderer
    from rendering.model import Model3D

import math
def render_obj(obj_m,rot,tra,cam_K,ren):
        ren.clear()
        ren.set_cam(cam_K)
        pred_pose=np.eye(4)
        pred_pose[:3,:3]=rot.reshape(3,3)
        if(tra[2]>100):tra=tra/1000
        pred_pose[:3,3]=tra
        ren.draw_model(obj_m, pred_pose)        
        img, depth = ren.finish()
        img = img[:,:,::-1]*255
        return img,depth
def render_obj_gpu(obj_m,rot,tra,cam_K,ren):
    if(tra[2]>100):tra=tra/1000
    #print(rot,tra)
    ren.set_cam(cam_K)
    depth,mask,bbox = ren.render_obj(obj_m,rot,tra)    
    return depth,depth
def icp_refinement(pts_tgt,obj_model,rot_pred,tra_pred,cam_K,ren):
    centroid_tgt = np.array([np.mean(pts_tgt[:,0]),np.mean(pts_tgt[:,1]),np.mean(pts_tgt[:,2])])
    if(tra_pred[2]<300 or tra_pred[2]>5000): 
        tra_pred = centroid_tgt*1000            
    
    #opengl renderer (bit slower, more precise)
    if(gpu_rendering):
        img_init,depth_init = render_obj_gpu(obj_model,rot_pred,tra_pred/1000,cam_K,ren)
    else:
        img_init,depth_init = render_obj(obj_models[obj_order_id],rot_pred,tra_pred/1000,cam_K,ren)
    init_mask = depth_init>0
    bbox_init = get_bbox_from_mask(init_mask>0)    
    tf =np.eye(4)    
    if(bbox_init[2]-bbox_init[0] < 10 or bbox_init[3]-bbox_init[1] < 10): 
        return tf,-1
    if(np.sum(init_mask)<10):
        return tf,-1
    points_src = np.zeros((bbox_init[2]-bbox_init[0],bbox_init[3]-bbox_init[1],6),np.float32)
    points_src[:,:,:3] = getXYZ(depth_init,cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2],bbox_init)
    points_src[:,:,3:] = get_normal(depth_init,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2],refine=True,bbox=bbox_init)
    points_src = points_src[init_mask[bbox_init[0]:bbox_init[2],bbox_init[1]:bbox_init[3]]>0]

    #adjust the initial translation using centroids of visible points
    centroid_src = np.array([np.mean(points_src[:,0]),np.mean(points_src[:,1]),np.mean(points_src[:,2])])
    trans_adjust = centroid_tgt - centroid_src
    tra_pred = tra_pred +trans_adjust*1000
    points_src[:,:3]+=trans_adjust

    icp_fnc = cv2.ppf_match_3d_ICP(100,tolerence=0.05,numLevels=4) #1cm
    retval, residual, pose=icp_fnc.registerModelToScene(points_src.reshape(-1,6), pts_tgt.reshape(-1,6))    
    
    tf[:3,:3]=rot_pred
    tf[:3,3]=tra_pred/1000 #in m     
    tf = np.matmul(pose,tf)    
    return tf,residual

cfg_fn =sys.argv[2]
cfg = inout.load_json(cfg_fn)
detect_type = cfg['detection_pipeline']
if detect_type=='rcnn':
    detection_dir=cfg['path_to_detection_pipeline']
    sys.path.append(detection_dir)
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib
    from tools.mask_rcnn_util import BopInferenceConfig
    def get_rcnn_detection(image_t,model):
        image_t_resized, window, scale, padding, crop = utils.resize_image(
                        np.copy(image_t),
                        min_dim=config.IMAGE_MIN_DIM,
                        min_scale=config.IMAGE_MIN_SCALE,
                        max_dim=config.IMAGE_MAX_DIM,
                        mode=config.IMAGE_RESIZE_MODE)
        results = model.detect([image_t_resized], verbose=0)
        r = results[0]
        rois = r['rois']
        if(scale!=1): 
            masks_all = r['masks'][window[0]:window[2],window[1]:window[3],:] 
            masks = np.zeros((image_t.shape[0],image_t.shape[1],masks_all.shape[2]),bool) 
            for mask_id in range(masks_all.shape[2]): 
                masks[:,:,mask_id]=resize(masks_all[:,:,mask_id].astype(np.float),(image_t.shape[0],image_t.shape[1]))>0.5 
            #resize all the masks             
            rois=rois/scale 
            window = np.array(window) 
            window[0] = window[0]/scale 
            window[1] = window[1]/scale 
            window[2] = window[2]/scale 
            window[3] = window[3]/scale      
        else: 
            masks = r['masks'][window[0]:window[2],window[1]:window[3],:] 
        rois = rois - [window[0],window[1],window[0],window[1]]
        obj_orders = np.array(r['class_ids'])-1
        obj_ids = model_ids[obj_orders] 
        #now c_ids are the same annotation those of the names of ply/gt files
        scores = np.array(r['scores'])
        return rois,obj_orders,obj_ids,scores,masks

elif detect_type=='retinanet':
    detection_dir=cfg['path_to_detection_pipeline']
    sys.path.append(detection_dir)
    from keras_retinanet import models
    from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
    from keras_retinanet.utils.visualization import draw_box, draw_caption
    from keras_retinanet.utils.colors import label_color
    def get_retinanet_detection(image_t,model):
        image = preprocess_image(image_t[:,:,::-1]) #needs bgr order bgr?
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        boxes = boxes[0]
        scores = scores[0]
        labels = labels[0]
        
        score_mask = scores>0
        if(np.sum(score_mask)==0):
            return np.array([[-1,-1,-1,-1]]),-1,-1,-1

        else:            
            scores = scores[score_mask]
            boxes =  boxes[score_mask]
            labels =  labels[score_mask]
            
            rois = np.zeros((boxes.shape[0],4),np.int)
            rois[:,0] = boxes[:,1]
            rois[:,1] = boxes[:,0]
            rois[:,2] = boxes[:,3]
            rois[:,3] = boxes[:,2]
            obj_orders = labels 
            obj_ids = model_ids[obj_orders]            

            return rois,obj_orders,obj_ids,scores
    
else:
    print("Unknown detection type")
    sys.exit()

score_type = cfg["score_type"]
#1-scores from a 2D detetion pipeline is used (used for the paper)
#2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, sued for the BOP challenge)

task_type = cfg["task_type"]
#1-Output all results for target object in the given scene
#2-ViVo task (2019 BOP challenge format, take the top-n instances)
cand_factor =float(cfg['cand_factor'])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

dataset=sys.argv[3]
vis=False
output_dir = cfg['path_to_output']


bop_dir,test_dir,model_plys,\
model_info,model_ids,rgb_files,\
depth_files,mask_files,gts,\
cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True,train=False)

im_width,im_height =cam_param_global['im_size'] 
cam_K = cam_param_global['K']
model_params =inout.load_json(os.path.join(bop_dir+"/models_xyz/",cfg['norm_factor_fn']))

if(gpu_rendering):
    ren = cuda_renderer(res_y=im_height,res_x=im_width)
    ren.set_cam(cam_K)
else:
    ren = Renderer((im_width,im_height),cam_K)
    ren.set_cam(cam_K)

depth_adjust=True

if(dataset=='itodd'):
    img_type='gray'
else:
    img_type='rgb'
    

if("target_obj" in cfg.keys()):
    target_obj = cfg['target_obj']
    remove_obj_id=[]
    incl_obj_id=[]
    for m_id,model_id in enumerate(model_ids):
        if(model_id not in target_obj):
            remove_obj_id.append(m_id)
        else:
            incl_obj_id.append(m_id)
    for m_id in sorted(remove_obj_id,reverse=True):
        print("Remove a model:",model_ids[m_id])
        del model_plys[m_id]
        del model_info['{}'.format(model_ids[m_id])]
        
    model_ids = model_ids[incl_obj_id]
    

print("Camera info-----------------")
print(im_width,im_height)
print(cam_K)
print("-----------------")

'''
standard estimation parameter for pix2pose
'''

th_outlier=cfg['outlier_th']
dynamic_th = True
if(type(th_outlier[0])==list):
    print("Individual outlier thresholds are applied for each object")
    dynamic_th=False
    th_outliers = np.squeeze(np.array(th_outlier))
th_inlier=cfg['inlier_th']
th_ransac=3

dummy_run=True
MODEL_DIR = os.path.join(bop_dir, "weight_detection")
if(detect_type=='rcnn'):
    #Load mask r_cnn
    '''
    standard estimation parameter for Mask R-CNN (identical for all dataset)
    '''
    config = BopInferenceConfig(dataset=dataset,
                            num_classes=model_ids.shape[0]+1,#1+len(model_plys),
                            im_width=im_width,im_height=im_height)
    config.DETECTION_MAX_INSTANCES=30
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config,
                            model_dir=MODEL_DIR)
    last_path = model.find_last()
    #Load the last model you trained and continue training
    model.load_weights(last_path, by_name=True)
elif(detect_type=='retinanet'):
    print("Loading weights for keras-retinanet")
    detection_weight_fn = cfg['detection_weight']
    model = models.load_model(os.path.join(MODEL_DIR,detection_weight_fn), backbone_name='resnet50')


'''
Load pix2pose inference weights
'''
load_partial=False
obj_pix2pose=[]
obj_names=[]
obj_models=[]
image_dummy=np.zeros((im_height,im_width,3),np.uint8)
if( 'backbone' in cfg.keys()):
    backbone = cfg['backbone']
else:
    backbone = 'paper'

for m_id,model_id in enumerate(model_ids):
    model_param = model_params['{}'.format(model_id)]
    obj_param=bop_io.get_model_params(model_param)
    weight_dir = bop_dir+"/pix2pose_weights/{:02d}".format(model_id)
    #weight_dir = "/home/kiru/media/hdd/weights/tless/tless_{:02d}".format(model_id)
    weight_fn = os.path.join(weight_dir,"inference.hdf5")
    print("load pix2pose weight for obj_{} from".format(model_id),weight_fn)
    if not(dynamic_th):
        th_outlier = [th_outliers[m_id]] #provid a fixed outlier value
        print("Set outlier threshold to ",th_outlier[0])    
    recog_temp = recog.pix2pose(weight_fn,camK= cam_K,
                                res_x=im_width,res_y=im_height,obj_param=obj_param,
                                th_ransac=th_ransac,th_outlier=th_outlier,
                                th_inlier=th_inlier,backbone=backbone)
    obj_pix2pose.append(recog_temp)    
    obj_names.append(model_id)
    ply_fn=model_plys[m_id]    
    if(gpu_rendering):
        obj_model = inout.load_ply(ply_fn)
        obj_model['pts']=obj_model['pts']*0.001 #mm to m scale
    else:
        obj_model = Model3D()
        obj_model.load(ply_fn,scale=0.001)#mm to m scale
    obj_models.append(obj_model)


test_target_fn = cfg['test_target']
target_list = bop_io.get_target_list(os.path.join(bop_dir,test_target_fn+".json"))

prev_sid=-1
result_dataset=[]

model_ids_list = model_ids.tolist()

if(dummy_run):
    #to activate networks before atual recognition
    #since only the first run takes longer time
    image_t = np.zeros((im_height,im_width,3),np.uint8)
    if(detect_type=='rcnn'):
        rois,obj_orders,obj_ids,scores,masks = get_rcnn_detection(image_t,model)
    elif(detect_type=='retinanet'):
        rois,obj_orders,obj_ids,scores = get_retinanet_detection(image_t,model)


for scene_id,im_id,obj_id_targets,inst_counts in target_list:
    print("Recognizing scene_id:{}, im_id:{}".format(scene_id,im_id))
    if(prev_sid!=scene_id):
        cam_path = test_dir+"/{:06d}/scene_camera.json".format(scene_id)
        cam_info = inout.load_scene_camera(cam_path)
        if(dummy_run):
            image_t = np.zeros((im_height,im_width,3),np.uint8)        
            for obj_id_target in obj_id_targets: #refreshing
                _,_,_,_,_,_ = obj_pix2pose[model_ids_list.index(obj_id_target)].est_pose(image_t,np.array([0,0,128,128],np.int))    
    
    prev_sid=scene_id #to avoid re-load scene_camera.json
    cam_param = cam_info[im_id]
    cam_K = cam_param['cam_K']
    depth_scale = cam_param['depth_scale'] #depth/1000 * depth_scale
    
    if(img_type=='gray'):
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.tif".format(im_id)
        image_gray = inout.load_im(rgb_path)
        #copy gray values to three channels    
        image_t = np.zeros((image_gray.shape[0],image_gray.shape[1],3),dtype=np.uint8)
        image_t[:,:,:]= np.expand_dims(image_gray,axis=2)
    else:
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.png".format(im_id)
        image_t = inout.load_im(rgb_path)        
    depth_path = test_dir+"/{:06d}/depth/{:06d}.png".format(scene_id,im_id)    
    depth_t = inout.load_depth(depth_path)/1000*depth_scale
    depth_valid = np.logical_and(depth_t>0.2, depth_t<5)

    points_tgt = np.zeros((depth_t.shape[0],depth_t.shape[1],6),np.float32)
    points_tgt[:,:,:3] = getXYZ(depth_t,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])
    points_tgt[:,:,3:] = get_normal(depth_t,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2],refine=True)

    t1=time.time()
    inst_count_est=np.zeros((len(inst_counts)))
    inst_count_pred = np.zeros((len(inst_counts)))
    
    if(detect_type=='rcnn'):
        rois,obj_orders,obj_ids,scores,masks = get_rcnn_detection(image_t,model)
    elif(detect_type=='retinanet'):
        rois,obj_orders,obj_ids,scores = get_retinanet_detection(image_t,model)
    
    
    result_score=[]
    result_objid=[]
    result_R=[]
    result_t=[]
    result_roi=[]
    result_img=[]
    vis=False

    for r_id,roi in enumerate(rois):
        if(roi[0]==-1 and roi[1]==-1):
            continue
        obj_id = obj_ids[r_id]        
        if not(obj_id in obj_id_targets):
            #skip if the detected object is not in the target object
            continue           
        obj_gt_no = obj_id_targets.index(obj_id)
        if(inst_count_pred[obj_gt_no]>inst_counts[obj_gt_no]*cand_factor):
          continue
        inst_count_pred[obj_gt_no]+=1

        obj_order_id = obj_orders[r_id]
        obj_pix2pose[obj_order_id].camK=cam_K.reshape(3,3)                
        img_pred,mask_pred,rot_pred,tra_pred,frac_inlier,bbox_t =\
        obj_pix2pose[obj_order_id].est_pose(image_t,roi.astype(np.int))            
        if(frac_inlier==-1):
            continue        

        '''
        ICP Refinement (when using depth)
        '''
        if(tra_pred[2]/1000<0.2):
                continue
            
        if(score_type==2 and detect_type=='rcnn'):
            mask_from_detect = masks[:,:,r_id]    
            union_mask = np.logical_or(mask_from_detect,mask_pred)
            union_mask = np.logical_and(union_mask,depth_valid)
            union = np.sum(union_mask)
            if(union<=30):
                mask_iou=0
                continue
            else:
                mask_iou = np.sum(np.logical_and(mask_from_detect,mask_pred))/union
           
            pts_tgt = points_tgt[union_mask]
            tf,residual = icp_refinement(pts_tgt,obj_models[obj_order_id],rot_pred,tra_pred,cam_K,ren)
            if(residual==-1):
                continue
            rot_pred_refined =tf[:3,:3]
            tra_pred_refined =tf[:3,3]*1000 
            if(gpu_rendering):                
                img_ref,depth_ref = render_obj_gpu(obj_models[obj_order_id],rot_pred_refined,tra_pred_refined/1000,cam_K,ren)
            else:
                img_ref,depth_ref = render_obj(obj_models[obj_order_id],rot_pred_refined,tra_pred_refined/1000,cam_K,ren)
                
            diff_mask = np.abs(depth_ref[union_mask]-depth_t[union_mask])<0.02
            ratio = np.sum(diff_mask)/union            
            score =scores[r_id]*np.sum(diff_mask)#ycbv
            rot_pred = rot_pred_refined
            tra_pred = tra_pred_refined
        else:
            score = scores[r_id]    

        result_score.append(score)        
        result_objid.append(obj_id)
        result_R.append(rot_pred)
        result_t.append(tra_pred)        
    
    if len(result_score)>0:
        result_score = np.array(result_score)
        result_score = result_score/np.max(result_score) #normalize
        sorted_id = np.argsort(1-result_score) #sort results
        time_spend =time.time()-t1 #ends time for the computation
        total_inst=0
        n_inst = np.sum(inst_counts)
    else:
        continue    
    
    for result_id in sorted_id:
        total_inst+=1
        if(task_type=='2' and total_inst>n_inst): #for vivo task
            break        
        obj_id = result_objid[result_id]
        R = result_R[result_id].flatten()
        t = (result_t[result_id]).flatten()
        score = result_score[result_id]
        obj_gt_no = obj_id_targets.index(obj_id)
        inst_count_est[obj_gt_no]+=1
        if(task_type=='2' and inst_count_est[obj_gt_no]>inst_counts[obj_gt_no]):
            #skip if the result exceeds the amount of taget instances for vivo task
            continue
        result_temp ={'scene_id':scene_id,'im_id': im_id,'obj_id':obj_id,'score':score,'R':R,'t':t,'time':time_spend }
        result_dataset.append(result_temp)


if(dataset=='tless'):
    output_path = os.path.join(output_dir,"pix2pose-iccv19_"+dataset+"-test-primesense.csv")
else:
    output_path = os.path.join(output_dir,"pix2pose-iccv19_"+dataset+"-test.csv")

print("Saving the result to ",output_path)
inout.save_bop_results(output_path,result_dataset)



