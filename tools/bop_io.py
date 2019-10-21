import numpy as np
import json
import os,sys
sys.path.append(".")
sys.path.append("./bop_toolkit")
from bop_toolkit_lib import inout
from bop_toolkit_lib import renderer

def get_target_list(target_path):
    targets = inout.load_json(target_path)
    prev_imid=-1
    prev_sid=-1
    target_list=[]
    for i in range(len(targets)):
        tgt = targets[i]    
        im_id = tgt['im_id']
        inst_count = tgt['inst_count']
        obj_id = tgt['obj_id']
        scene_id = tgt['scene_id']
        if(prev_imid!=im_id or prev_sid!=scene_id):
            if(prev_imid!=-1):
                target_list.append([prev_sid,prev_imid,obj_ids,inst_counts])
            obj_ids= [obj_id]
            inst_counts= [inst_count]
        else:
            obj_ids.append(obj_id)
            inst_counts.append(inst_count)
        prev_imid=im_id
        prev_sid=scene_id
    target_list.append([prev_sid,prev_imid,obj_ids,inst_counts]) #append the list image
    return target_list

def get_model_params(model_param):
    obj_param = np.zeros((6))
    obj_param[0]=model_param['x_scale']
    obj_param[1]=model_param['y_scale']
    obj_param[2]=model_param['z_scale']

    obj_param[3]=model_param['x_ct']
    obj_param[4]=model_param['y_ct']
    obj_param[5]=model_param['z_ct']
    return obj_param


def get_dataset(cfg,dataset,train=True,incl_param=False,eval=False,eval_model=False):
    #return serialized datset information
    bop_dir = cfg['dataset_dir']
    if eval_model:
        postfix_model = '_eval'
    else:
        postfix_model = ''
    if(dataset=='lmo'):
      bop_dataset_dir = os.path.join(bop_dir,"lmo")
      test_dir = bop_dataset_dir+"/test"
      train_dir = bop_dataset_dir+"/train"
      model_dir = bop_dataset_dir+"/models"+postfix_model
      model_scale=0.001
    elif(dataset=='ruapc'):
      bop_dataset_dir = os.path.join(bop_dir,"ruapc")
      test_dir = bop_dataset_dir+"/test"
      train_dir = bop_dataset_dir+"/train"
      model_dir = bop_dataset_dir+"/models"+postfix_model
      model_scale=0.001
    elif(dataset=='hb'):
      bop_dataset_dir = os.path.join(bop_dir,"hb")
      test_dir = bop_dataset_dir+"/test"
      train_dir = bop_dataset_dir+"/train"
      model_dir = bop_dataset_dir+"/models"+postfix_model
      model_scale=0.0001
    elif(dataset=='icbin'):
        bop_dataset_dir = os.path.join(bop_dir,"icbin")
        test_dir = bop_dataset_dir+"/test"
        train_dir = bop_dataset_dir+"/train"
        model_dir = bop_dataset_dir+"/models"+postfix_model
        model_scale=0.001
    elif(dataset=='itodd'):
        bop_dataset_dir = os.path.join(bop_dir,"itodd")
        test_dir = bop_dataset_dir+"/test"
        train_dir = bop_dataset_dir+"/train"
        model_dir = bop_dataset_dir+"/models"+postfix_model
        model_scale=0.001
    elif(dataset=='tudl'):
        bop_dataset_dir = os.path.join(bop_dir,"tudl")
        test_dir = bop_dataset_dir+"/test"
        train_dir = bop_dataset_dir+"/train_real"
        model_dir = bop_dataset_dir+"/models"+postfix_model
        model_scale=0.001
    elif(dataset=='tless'):
        bop_dataset_dir = os.path.join(bop_dir,"tless")
        test_dir = bop_dataset_dir+"/test_primesense"
        train_dir = bop_dataset_dir+"/train_primesense"
        if not(train) and not(eval_model):
            model_dir = bop_dataset_dir+"/models_reconst" #use this only for vis
        elif eval_model:
            model_dir = bop_dataset_dir+"/models_eval"
        else:
            model_dir = bop_dataset_dir+"/models_cad"
        model_scale=0.001
    elif(dataset=='ycbv'):
        bop_dataset_dir = os.path.join(bop_dir,"ycbv")
        test_dir = bop_dataset_dir+"/test"
        train_dir = bop_dataset_dir+"/train"
        model_dir = bop_dataset_dir+"/models"+postfix_model
        model_scale=0.001
    elif(dataset=='lm'):
        bop_dataset_dir = os.path.join(bop_dir,"lm")
        test_dir = bop_dataset_dir+"/test"
        train_dir = bop_dataset_dir+"/train"
        model_dir = bop_dataset_dir+"/models"+postfix_model
        model_dir  = "/home/kiru/media/hdd_linux/PoseDataset/hinterstoisser/model_eval"
        model_scale=0.001
    
    model_info = inout.load_json(os.path.join(model_dir,"models_info.json"))
    if(dataset=='ycbv'):
        cam_param_global = inout.load_cam_params(os.path.join(bop_dataset_dir,"camera_uw.json"))
    else:
        cam_param_global = inout.load_cam_params(os.path.join(bop_dataset_dir,"camera.json"))
    
    im_size=np.array(cam_param_global['im_size'])[::-1]
    
    model_plys=[]
    rgb_files=[]
    depth_files=[]
    mask_files=[]
    gts=[]
    params=[]
    model_ids = []
    for model_id in model_info.keys():
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        if(os.path.exists(ply_fn)): model_ids.append(int(model_id)) #add model id only if the model.ply file exists

    model_ids = np.sort(np.array(model_ids))
    for model_id in model_ids:
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        model_plys.append(ply_fn)
        print(model_id,ply_fn)
    print("if models are not fully listed above, please make sure there are ply files available")
    if(train): 
        target_dir =train_dir    
        if(os.path.exists(target_dir)):        
            for dir in os.listdir(target_dir): #loop over a seqeunce 
                current_dir = target_dir+"/"+dir
                if os.path.exists(os.path.join(current_dir,"scene_camera.json")):
                    scene_params = inout.load_scene_camera(os.path.join(current_dir,"scene_camera.json"))            
                    scene_gt_fn = os.path.join(current_dir,"scene_gt.json")
                    has_gt=False
                    if os.path.exists(scene_gt_fn):
                        scene_gts = inout.load_scene_gt(scene_gt_fn)
                        has_gt=True
                    for img_id in sorted(scene_params.keys()):
                        im_id = int(img_id)
                        if(dataset=="itodd" and not(train)):
                            rgb_fn = os.path.join(current_dir+"/gray","{:06d}.tif".format(im_id))
                        else:
                            rgb_fn = os.path.join(current_dir+"/rgb","{:06d}.png".format(im_id))
                        depth_fn = os.path.join(current_dir+"/depth","{:06d}.png".format(im_id))
                        if(train):
                            if(dataset=='hb' or dataset=='itodd' or dataset=='ycbv'):
                                mask_fn = os.path.join(current_dir+"/mask","{:06d}.png".format(im_id))
                            else:
                                mask_fn = os.path.join(current_dir+"/mask","{:06d}_000000.png".format(im_id))
                            mask_files.append(mask_fn)
                        rgb_files.append(rgb_fn)
                        depth_files.append(depth_fn)
                        if(has_gt):gts.append(scene_gts[im_id])
                        params.append(scene_params[im_id])
    else:
        target_dir =test_dir    
                 
    if(incl_param):
        return bop_dataset_dir,target_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global,params
    else:
        return bop_dataset_dir,target_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global
