import keras
from pix2pose_model import ae_model as ae
from keras import backend as K
import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model

class pix2pose():
    def __init__(self,weight_fn,camK,res_x,res_y,obj_param,th_ransac=3.0,th_outlier=[0.1,0.2,0.3],th_inlier=0.1,box_size=1.5,dist_coeff=None,backbone="paper",**kwargs):
        self.camK=camK
        self.res_x= res_x
        self.res_y = res_y
        self.th_ransac = th_ransac
        self.th_o= th_outlier
        self.th_i=th_inlier
        self.obj_scale = obj_param[:3] #x,y,z
        self.obj_ct = obj_param[3:] #x,y,z
        self.box_size= box_size
        self.dist_coeff = dist_coeff
        if(backbone=='paper'):
            self.generator_train = ae.aemodel_unet_prob(p=1.0) #output:3gae
            self.generator_train.load_weights(weight_fn)
        elif(backbone=='resnet50'):
            self.generator_train = ae.aemodel_unet_resnet50(p=1.0)
            self.generator_train.load_weights(weight_fn)            

    def get_boxes(self,bbox,v_max,u_max,ct=np.array([-1]),max_w=9999):
        if(ct[0]==-1):
            bbox_ct_v =int((bbox[0]+bbox[2])/2)
            bbox_ct_u =int((bbox[1]+bbox[3])/2)
        else:
            bbox_ct_v =ct[0]
            bbox_ct_u =ct[1]

        width = bbox[3]-bbox[1]
        height = bbox[2]-bbox[0]
        w = min(max_w,max(width*self.box_size,height*self.box_size))
        h = w
        v1_ori = bbox_ct_v-int(h/2)
        v2_ori = bbox_ct_v+int(h/2)
        u1_ori = bbox_ct_u-int(w/2)
        u2_ori = bbox_ct_u+int(w/2)

        shift_v_min=0
        shift_u_min=0
        shift_v_max=0
        shift_u_max=0
        v1=v1_ori
        v2=v2_ori
        u1=u1_ori
        u2=u2_ori
        if(v1_ori<0):
            shift_v_min=np.abs(v1_ori)
            v1 = 0
        if(v2_ori>v_max):
            shift_v_max =-np.abs(v2_ori-v_max)
            v2 = v_max
        if(u1_ori<0):
            shift_u_min=np.abs(u1_ori)
            u1=0
        if(u2_ori>u_max):
            shift_u_max=-np.abs(u2_ori-u_max)
            u2=u_max
        vv1 = shift_v_min
        vv2 = shift_v_max+(v2_ori-v1_ori)
        uu1 = shift_u_min
        uu2 = shift_u_max+(u2_ori-u1_ori)
        return v1_ori,v2_ori,u1_ori,u2_ori,v1,v2,u1,u2,vv1,vv2,uu1,uu2
    def est_pose(self,rgb,bbox,gt_trans=np.eye((4)),z_iter=False):
        v1_ori,v2_ori,u1_ori,u2_ori,v1,v2,u1,u2,vv1,vv2,uu1,uu2 = self.get_boxes(bbox,rgb.shape[0],rgb.shape[1])
        cx_o=(bbox[3]+bbox[1])/2
        cy_o=(bbox[2]+bbox[0])/2
        w_stage_1=v2_ori-v1_ori
        base_image = np.zeros((v2_ori-v1_ori,u2_ori-u1_ori,3))
        image_no_mask_zero = np.copy(rgb[v1:v2,u1:u2]).astype(np.float32)
        image_no_mask_zero = (image_no_mask_zero-[128,128,128])/128
        if(base_image.shape[0]<5 or base_image.shape[1]<5 or image_no_mask_zero.shape[0]<5 or image_no_mask_zero.shape[1]<5):
            return np.zeros((1)),-1,-1,-1,-1,np.array([v1,v2,u1,u2],np.int)
            
        base_image[vv1:vv2,uu1:uu2] = image_no_mask_zero
        input_img=resize(base_image,(128,128),order=1,mode='reflect')

        decode,prob = self.generator_train.predict( np.expand_dims(input_img,axis=0))
        img_pred =(decode[0]+1)/2
        img_pred[img_pred > 1] = 1
        img_pred[img_pred < 0] = 0

        non_gray = np.linalg.norm(decode[0],axis=2)>0.3
        n_init_mask = np.sum(non_gray)
        input_refined=[]
        box_refined=[]
        for th_o in self.th_o:
            prob_mask = prob[0,:,:,0]<th_o
            non_gray_prob = np.logical_and(non_gray,prob_mask)
            if np.sum(non_gray_prob)<10:
                continue
            vu_stage1 = np.where(non_gray)
            if(len(vu_stage1[0])==0):
                continue
            bbox = np.array([np.min(vu_stage1[0]),np.min(vu_stage1[1]),np.max(vu_stage1[0]),np.max(vu_stage1[1])])
            bbox = bbox*np.array([(v2_ori-v1_ori)/128,(u2_ori-u1_ori)/128,(v2_ori-v1_ori)/128,(u2_ori-u1_ori)/128])
            non_gray_ori = resize(non_gray_prob,(v2_ori-v1_ori,u2_ori-u1_ori),order=1,mode='constant',cval=0)>0.9
            non_gray_ori = non_gray_ori[vv1:vv2,uu1:uu2]
            bg_full = np.ones((rgb.shape[0],rgb.shape[1]),bool)
            bg_full[v1:v2,u1:u2]=np.invert(non_gray_ori)

            cx_m  = int((np.mean(vu_stage1[1])-(127/2))+cx_o)
            cy_m  = int((np.mean(vu_stage1[0])-(127/2))+cy_o)
            v1_ori_2,v2_ori_2,u1_ori_2,u2_ori_2,v1_2,v2_2,u1_2,u2_2,vv1_2,vv2_2,uu1_2,uu2_2 = self.get_boxes(bbox,rgb.shape[0],rgb.shape[1],ct=np.array([cy_m,cx_m]),max_w=w_stage_1)
            box_refined.append([v1_ori_2,v2_ori_2,u1_ori_2,u2_ori_2,v1_2,v2_2,u1_2,u2_2,vv1_2,vv2_2,uu1_2,uu2_2])
            
            base_image = np.zeros((v2_ori_2-v1_ori_2,u2_ori_2-u1_ori_2,3))
            image_no_mask_zero = np.copy(rgb[v1_2:v2_2,u1_2:u2_2])
            image_no_mask_zero = (image_no_mask_zero-[128,128,128])/128
            image_no_mask_zero[bg_full[v1_2:v2_2,u1_2:u2_2]]=0
            if(base_image.shape[0]<5 or base_image.shape[1]<5 or image_no_mask_zero.shape[0]<5 or image_no_mask_zero.shape[1]<5 or\
               base_image[vv1_2:vv2_2,uu1_2:uu2_2].shape[0]==0 or base_image[vv1_2:vv2_2,uu1_2:uu2_2].shape[1]==0):
                continue            
            base_image[vv1_2:vv2_2,uu1_2:uu2_2] = image_no_mask_zero
            input_img2=resize(base_image,(128,128),order=1,mode='reflect')
            input_refined.append(input_img2)
            #[todo] predict for 3 images at the same time and pnp ransac for 3 images, decide using the no. of inliers

        if(len(input_refined)<=0):
            #print("not valid output from the first stage")
            return img_pred,-1,-1,-1,-1,np.array([v1,v2,u1,u2],np.int)
        
        decode,prob = self.generator_train.predict( np.array(input_refined))
        max_inlier=-1
        min_dist=9999999
        for cand_id in range(len(input_refined)):
            v1_ori,v2_ori,u1_ori,u2_ori,v1,v2,u1,u2,vv1,vv2,uu1,uu2=box_refined[cand_id]
            img_prob_ori = resize(prob[cand_id,:,:,0],(v2_ori-v1_ori,u2_ori-u1_ori),order=1,mode='constant',cval=1)
            img_prob_ori = img_prob_ori[vv1:vv2,uu1:uu2]

            gray = np.linalg.norm(decode[cand_id],axis=2)<0.3 
            non_gray = np.invert(gray)
            decode[cand_id,gray,:]=0 #difference..

            img_pred =(decode[cand_id]+1)/2
            img_pred[img_pred > 1] = 1
            img_pred[img_pred < 0] = 0
            img_pred_ori = resize(img_pred,(v2_ori-v1_ori,u2_ori-u1_ori),order=1,mode='constant',cval=0.5)*255

            non_gray = resize(non_gray.astype(float),(v2_ori-v1_ori,u2_ori-u1_ori),order=1,mode='constant',cval=0)>0.9
            non_gray = non_gray[vv1:vv2,uu1:uu2]
            n_non_gray = np.sum(non_gray)
            if n_non_gray<10:                
                continue
            img_pred_ori = img_pred_ori[vv1:vv2,uu1:uu2]
            rgb_aug_test2 = np.zeros((rgb.shape[0],rgb.shape[1],3),np.uint8)
            rgb_aug_test2[v1:v2,u1:u2]=[128,128,128]
            rgb_aug_test2[v1:v2,u1:u2]=img_pred_ori

            rot_pred_cand,tra_pred_cand,valid_mask,n_inliers = self.pnp_ransac(rgb_aug_test2,img_prob_ori,non_gray,v1,v2,u1,u2)
            #print("n_inliers:",n_inliers,rot_pred_cand,tra_pred_cand)
            if True:
                non_gray_full = np.zeros((rgb.shape[0],rgb.shape[1]),bool)                  
                non_gray_full[v1:v2,u1:u2] = non_gray
                non_gray_vu = np.where(non_gray_full)                                        
                ct_pt = np.array([np.mean(non_gray_vu[0]),np.mean(non_gray_vu[1])]) 
                if(tra_pred_cand[2]==0):
                    dist=99999
                else:
                    proj_u = self.camK[0,0]*tra_pred_cand[0]/tra_pred_cand[2]+self.camK[0,2]
                    proj_v = self.camK[1,1]*tra_pred_cand[1]/tra_pred_cand[2]+self.camK[1,2]
                    dist = ((proj_v-ct_pt[0])**2 + (proj_u-ct_pt[1])**2)/(n_inliers+1E-6)
            
                if(dist <min_dist):                
                    rot_pred = rot_pred_cand
                    tra_pred = tra_pred_cand                
                    max_inlier = n_inliers
                    min_dist = dist
                    #frac of max_inlier
                    valid_mask_full = np.zeros((rgb.shape[0],rgb.shape[1]),bool)
                    valid_mask_full[v1:v2,u1:u2]=valid_mask
                    img_pred_f = img_pred_ori
            else:    
                n_inliers = n_inliers / n_non_gray
                if(n_inliers>max_inlier):
                    valid_mask_full = np.zeros((rgb.shape[0],rgb.shape[1]),bool)
                    valid_mask_full[v1:v2,u1:u2]=valid_mask
                    rot_pred = rot_pred_cand
                    tra_pred = tra_pred_cand
                    img_pred_f = img_pred_ori
                    max_inlier = n_inliers
                    #frac of max_inlier
        if(max_inlier==-1):
            #print("not valid max_inlier at the second stage")
            return img_pred,-1,-1,-1,-1,np.array([v1,v2,u1,u2],np.int)
        else:
            return img_pred_f.astype(np.uint8),valid_mask_full,rot_pred,tra_pred,max_inlier/n_init_mask,np.array([v1,v2,u1,u2],np.int)        

    def pnp_ransac(self,rgb_aug_test,img_prob_ori,non_zero,v1,v2,u1,u2):
            rgb_aug_crop =rgb_aug_test[v1:v2,u1:u2]
            xyz = np.copy(rgb_aug_crop)
            xyz  =xyz/255
            xyz = xyz*2-1
            xyz[:,:,0]=xyz[:,:,0]*self.obj_scale[0]+self.obj_ct[0]
            xyz[:,:,1]=xyz[:,:,1]*self.obj_scale[1]+self.obj_ct[1]
            xyz[:,:,2]=xyz[:,:,2]*self.obj_scale[2]+self.obj_ct[2]
            confidence_mask = img_prob_ori< self.th_i
            valid_mask = np.logical_and(non_zero ,confidence_mask)

            vu_list_s= np.where(valid_mask==1)
            n_pts_s=len(vu_list_s[0])
            img_pts_s = np.zeros((n_pts_s,2))
            obj_pts_s=xyz[vu_list_s[0],vu_list_s[1]]
            img_pts_s[:]=np.stack( (vu_list_s[1],vu_list_s[0]),axis=1) #u,v order
            img_pts_s[:,0]=img_pts_s[:,0]+u1
            img_pts_s[:,1]=img_pts_s[:,1]+v1
            img_pts_s = np.ascontiguousarray(img_pts_s[:,:2]).reshape((n_pts_s,1,2))
            if(n_pts_s <6):
                return np.eye(3),np.array([0,0,0]),valid_mask,-1
            ret, rvec, tvec,inliers = cv2.solvePnPRansac(obj_pts_s, img_pts_s, self.camK,None,\
                                      flags=cv2.SOLVEPNP_EPNP,reprojectionError=5,iterationsCount=100)
            if(inliers is None):
                return np.eye(3),np.array([0,0,0]),-1,-1
            else:
                rot_pred = np.eye(3)
                tra_pred = tvec[:,0]
                cv2.Rodrigues(rvec, rot_pred)                
                return rot_pred,tra_pred,valid_mask,len(inliers)

