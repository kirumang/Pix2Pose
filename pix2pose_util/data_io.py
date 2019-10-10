import os
import skimage
from skimage.filters import gaussian
from skimage import io
from skimage.transform import resize,rotate
from skimage.filters import gaussian
from imgaug import augmenters as iaa

import numpy as np
import random

class data_generator():
    def __init__(self,data_dir, back_dir,
                 batch_size=50,gan=True,imsize=128,
                 res_x=640,res_y=480,
                 **kwargs):
        '''
        data_dir: Folder that contains cropped image+xyz
        back_dir: Folder that contains random background images
            batch_size: batch size for training
        gan: if False, gt for GAN is not yielded
        '''
        self.data_dir = data_dir
        self.back_dir = back_dir
        self.imsize=imsize
        self.batch_size = batch_size
        self.gan = gan
        self.backfiles = os.listdir(back_dir)
        data_list = os.listdir(data_dir)
        self.datafiles=[]
        self.res_x=res_x
        self.res_y=res_y

        for file in data_list:
            if(file.endswith(".npy")):
                self.datafiles.append(file)

        self.n_data = len(self.datafiles)
        self.n_background = len(self.backfiles)
        print("Total training views:", self.n_data)

        self.seq_syn= iaa.Sequential([
                                    iaa.WithChannels(0, iaa.Add((-15, 15))),
                                    iaa.WithChannels(1, iaa.Add((-15, 15))),
                                    iaa.WithChannels(2, iaa.Add((-15, 15))),
                                    iaa.ContrastNormalization((0.8, 1.3)),
                                    iaa.Multiply((0.8, 1.2),per_channel=0.5),
                                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                                    iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
                                    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
                                    ], random_order=True)

    def get_patch_pair(self,v_id,batch_count):
        imgs = np.load(os.path.join(self.data_dir,self.datafiles[v_id])).astype(np.float32)
        is_real=False
        if imgs.shape[2]==7:
            #this is real image
            p_vis_mask = imgs[:,:,6]>0
            is_real=True
            

        real_img =imgs[:,:,:3]/255
        p_xyz =imgs[:,:,3:6]/255

        p_height = p_xyz.shape[0]
        p_width = p_xyz.shape[1]
        p_mask_no_occ = np.sum(p_xyz,axis=2)>0
        p_xyz[np.invert(p_mask_no_occ)]=[0.5,0.5,0.5]


        back_fn = self.backfiles[int(random.random()*(self.n_background-1))]
        back_img = io.imread(self.back_dir+"/"+back_fn)
        if back_img.ndim != 3:
            back_img = skimage.color.gray2rgb(back_img)
        back_img = back_img.astype(np.float32)/255
        need_resize=False
        desired_size_h=desired_size_w=0
        if(back_img.shape[0] < p_xyz.shape[0]*2):
            desired_size_h=p_xyz.shape[0]*2
            need_resize=True
        if(back_img.shape[1] < p_xyz.shape[1]*2):
            desired_size_w=p_xyz.shape[1]*2
            need_resize=True
        if(need_resize):
            back_img = resize(back_img,(max(desired_size_h,back_img.shape[0]),max(desired_size_w,back_img.shape[1])),order=1,mode='reflect')

        img_augmented = self.seq_syn.augment_image(real_img*255)/255
        v_limit = back_img.shape[0]-real_img.shape[0]-20
        u_limit = back_img.shape[1]-real_img.shape[1]-20
        v_ref =int(random.random()*v_limit+10)
        u_ref =int(random.random()*u_limit+10)
        #combine two image
        p_back_img = back_img[v_ref:v_ref+p_height,u_ref:u_ref+p_width]
        img_augmented[np.invert(p_mask_no_occ)]=p_back_img[np.invert(p_mask_no_occ)]

        image_ref = np.copy(back_img)
        image_ref[v_ref:v_ref+p_height,u_ref:u_ref+p_width]=img_augmented

        xyz = np.ones((image_ref.shape[0],image_ref.shape[1],3))*[0.5,0.5,0.5]
        xyz[v_ref:v_ref+p_height,u_ref:u_ref+p_width]=p_xyz


        image =np.copy(image_ref)
        mask_no_occ = np.zeros((image_ref.shape[0],image_ref.shape[1]),bool)
        mask_no_occ[v_ref:v_ref+p_height,u_ref:u_ref+p_width] = p_mask_no_occ
        mask_no_occ_ori = np.copy(mask_no_occ)
        mask_visible=np.copy(mask_no_occ)
        mask_foreground = np.zeros_like(mask_visible)

        bbox = np.array([v_ref,u_ref,v_ref+p_height,u_ref+p_width])

        bbox_ct_v = int((bbox[0]+bbox[2])/2 + (random.random()*10-5)) #-5~5
        bbox_ct_u = int((bbox[1]+bbox[3])/2 + (random.random()*10-5)) #-5~5
        width = (bbox[3]-bbox[1])*(1+ (random.random()*0.6-0.3))
        height = (bbox[2]-bbox[0])*(1+(random.random()*0.6-0.3))
        max_wh = max(width*1.5,height*1.5)
        h = max_wh
        w = max_wh

        v1 = bbox_ct_v-int(h/2)
        v2 = bbox_ct_v+int(h/2)
        u1 = bbox_ct_u-int(w/2)
        u2 = bbox_ct_u+int(w/2)

        base_image = np.zeros((v2-v1,u2-u1,3))
        mask_image = np.zeros((v2-v1,u2-u1))

        base_image_depth =np.zeros((v2-v1,u2-u1))
        base_image_dx = np.zeros((v2-v1,u2-u1))
        base_image_dy = np.zeros((v2-v1,u2-u1))


        tgt_image = np.zeros((v2-v1,u2-u1,3))

        shift_v_min=0
        shift_u_min=0
        shift_v_max=0
        shift_u_max=0

        if(v1<0):
            shift_v_min=np.abs(v1)
            v1 = 0

        if(v2>image.shape[0]):
            shift_v_max =-np.abs(v2-image.shape[0])
            v2 = image.shape[0]

        if(u1<0):
            shift_u_min=np.abs(u1)
            u1=0

        if(u2>image.shape[1]):
            shift_u_max=-np.abs(u2-image.shape[1])
            u2=image.shape[1]

        h_aug = int( (random.random()*0.5+0.2)*h) #~0.3 for dcgan7/occ7, 0.1~0.5 for occ8
        w_aug = int( (random.random()*0.5+0.2)*w)
        bbox_ct_v_t = int((bbox[0]+bbox[2])/2 )
        bbox_ct_u_t = int((bbox[1]+bbox[3])/2 )
        ratio = 0.5
        d_pos_v = int(bbox_ct_v_t+(random.random()*ratio*2-ratio)*height) #-0.5~0.5 from center
        d_pos_u = int(bbox_ct_u_t+(random.random()*ratio*2-ratio)*width)  #-0.5~0.5 from center
        if(h_aug > 0 and w_aug>0):
            mask_no_occ[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=0
            mask_visible[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=0
            mask_foreground[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=1

        mask_b = np.invert(mask_no_occ)
        back_v = min(back_img.shape[0],image.shape[0])
        back_u = min(back_img.shape[1],image.shape[1])
        if(len(back_img.shape)==3):
            image[:back_v,:back_u]=back_img[:back_v,:back_u]
        elif(len(back_img.shape)==2):
            image[:back_v,:back_u]=np.expand_dims(back_img[:back_v,:back_u],axis=2)

        if(image.shape[0]>back_img.shape[0]):
            width = min(image.shape[0]-back_v,back_img.shape[0])
            image[back_v:back_v+width,:back_u]=back_img[:width,:back_u]
        if(image.shape[1]>back_img.shape[1]):
            height = min(image.shape[1]-back_u,back_img.shape[1])
            image[:back_v,back_u:back_u+height]=back_img[:back_v,:height]

        #expand mask_no_occ or remove somearea
        #get the boundary of mask_no_occ

        image[mask_no_occ]=image_ref[mask_no_occ]
        boundary_full = np.zeros_like(mask_no_occ)
        img_blurred = np.zeros((image.shape[0],image.shape[1],3))
        p= np.gradient(mask_no_occ[v1:v2,u1:u2].astype(float))
        boundary = np.logical_or(p[0]>0,p[1]>0)
        sigma = random.random()*2
        boundary_full[v1:v2,u1:u2] = gaussian(boundary.astype(float),sigma=sigma)>0
        sigma = random.random()*2
        img_blurred[v1:v2,u1:u2] = gaussian(image[v1:v2,u1:u2],sigma=sigma)
        image[boundary_full] = img_blurred[boundary_full]

        c_img =(xyz-[0.5,0.5,0.5])/0.5
        #background: max(noise + max_depth value,normalization bound)
        #occluded_part : min(nois-(original_depth-0.1), original_depth-0.05)
        mask_wrong_back=np.zeros_like((mask_no_occ_ori))

        cutcut=True
        if (batch_count%2==0): #no gray background, 1-stage with prob
            mask_temp = (mask_no_occ_ori[v1:v2,u1:u2]).astype(np.float)#from glu
            #mask_temp = (mask_no_occ[v1:v2,u1:u2]).astype(np.float)
            sigma_ran= min(max(random.gauss(0.5,0.3),0.1),1.0)
            mask_temp = gaussian(mask_temp,sigma=sigma_ran)
            mask_temp = (mask_temp>0).astype(bool)
            #[cutcut]
            if(cutcut):
                c_img_gaus = gaussian(c_img[v1:v2,u1:u2],sigma=sigma_ran)
                radius = np.linalg.norm(c_img_gaus,axis=2)
                non_gray = radius>0.3
                mask_temp =np.logical_and(mask_temp,non_gray)#[cutcut]

            image_no_mask_zero = image[v1:v2,u1:u2]
            background_mask = np.invert(mask_temp)
            image_no_mask_zero[background_mask]=[0.5,0.5,0.5]

                #again random occlusion gray #simulate: when some parts are missed by the first stage
            h_aug = int( (random.random()*0.5+0.0)*h)
            w_aug = int( (random.random()*0.5+0.0)*w)
            bbox_ct_v_t = int((bbox[0]+bbox[2])/2 )
            bbox_ct_u_t = int((bbox[1]+bbox[3])/2 )
            ratio = 0.5
            d_pos_v = int(bbox_ct_v_t+(random.random()*ratio*2-ratio)*height) #-0.5~0.5 from center
            d_pos_u = int(bbox_ct_u_t+(random.random()*ratio*2-ratio)*width)  #-0.5~0.5 from center
            if(h_aug > 0 and w_aug>0):
                #remove 80% of them randomly
                #foreground occlusion
                mask_bye=np.zeros_like((mask_no_occ_ori))
                cropped = mask_bye[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]
                mask_bye[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=1 #caution
                #mask_foreground[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=1
                mask_visible[mask_bye]=0
                image_no_mask_zero[mask_bye[v1:v2,u1:u2]]=[0.5,0.5,0.5]

            #background inclusion
            h_aug = int( (random.random()*0.5+0.0)*h)
            w_aug = int( (random.random()*0.5+0.0)*w)
            bbox_ct_v_t = int((bbox[0]+bbox[2])/2 )
            bbox_ct_u_t = int((bbox[1]+bbox[3])/2 )
            ratio = 0.5
            d_pos_v = int(bbox_ct_v_t+(random.random()*ratio*2-ratio)*height) #-0.5~0.5 from center
            d_pos_u = int(bbox_ct_u_t+(random.random()*ratio*2-ratio)*width)  #-0.5~0.5 from center
            if(h_aug > 0 and w_aug>0):
                mask_wrong_back[d_pos_v:d_pos_v+h_aug,d_pos_u:d_pos_u+w_aug]=1
                mask_wrong_back_inter = np.logical_and(mask_wrong_back,np.invert(mask_no_occ_ori))[v1:v2,u1:u2]
                image_crop =image_ref[v1:v2,u1:u2]
                image_no_mask_zero[mask_wrong_back_inter==1]=image_crop[mask_wrong_back_inter==1]
            image_no_mask_zero = (image_no_mask_zero-[0.5,0.5,0.5])/0.5
        else:
            #add gaussian blur on the boundary
            image_no_mask_zero = image[v1:v2,u1:u2]
            image_no_mask_zero = (image_no_mask_zero-[0.5,0.5,0.5])/0.5


        #random flip and rotation is also possible --> good for you!
        base_image[shift_v_min:shift_v_max+base_image.shape[0],shift_u_min:shift_u_max+base_image.shape[1]] = image_no_mask_zero
         #0~255 -> -128 to 128 -> -1 to 1
        tgt_image[shift_v_min:shift_v_max+tgt_image.shape[0],shift_u_min:shift_u_max+tgt_image.shape[1]]=c_img[v1:v2,u1:u2]
        mask_image[shift_v_min:shift_v_max+base_image.shape[0],shift_u_min:shift_u_max+base_image.shape[1]] = mask_no_occ_ori[v1:v2,u1:u2]
        #rotate -15 to 15 degree
        r_angle = random.random()*30-15
        base_image = rotate(base_image, r_angle,mode='reflect')
        tgt_image =  rotate(tgt_image, r_angle,mode='reflect')

        mask_area_crop = rotate(mask_image.astype(np.float),r_angle)

        src_image_resized = resize(base_image,(self.imsize,self.imsize),order=1,mode='reflect')
        tgt_image_resized = resize(tgt_image,(self.imsize,self.imsize),order=1,mode='reflect')
        mask_area_resized = resize(mask_area_crop,(self.imsize,self.imsize),order=1,mode='reflect')

        return src_image_resized,tgt_image_resized,mask_area_resized


    def generator(self):
        scene_seq = np.arange(self.n_data)
        np.random.shuffle(scene_seq)
        idx=0
        batch_index=0
        batch_count=0
        batch_src =np.zeros((self.batch_size,self.imsize,self.imsize,3)) #templates
        batch_tgt =np.zeros((self.batch_size,self.imsize,self.imsize,3)) #templates
        batch_tgt_disc =np.zeros((self.batch_size))
        batch_prob = np.zeros((self.batch_size,self.imsize,self.imsize,1))
        batch_mask = np.zeros((self.batch_size,self.imsize,self.imsize,1))

        batch_tgt_disc[:]=1
        while True:
            v_id = scene_seq[idx]
            idx+=1
            if(idx >= scene_seq.shape[0]):
                idx=0
                np.random.shuffle(scene_seq)

            s_img,t_img,mask_area = self.get_patch_pair(v_id,batch_count)
            batch_src[batch_index] = s_img

            batch_tgt[batch_index] =t_img
            batch_prob[batch_index,:,:,0] =mask_area
            batch_index+=1
            if(batch_index >= self.batch_size):
                batch_index=0
                batch_count+=1
                if(batch_count>=100):
                    batch_count=0
                if(self.gan):
                    yield batch_src, batch_tgt ,batch_tgt_disc,batch_prob
                else:
                    yield batch_src, batch_tgt

                    
                    
                    
