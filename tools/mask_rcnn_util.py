
import os,sys
#Please specify the mask_rcnn directory [todo: using json for specify the folder]
#github:https://github.com/matterport/Mask_RCNN 
#MASKRCNN_DIR="/home/kiru/common_ws/Mask_RCNN_Mod/"
#sys.path.append(MASKRCNN_DIR)
#sys.path.append(".")
#sys.path.append("./bop_toolkit")
import numpy as np
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import log
import skimage

class BopDetectConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 50000/IMAGES_PER_GPU
    VALIDATION_STEPS = 5
    DETECTION_MIN_CONFIDENCE= 0.5

    def __init__(self, dataset,num_classes,im_width,im_height):
        self.NAME = dataset
        self.NUM_CLASSES = num_classes               
        self.IMAGE_MAX_DIM =max(im_width,im_height)#min(max(im_width,im_height),1024)#due to itodd
        self.IMAGE_MIN_DIM =min(im_width,im_height)#max(min(im_width,im_height),480)
        if(self.IMAGE_MAX_DIM%64>0):
            frac = int(self.IMAGE_MAX_DIM/64)+1
            self.IMAGE_MAX_DIM = frac*64 #set image to the nearest size that            
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM,self.IMAGE_MAX_DIM ])                
        super().__init__()        
        
class BopInferenceConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 1    
    VALIDATION_STEPS = 5
    DETECTION_MIN_CONFIDENCE=0.001
    DETECTION_MAX_INSTANCES=200
    DETECTION_NMS_THRESHOLD=0.7 #0.7
    
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_NMS_THRESHOLD = 0.9
    
    def __init__(self, dataset,num_classes,im_width,im_height):
        self.NAME = dataset
        self.NUM_CLASSES = num_classes
        self.IMAGE_MAX_DIM =max(im_width,im_height)
        self.IMAGE_MIN_DIM =min(im_width,im_height)
        if(self.IMAGE_MAX_DIM%64>0):
            frac = int(self.IMAGE_MAX_DIM/64)+1
            self.IMAGE_MAX_DIM = frac*64 #set image to the nearest size that            
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM,self.IMAGE_MAX_DIM ])
        super().__init__()

class BopDataset(utils.Dataset):
    def set_dataset(self, dataset,model_ids,train_dir):
        self.dataset = dataset
        self.model_ids = model_ids
        self.train_dir = train_dir
        self.n_class=self.model_ids.shape[0]
        for i in range(self.model_ids.shape[0]):
                self.add_class(self.dataset, i+1, "{:02d}".format(self.model_ids[i]))
    def load_dataset(self):
        self.class_map = self.model_ids
        self.gts=[]
        self.mask_fns=[]
        
        files = sorted(os.listdir(self.train_dir))
        n_img=0
        for i,file in enumerate(files):
            if file.endswith(".png") or file.endswith(".jpg"):
                img_path = os.path.join(self.train_dir,file)
                mask_fn = file[:-4]+".npy" #.replace(".png",".npy")
                mask_path = os.path.join(self.train_dir+"/mask/",mask_fn)
                self.add_image(self.dataset,image_id=n_img,path=img_path)
                self.mask_fns.append(mask_path)
                n_img+=1
        self.n_real = len(self.mask_fns)  
              
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        ##Real-time augmentation gogo

        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        return image
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == self.dataset:
            return info[self.dataset]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        mask_fn = self.mask_fns[image_id]
        mask = np.load(mask_fn)
        n_inst=0
        mask_gt = np.zeros((mask.shape[0],mask.shape[1],np.max(mask)+1),np.bool)
        class_ids = np.zeros((np.max(mask)+1),np.int32)
        '''
        for i in np.arange(1,self.n_class+1): 
            mask_temp = (mask==i)
            if(np.sum(mask_temp)>0):
                mask_gt[mask_temp,n_inst]=1
                class_ids[n_inst]=i
                n_inst+=1
        '''
        mask = mask-1 #-1: background, 0~N instance%n_class= class_id (from 0~ to n-1), 
        for i in np.arange(0,np.max(mask)+1): 
            mask_temp = (mask==i)
            if(np.sum(mask_temp)>0):
                mask_gt[mask_temp,n_inst]=1
                class_ids[n_inst]=(i%self.n_class)+1
                n_inst+=1

        mask_gt = mask_gt[:,:,:n_inst]
        class_ids = class_ids[:n_inst]
        return mask_gt.astype(np.bool), class_ids.astype(np.int32)

