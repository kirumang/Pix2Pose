
import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#Please specify the mask_rcnn directory
MASKRCNN_DIR="/home/kiru/common_ws/Mask_RCNN_Mod"
sys.path.append(MASKRCNN_DIR)
sys.path.append(".")
sys.path.append("./bop_toolkit")
import numpy as np
from bop_toolkit_lib import inout
from tools import bop_io

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from imgaug import augmenters as iaa
from tools.mask_rcnn_util import BopDetectConfig,BopDataset
import skimage

if(len(sys.argv)!=3):
    print("python3 tools/1_2_train_maskrcnn.py [cfg_fn] [dataset]")
cfg_fn = sys.argv[1] #"cfg/cfg_bop2019.json"
cfg = inout.load_json(cfg_fn)
dataset=sys.argv[2]


bop_dir,_,_,_,model_ids,_,_,_,_,cam_param_global = bop_io.get_dataset(cfg,dataset,train=True)
im_width,im_height = cam_param_global['im_size']

MODEL_DIR = os.path.join(bop_dir, "weight_detection")
config = BopDetectConfig(dataset=dataset,
                        num_classes=model_ids.shape[0]+1,#1+len(model_plys),
                        im_width=im_width,im_height=im_height)

config.display()
dataset_train = BopDataset()
dataset_train.set_dataset(dataset,model_ids,
                          os.path.join(bop_dir,"train_detect"))
dataset_train.load_dataset()
dataset_train.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

init_with = "coco"
if(os.path.exists(MODEL_DIR)): 
    for dir,root,files in os.walk(MODEL_DIR):
        for file in files:
            if(file.endswith('.h5')):
                init_with="last"
                break

if init_with == "coco":
    COCO_MODEL_PATH = os.path.join(MASKRCNN_DIR, "mask_rcnn_coco.h5")
    if not(os.path.exists(COCO_MODEL_PATH)):
        print("please download and move pretrained weights, mask_rcnn_coco.h5, in ",MASKRCNN_DIR )
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    last_path = model.find_last()
    #Load the last model you trained and continue training
    model.load_weights(last_path, by_name=True)

model.train(dataset_train, dataset_train,
            learning_rate=config.LEARNING_RATE,
            epochs=5, layers="5+" )
