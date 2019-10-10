import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from keras.layers import Input
from keras.models import Model

sys.path.append(".")  # To find local version of the library

from pix2pose_model import ae_model as ae

#find the last weight in each folder and convert it to the inference weights
if(len(sys.argv)!=2 and len(sys.argv)!=3):
    print("python3 tools/4_convert_weights_inference.py <weight_dir> < 1-(optional)to overwrite>")
weight_dir = sys.argv[1]
pass_exists=True
if len(sys.argv)>2:
    if(sys.argv[2]=='1' or int(sys.argv[2])==1 ):
        pass_exists=False
        
for root,dir,files in os.walk(weight_dir):
    if len(files)>0:
        weight_fn=""
        recent_epoch=0
        for fn_temp in files:
            if(fn_temp.startswith("pix2pose"+".") and fn_temp.endswith("hdf5")):
                    temp_split  = fn_temp.split(".")
                    epoch_split = temp_split[1].split("-")
                    epoch_split2= epoch_split[0].split("_")
                    epoch_temp = int(epoch_split2[0])
                    if(epoch_temp>recent_epoch):
                        recent_epoch = epoch_temp
                        weight_fn = fn_temp            
        if os.path.exists(os.path.join(root,"inference.hdf5")) and pass_exists:
            print("A converted file exists in ",os.path.join(root,"inference.hdf5"))
            continue
        if(weight_fn!=""):
            generator_train = ae.aemodel_unet_prob(p=1.0)
            discriminator = ae.DCGAN_discriminator()
            imsize=128
            dcgan_input = Input(shape=(imsize, imsize, 3))
            dcgan_target = Input(shape=(imsize, imsize, 3))
            prob_gt = Input(shape=(imsize, imsize, 1))
            gen_img,prob = generator_train(dcgan_input)
            recont_l =ae.transformer_loss([np.eye(3)])([gen_img,dcgan_target,prob_gt,prob_gt])
            disc_out = discriminator(gen_img)
            dcgan = Model(inputs=[dcgan_input,dcgan_target,prob_gt],outputs=[recont_l,disc_out,prob])
            print("load recent weights from ", os.path.join(root,weight_fn))
            dcgan.load_weights(os.path.join(root,weight_fn))

            print("save recent weights to ", os.path.join(root,"inference.hdf5"))
            generator_train.save_weights(os.path.join(root,"inference.hdf5"))
