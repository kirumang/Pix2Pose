# Pix2Pose
Original implementation of the paper, Kiru Park, Timothy Patten and Markus Vincze, "Pix2Pose: Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation", ICCV 2019, https://arxiv.org/abs/1908.07433

### Requirements:
* Tested environment: Ubuntu 16.04 (64bit)
* Python 3.5
* Tensorflow > 1.8
* Keras > 2.2.0
* CUDA 9.0
* See python requirements in requirements.txt

### For detection pipelines,
* Keras implementation of [Mask-RCNN](https://github.com/matterport/Mask_RCNN): used for LineMOD in the paper and all datasets in the BOP Challenge, 
```
git clone https://github.com/matterport/Mask_RCNN.git
```

* Keras implementation of [Retinanet](https://github.com/fizyr/keras-retinanet.git): used for evaluation of the T-Less dataset in the paper
```
git clone https://github.com/fizyr/keras-retinanet.git
```


- [ ] Addtional requirements have to be checked in a new docker environment


---
### Citation
If you use this code, please cite the following

```
@InProceedings{Park_2019_ICCV,
author = {Park, Kiru and Patten, Timothy and Vincze, Markus},
title = {Pix2Pose: Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```

---

### Run the recognition for BOP datasets
The original codes are updated to support the format of the most recent 6D pose benchmark, [BOP: Benchmark for 6D Object Pose Estimation](https://bop.felk.cvut.cz/home/)

1. Download a dataset from the BOP website and extract files in a folder
   - e.g.) <path_to_dataset>/<dataset_name>
   - For the recognition, "Base archive", "Object models", and "Test images" have to be downloaded at least.
2. Download and extract weights in the same dataset folder used in 1.
3. Make sure the directories follows the structure below.
     - <path_to_dataset>/<dataset_name>/models or model_eval or model_recont..: model directory that contains .ply files of models
     - <path_to_dataset>/<dataset_name>/models_xyz: norm_factor.json and .ply files of colorized 3d models
     - <path_to_dataset>/<dataset_name>/weight_detection: weight files for the detection
     - <path_to_dataset>/<dataset_name>/pix2pose_weights/<obj_name>/inference.hdf5 : weight files for each objects
3. Set config file
   1. Set directories properly based on your environment
   2. For the bop challenge dataset: <path_to_src>/cfg/cfg_bop2019.json      
   3. Use trained weights for the paper: <path_to_src>/cfg/cfg_<dataset_name>_paper.json (e.g., cfg_linemod_paper.json)
   4. score_type: 1-scores from a 2D detetion pipeline is used (used for the paper), 2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, used for the BOP challenge)
   5. task_type : 1 - SiSo task (LineMOD, T-Less in the paper, 2017 BOP Challenge), 2 - ViVo task (2019 BOP challenge format)  
   6. cand_factor: a factor for the number of detection candidates 
4. Execute the script
```
python3 5_evaluation_bop_basic.py <gpu_id> <path_cfg_json> <dataset_name>
```

5. The output will be stored in the 'path_to_output' in csv format, which can be used to calculate metric using [bop_toolkit](https://github.com/thodan/bop_toolkit).

**Important Note** Differ from the paper, we used multiple outlier thresholds in the second stage for the BOP challenge since it is not allowed to have different parameters for each object or each dataset. This can be done easily by set the "outlier_th" in a 1D-array (refer to cfg_bop2019.json). In this setup, the best result, which has the largest inlier points, will be derived during estimation after applying all values in the second stage. To reproduce the results in the paper with fixed outlier threshold values, a 2D-array should be given as in "cfg_linemode_paper.json" or "cfg_tless_paper.json")


#### ROS interface (tested with ROS-Kinetic)
- Install ros_numpy: ```pip3 install ros_numpy```
- To Run the ROS interface with our Python 3.5 code (since ROS-Kinectic uses python 2.7), we need a trick to run ROS node.
```
export PYTHONPATH=/usr/local/lib/python3.5/dist-packages:$PYTHONPATH(including other ROS related pathes)
```
- Thus, libraries will be loaded from python3.5 path, while loading ros related packages (rospy) from ros library directories in python 2.7.
- You have to specify the topic for RGB images + camera instrinsics in "ros_config.json" file. For more detail, please check out [ros_api_manual](http://github/kirumang/Pix2Pose/ros_kinetic/ros_api.manual.md)
- 
- [WIP] Depth ICP when depth image topic is available. 
- Current ros_config.json is to detect and estimate pose of YCB-Video objects. download trained weights of YCB-V dataset to run this example. 
---

### Training for a new dataset

We assume the dataset is organized in the BOP 2019 format.
For a new dataset (not in the BOP), modify bop_io.py properly to provide proper directories for training.

#### 1. Convert 3D models to colored coodinate models        
```
python 2_1_ply_file_to_3d_coord_model <dataset_name>
```
The file converts 3D models and save them to the target folder with a dimension information in a file, "norm_factor.json".

#### 2. Render and generate training pairs
```
python 2_2_render_pix2pose_training.py <dataset_name>
```

#### 3. Train pix2pose network for each object
```
python 3_train_pix2pose.py <dataset_name> <obj_name> [background_img_folder]
```


#### 4. Convert the last wegiht file to an inference file.
```
python 4_convert_weights_inference.py <pix2pose_weights folder>
```
This program looks for the last weight file in each directory  

#### 5. [Optional] Training of 2D detection pipelines (if required, skip this when you have your own 2D detection pipeline)

##### (1) Generation of images for 2D detection training        
```
python 1_1_scene_gen_for_detection.py <dataset_name> <mask=1(true)/0(false)>
```
Output files
- a number of augmented images using crops of objects in training images
- For Mask-RCNN: /mask/*.npy files
- For Retinanet(Keras-retinanet): gt.csv / label.csv
- Generated images will be saved in "<path_to_dataset>/<dataset_name>/train_detect/"

##### (2) Train Mask-RCNN or Keras-Retinanet
To train Mask-RCNN, the pre-trained weight for the MS-COCO dataset should be place in <path/to/Mask-RCNN>/mask_rcnn_coco.h5. 
```
python 1_2_train_maskrcnn.py <dataset_name>
```
or
Train Keras-retinanet using the script in the repository. It is highly recommended to initialize the network using the weights trained for the MS-COCO dataset. [link](https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5)

```
keras_retinanet/bin/train.py csv <path_to_dataset>/gt.csv <path_to_dataset>/label.csv --freeze-backbone --weights resnet50_coco_best_v2.1.0.h5
```
After training, the weights should be converted into inference model by,
```
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5
```
---

### Disclaimers:
* The paper should be cosidered the main reference for this work. All the details of the algorithm and the training are reported there.
* The rendering codes in '/rendering' are modified from the code in https://github.com/wadimkehl/ssd-6d.
* Please check out original repositories for more details of 2D detection pipelines, parameters and setups, for traininig Mask-RCNN(https://github.com/matterport/Mask_RCNN) or Keras-Retinanet(https://github.com/fizyr/keras-retinanet)

---

### Download pre-trained weights 
* Please refer to the paper for other details regarding the training
    
  * T-Less: 2D Retinanet weights + Pix2Pose weights [link](https://drive.google.com/open?id=1XjGpniXgoxzGWxq4sul1FvszUoLROkke) 
    * Given real training images are used for training (primesense)
    * reconstructed models are used to calculate VSD scores.
---

### Download: trained weights for the BOP challenge 2019
**The weights will be upload after the final submission of the challenge results.**
For the BOP challenge, we used Mask-RCNN to measure a score values for the current estimations using ovelapped ratios between masks from Mask-RCNN and the Pix2Pose estimation. All the hyperparameters, including augmentation, are set to the same for all datasets during the training and test. (33K iterations using 50 images in a mini batch)

These trained weights here are used to submit the results of core datasets in [the BOP Challenge 2019](https://bop.felk.cvut.cz/challenges).
Due to the large number of objects for training, the number of iterations are reduced (200 epoch --> 100 epoch). 

Download the zip files and extract them to the bop dataset folder
e.g., for hb, the extracted files should placed in
- [path to bop dataset]/hb/weight_detection/hb20190927T0827/mask_rcnn_hb_0005.h5
- [path to bop dataset]/hb/pix2pose_weights/[obj_no]

* LMO: 2D Mask R-CNN Detection + Pix2Pose weights
 
  *[Note] Networks are trained using synthetic images to follow the rule of the challenge, which is a totally different condition that we assumed in our paper. In the paper, we used only real images in the LineMOD dataset for trainng.

* T-Less: 2D Mask R-CNN Detection + Pix2Pose weights 
* ITODD: 2D Mask R-CNN Detection + Pix2Pose weight
* HB: 2D Mask R-CNN Detection + Pix2Pose weights
* YCBV: 2D Mask R-CNN Detection + Pix2Pose weights
* ICBIN: 2D Mask R-CNN Detection + Pix2Pose weights
* TUDL: 2D Mask R-CNN Detection + Pix2Pose weights


### Contributors:
* Kiru Park - email: park@acin.tuwien.ac.at / kirumang@gmail.com
