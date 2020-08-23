# Pix2Pose
Original implementation of the paper, Kiru Park, Timothy Patten and Markus Vincze, "Pix2Pose: Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation", ICCV 2019, https://arxiv.org/abs/1908.07433

# Notice
Codes that have been used to produce results for the BOP challenge 2020 are updated.
Thanks to PBR training images provided by the challenge, the results of LM-O, HB, and ITODD are significantly improved. 

The modifications from the original implementation of the paper are follows,

1) Replaced the encoder part with the first three blocks of Resnet-50 with pre-trained weights using ImageNet.

2) Increased a threshold for inlier pixels during PnP-Ransac operation (3 -> 5).

3) Detection results from Mask-RCNN are reused if predictions for each detection are not successful. In this case, Pix2Pose is performed for other objects that do not have good results yet.

6) A minor bug that causes bad detection results for the T-Less dataset is fixed. (different image resolutions were used during training and inference)

7) Increased the number of RPN proposals and NMS thresholds in Mask-RCNN (1000/0.7 to 2000/0.9), which produces more detection proposals

*(w/ICP)*

1) Parameters for the ICP refinement are optimized.

2) Adjusted inlier and outlier thresholds for Pix2Pose (inlier: 0.15 -> 0.2, outlier: [0.15,0.25,0.35] -> [0.2,0.3,0.35]).
   
3) A score of each hypothesis is computed by a new form, max(0,0.2-[depth_difference per pixel])/0.2, instead of counting the number of pixels that have less than 0.2 depth differences.


The official results are:

| BOP Score'20                           | AVG   | LM-O  | T-Less | TUD-L | IC-BIN | ITODD | HB    | YCB-V |
|----------------------------------------|-------|-------|--------|-------|--------|-------|-------|-------|
| Pix2Pose(RGB + Depth ICP)              | 0.591 | 0.588 | 0.512  | 0.820 | 0.390  | 0.351 | 0.695 | 0.780 |
| Pix2Pose(RGB only)                     | 0.342 | 0.363 | 0.344  | 0.420 | 0.226  | 0.134 | 0.446 | 0.457 |
| Vidal-Sensors18 (the best in '18,'19)  | 0.569 | 0.582 | 0.538  | 0.876 | 0.393  | 0.435 | 0.706 | 0.450 |
| CosyPose-ICP (ECCV'20, the best in '20) | 0.698 | 0.714 | 0.701  | 0.939 | 0.647  | 0.313 | 0.712 | 0.861 |


PBR Training images are used for LM-O, IC-BIN, ITODD, HB without additional images, and real training images are used for T-Less, TUD-L, YCB-V. To reproduce the same results, cfg/cfg_bop_2020.json or cfg/cfg_bop_2020_rgb.json (for RGB only results) has to be used with our up-to-date code.


### Requirements:
* Tested environment: Ubuntu 16.04 (64bit)
* Python > 3.5
* Tensorflow > 1.8
* Keras > 2.2.0
* CUDA > 9.0
* Bop_toolkit (https://github.com/thodan/bop_toolkit)  
* (optional:for faster icp refinement) pycuda 
* See python requirements in requirements.txt
* (optional) Docker + Nvidia-docker (https://github.com/NVIDIA/nvidia-docker)


### For detection pipelines,
* Keras implementation of [Mask-RCNN](https://github.com/matterport/Mask_RCNN): used for LineMOD in the paper and all datasets in the BOP Challenge, 
```
git clone https://github.com/matterport/Mask_RCNN.git
```

* Keras implementation of [Retinanet](https://github.com/fizyr/keras-retinanet.git): used for evaluation of the T-Less dataset in the paper
```
git clone https://github.com/fizyr/keras-retinanet.git
```


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
4. Set config file
   1. Set directories properly based on your environment
   2. For the bop challenge dataset: <path_to_src>/cfg/cfg_bop2019.json      
   3. Use trained weights for the paper: <path_to_src>/cfg/cfg_<dataset_name>_paper.json (e.g., cfg_tless_paper.json)
   4. score_type: 1-scores from a 2D detetion pipeline is used (used for the paper), 2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, used for the BOP challenge)
   5. task_type : 1 - SiSo task (2017 BOP Challenge), 2 - ViVo task (2019 BOP challenge format)  
   6. cand_factor: a factor for the number of detection candidates 
5. Execute the script
```
python3 tools/5_evaluation_bop_basic.py <gpu_id> <cfg_path> <dataset_name>
```

to run with the 3D-ICP refinement, 
```
python3 tools/5_evaluation_bop_icp3d.py <gpu_id> <path_cfg_json> <dataset_name>
```


5. The output will be stored in the 'path_to_output' in csv format, which can be used to calculate metric using [bop_toolkit](https://github.com/thodan/bop_toolkit).

**Important Note** Differ from the paper, we used multiple outlier thresholds in the second stage for the BOP challenge since it is not allowed to have different parameters for each object or each dataset. This can be done easily by set the "outlier_th" in a 1D-array (refer to cfg_bop2019.json). In this setup, the best result, which has the largest inlier points, will be derived during estimation after applying all values in the second stage. To reproduce the results in the paper with fixed outlier threshold values, a 2D-array should be given as in "cfg_tless_paper.json")

#### (Optional) Environment setup using Docker
1. Build Dockerfile ```docker build -t <container_name> .```
2. Start the container with
 ```
 nvidia-docker run -it -v <dasetdir>:/bop -v <detection_repo>:<detection_dir> -v <other_dir>:<other_dir> <container_name> bash
 ```


#### ROS interface (tested with ROS-Kinetic)
- Install ros_numpy: ```pip3 install ros_numpy```
- To Run the ROS interface with our Python 3.5 code (since ROS-Kinectic uses python 2.7), we need a trick to run ROS node.
For example,
```
export PYTHONPATH=/usr/local/lib/python3.5/dist-packages:$PYTHONPATH(including other ROS related pathes)
```
- The first path can be replaced with the dist-packages folder in the virtual environment. Thus, libraries will be loaded from python3.5 path, while loading ros related packages (rospy) from ros library directories in python 2.7.
- You have to specify the topic for RGB images + camera instrinsics in "ros_config.json" file. For more detail, please check out [ros_api_manual](ros_kinetic/ros_api.manual.md)
- ICP refinement when the depth image topic is available.
- Current ros_config.json is to detect and estimate pose of YCB-Video objects. Download trained weights of YCB-V dataset to run this example. 
---

### Training for a new dataset

We assume the dataset is organized in the BOP 2019 format.
For a new dataset (not in the BOP), modify [bop_io.py](tools/bop_io.py) properly to provide proper directories for training. Theses training codes are used to prepare and train the network for the BOP 2019.

#### 1. Convert 3D models to colored coodinate models        
```
python3 tools/2_1_ply_file_to_3d_coord_model <cfg_path> <dataset_name>
```
The file converts 3D models and save them to the target folder with a dimension information in a file, "norm_factor.json".

#### 2. Render and generate training pairs
```
python3 tools/2_2_render_pix2pose_training.py <cfg_path> <dataset_name>
```

#### 3. Train pix2pose network for each object
```
python3 tools/3_train_pix2pose.py <cfg_path> <dataset_name> <obj_name> [background_img_folder]
```


#### 4. Convert the last wegiht file to an inference file.
```
python3 tools/4_convert_weights_inference.py <pix2pose_weights folder>
```
This program looks for the last weight file in each directory  

#### 5. [Optional] Training of 2D detection pipelines (if required, skip this when you have your own 2D detection pipeline)

##### (1) Generation of images for 2D detection training        
```
python3 tools/1_1_scene_gen_for_detection.py <cfg_path> <dataset_name> <mask=1(true)/0(false)>
```
Output files
- a number of augmented images using crops of objects in training images
- For Mask-RCNN: /mask/*.npy files
- For Retinanet(Keras-retinanet): gt.csv / label.csv
- Generated images will be saved in "<path_to_dataset>/<dataset_name>/train_detect/"

##### (2) Train Mask-RCNN or Keras-Retinanet
To train Mask-RCNN, the pre-trained weight for the MS-COCO dataset should be place in <path/to/Mask-RCNN>/mask_rcnn_coco.h5.
```
python3 tools/1_2_train_maskrcnn.py <cfg_path> <dataset_name>
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

### Download trained weights 
* Please refer to the paper for other details regarding the training
    
  * T-Less: 2D Retinanet weights + Pix2Pose weights [link](https://drive.google.com/open?id=1XjGpniXgoxzGWxq4sul1FvszUoLROkke) 
    * Given real training images are used for training (primesense)
    * reconstructed models are used to calculate VSD scores.
    * To test using all test images, download and copy [all_target_tless.json](https://drive.google.com/open?id=1O6dKfWoe0ERlXm6Gg_3XvdxWo4a2BWVY) file into the dataset folder (together with the test_targets_bop19.json file) 
---

### Download: trained weights for the BOP challenge 2020
These trained weights here are used to submit the results of core datasets in [the BOP Challenge 2020](https://bop.felk.cvut.cz/challenges). 

First of all, norm_factors have to be downloded and placed in the following path:
```[path/to/bop_dataset]/[dataset_name]/models_xyz/norm_factor.json```

Download link: [Norm_factor files for all 7 dataset](https://drive.google.com/file/d/1Epilq368c50H4OJLiy0hsZwypCyUbN5y/view?usp=sharing)


Download the zip files and extract them to the bop dataset folder
e.g., for TLess, the extracted files should be placed in
- [path to bop dataset]/tless/weight_detection/tless20190927T0827/mask_rcnn_tless_0005.h5
- [path to bop dataset]/tless/pix2pose_weights/[obj_no]

* LM-O: [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/1Xo_vKbjCW4eHtmbMYoCQYyRqFAc_RHsL/view?usp=sharing)

* T-Less: [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/1_uf40NIsmlje6bZLpRHR3Qk7reCk0PUx/view?usp=sharing)

* TUDL: [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/1EOthiwIZGH9G0bL4NbRhhimODiaFWPdc/view?usp=sharing)

* ICBIN: [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/14ylgR0SlHXXkfi8mbm68snOH-DKvLkMV/view?usp=sharing)

* ITODD: [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/1W7cDJZX3QMaD3dIQ_RnDNsOb7AytBbPr/view?usp=sharing)

* HB: [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/1sOEpkV202YoXiB82SMYSPVKwbwUp03RI/view?usp=sharing)

* YCBV : [2D Mask R-CNN Detection + Pix2Pose weights](https://drive.google.com/file/d/1au-jcTNQVZNV8aEpuTsMb68IZoVruEN8/view?usp=sharing)



### Contributors:
* Kiru Park - email: park@acin.tuwien.ac.at / kirumang@gmail.com
