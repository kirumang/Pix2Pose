import numpy as np
import cv2
import random
import transforms3d as tf3d
import math
import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


class cuda_renderer():
    def __init__(self,fx=572.4114,fy=573.57043,cx=325.26110,cy=242.04899,res_y=480,res_x=640,n_block=256):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.res_y = res_y
        self.res_x = res_x
        self.n_block = n_block
        
        
        self.mod = SourceModule \
            (        """

        #include<stdio.h>
        #define RES_X (""" + str(self.res_x) + """)
        #define RES_Y (""" + str(self.res_y) + """)

        __global__ void get_render(float *v1, float *v2, float *v3,
                                   float *u1, float *u2, float *u3,
                                   float *z1, float *z2, float* z3,
                                   float *d_buf,float* max_idx,float* mask,float* bbox)
        {
                unsigned int idx = threadIdx.x+(blockIdx.x*blockDim.x);
                if(max_idx[idx]==1)
                {
                int max_v,max_u,min_v,min_u;
                int vv1 = v1[idx];
                int vv2 = v2[idx];
                int vv3 = v3[idx];
                int uu1 = u1[idx];
                int uu2 = u2[idx];
                int uu3 = u3[idx];
                float zz1 = z1[idx];
                float zz2 = z2[idx];
                float zz3 = z3[idx];

                min_v = min(min(vv1,vv2),vv3);
                min_u = min(min(uu1,uu2),uu3);
                max_v = max(max(vv1,vv2),vv3);
                max_u = max(max(uu1,uu2),uu3);

                for(int v=min_v;v<=max_v;v++)
                {
                    for(int u=min_u;u<=max_u;u++)
                    {
                             if(v>=0 && v<RES_Y && u>=0 && u<RES_X)
                                {

                                    int ii = v*RES_X+u;
                                    float denominator = ((vv2 - vv3)*(uu1 - uu3) + (uu3 - uu2)*(vv1 - vv3));
                                    float a = ((vv2 - vv3)*(u - uu3) + (uu3 - uu2)*(v - vv3)) / denominator;
                                    float b = ((vv3 - vv1)*(u - uu3) + (uu1 - uu3)*(v - vv3)) / denominator;
                                    float c = 1 - a - b;
                                    float z = (a*zz1 + b*zz2 + c*zz3)/(a+b+c);

                                    if (0 <= a && a <= 1 && 0 <= b && b <= 1 && 0 <= c && c <= 1)
                                        {
                                         atomicMin( (unsigned int*) &d_buf[ii], __float_as_int(z));
                                         atomicMax( (unsigned int*) &mask[ii], 1);
                                         atomicMin( (unsigned int*)  &bbox[0], __float_as_int( (float)v));
                                         atomicMin( (unsigned int*)  &bbox[1], __float_as_int( (float)u));
                                         atomicMax( (unsigned int*)  &bbox[2], __float_as_int( (float)v));
                                         atomicMax( (unsigned int*)  &bbox[3], __float_as_int( (float)u));
                                        }
                                    //atomicMin( (unsigned int*) &d_buf[ii],1);
                                }
                    }
                }

                }



        }

        

        """
            )
        self.rendering = self.mod.get_function("get_render")
       

    def cuda_render(self,pts,face_set):
        pts = pts.astype(np.float32)
        v = ((np.round(self.fy*pts[:,1]/pts[:,2]+self.cy)).astype(np.int)).astype(np.float32)
        u = ((np.round(self.fx*pts[:,0]/pts[:,2]+self.cx)).astype(np.int)).astype(np.float32)
        depth_b = gpuarray.zeros((self.res_y*self.res_x), dtype=np.float32)+100#+90000
        depth_mask = np.zeros((self.res_y*self.res_x),dtype=np.float32)
        bbox = gpuarray.zeros((4),dtype=np.float32)
        bbox[0:2]=np.array([9999,9999],dtype=np.float32)


        max_idx = np.ones((face_set.shape[0]), dtype=np.float32)
        grid_n= int((face_set.shape[0]/self.n_block))+1
        self.rendering(drv.In(v[face_set[:,0]]), drv.In(v[face_set[:,1]]),drv.In(v[face_set[:,2]]),
                          drv.In(u[face_set[:,0]]), drv.In(u[face_set[:,1]]),drv.In(u[face_set[:,2]]),
                          drv.In(pts[face_set[:,0],2]), drv.In(pts[face_set[:,1],2]),drv.In(pts[face_set[:,2],2]),
                          depth_b,drv.In(max_idx), drv.Out(depth_mask),bbox,
                          block=(self.n_block, 1, 1), grid=(grid_n, 1, 1))
        img = depth_b.get()
        img[img==100]=0
        img= np.reshape(img,(self.res_y,self.res_x))
        mask = np.reshape(depth_mask,(self.res_y,self.res_x)).astype(bool)
        bbox_final = bbox.get()
        return img,mask,bbox_final.astype(np.int)

    
    def render_obj(self,ply_model,R=np.eye(3),t=np.array([0,0,1])):
        pts = (np.matmul(R,ply_model['pts'].T) + np.expand_dims(t,axis=1)).T #Nx3
        face_set = ply_model['faces'].astype(np.int)
        return self.cuda_render(pts,face_set)

    def set_cam(self,cam_K):
        self.fx = cam_K[0,0]
        self.fy = cam_K[1,1]
        self.cx = cam_K[0,2]
        self.cy = cam_K[1,2]

