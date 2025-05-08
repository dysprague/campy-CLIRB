
import os, sys, time, csv, logging
import numpy as np
from collections import deque
#import tensorflow as tf
from scipy import io as sio
import cv2
import traceback

import tensorflow as tf
import tensorrt 

from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
from typing import List, Optional, Text

from time import perf_counter

def read_frames(video_path, fidxs=None, grayscale=True):
    """Read frames from a video file.
    
    Args:
        video_path: Path to MP4
        fidxs: List of frame indices or None to read all frames (default: None)
        grayscale: Keep only one channel of the images (default: True)
    
    Returns:
        Loaded images in array of shape (n_frames, height, width, channels) and dtype uint8.
    """
    vr = cv2.VideoCapture(video_path)
    if fidxs is None:
        fidxs = np.arange(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fidx in fidxs:
        vr.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        img = vr.read()[1]
        if grayscale:
            img = img[:, :, [0]]
        frames.append(img)
    return np.stack(frames, axis=0)

def get_available_gpus() -> List[tf.config.PhysicalDevice]:
    """Return a list of available GPUs."""
    return tf.config.get_visible_devices("GPU")

def disable_preallocation():
    """Disable preallocation of full GPU memory on all available GPUs.

    This enables memory growth policy so that TensorFlow will not pre-allocate all
    available GPU memory.

    Preallocation can be more efficient, but can lead to CUDA startup errors when the
    memory is not available (e.g., shared, multi-session and some *nix systems).

    See also: enable_gpu_preallocation
    """
    for gpu in get_available_gpus():
        tf.config.experimental.set_memory_growth(gpu, True)

class OptimizedModel():
    def __init__(self, saved_model_dir = None):
        self.loaded_model_fn = None
        
        if not saved_model_dir is None:
            self.load_model(saved_model_dir)
            
    
    def predict(self, input_data, batch_size=None): 
        if self.loaded_model_fn is None:
            raise(Exception("Haven't loaded a model"))
            
        if batch_size is not None:
            all_inds = np.arange(len(input_data))
            all_preds = []
            for inds in np.array_split(all_inds, int(np.ceil(len(all_inds) / batch_size))):
                all_preds.append(self.predict(input_data[inds]))
            return all_preds
                
#         x = tf.constant(input_data.astype('float32'))
        x = tf.constant(input_data)
        labeling = self.loaded_model_fn(input=x)
        try:
            preds = labeling['predictions'].numpy()
        except:
            try:
                preds = labeling['probs'].numpy()
            except:
                try:
                    preds = labeling[next(iter(labeling.keys()))]
                except:
                    raise(Exception("Failed to get predictions from saved model object"))
        return preds
    
    def load_model(self, saved_model_dir):
        saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
        wrapper_fp32 = saved_model_loaded.signatures['serving_default']
        
        self.loaded_model_fn = wrapper_fp32

if __name__ == '__main__':

    model_path = '../models/250421_183045.single_instance.n=8280.trt.FP32'
    model = OptimizedModel(model_path)
    
    video_path = '../test/example.mp4'
    
    t0 = perf_counter()
    
    frame = read_frames(video_path, fidxs=[0], grayscale=False)
    
    t1 = perf_counter()
    
    with tf.device('/CPU:0'):
        imresized = tf.transpose(tf.cast(tf.image.resize(frame, size=[600,960], method='bilinear', preserve_aspect_ratio=False, antialias=False,), tf.float32), perm=[0,3,1,2])
    #ready for processing 

    print(imresized.device)
    
    t2 = perf_counter()
    
    #disable_preallocation()
    
    with tf.device('/GPU:0'):
        gpu_tensor = tf.cast(tf.Variable(initial_value=tf.zeros((3,3,600,960))), tf.float32)
    
    model.predict(gpu_tensor) #initialize graph 
    
    t3 = perf_counter()
    
    for i in range(3):
        gpu_tensor[i].assign(imresized) # check how long to load frames onto GPU

    print(gpu_tensor.device)
    
    t4 = perf_counter()
    
    output = model.predict(gpu_tensor)
    
    t5 = perf_counter()
    
    print(f'Read frame time: {(t1-t0)*1000} msec')
    print(f'CPU im preprocessing time: {(t2-t1)*1000} msec')
    print(f'Model and gpu tensor initialization time: {(t3-t2)*1000} msec')
    print(f'Place preprocessed frames on GPU time: {(t4-t3)*1000} msec')
    print(f'Model prediction time: {(t5-t4)*1000} msec')
