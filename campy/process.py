import sys, time, logging, warnings
import numpy as np
import tensorflow as tf
import os, csv
import tensorrt 
from typing import Tuple, Optional

from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
from typing import List, Optional, Text

from campy.teensy import Teensy

import traceback

precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}

def find_global_peaks_rough(
    cms: tf.Tensor, threshold: float = 0.1
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Adapted from SLEAP
    
    Find the global maximum for each sample and channel.

    Args:
        cms: Tensor of shape (samples, height, width, channels).
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will be replaced with NaNs.

    Returns:
        A tuple of (peak_points, peak_vals).

        peak_points: float32 tensor of shape (samples, channels, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (samples, channels) containing the values at
        the peak points.
    """
    # Find row maxima.
    max_img_rows = tf.reduce_max(cms, axis=2)
    argmax_rows = tf.reshape(tf.argmax(max_img_rows, axis=1), [-1])

    # Find col maxima.
    max_img_cols = tf.reduce_max(cms, axis=1)
    argmax_cols = tf.reshape(tf.argmax(max_img_cols, axis=1), [-1])

    # Construct sample and channel subscripts.
    channels = tf.cast(tf.shape(cms)[-1], tf.int64)
    total_peaks = tf.cast(tf.shape(argmax_cols)[0], tf.int64)
    sample_subs = tf.range(total_peaks, dtype=tf.int64) // channels
    channel_subs = tf.math.mod(tf.range(total_peaks, dtype=tf.int64), channels)

    # Gather subscripts.
    peak_subs = tf.stack([sample_subs, argmax_rows, argmax_cols, channel_subs], axis=1)

    # Gather values at global maxima.
    peak_vals = tf.gather_nd(cms, peak_subs)

    # Convert to points form (samples, channels, 2).
    peak_points = tf.reshape(
        tf.cast(tf.stack([argmax_cols, argmax_rows], axis=-1), tf.float32),
        [-1, channels, 2],
    )
    peak_vals = tf.reshape(peak_vals, [-1, channels])

    # Mask out low confidence points.
    peak_points = tf.where(
        tf.expand_dims(peak_vals, axis=-1) < threshold,
        x=tf.constant(np.nan, dtype=tf.float32),
        y=peak_points,
    )

    return peak_points, peak_vals

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


def ProcessData():
    processdata = {}
    processdata["timeStamp"] = []
    processdata["frameNumber"] = []
    processdata["frameProcessTime"] = []
    processdata["LoadOnQueue"] = []
    
    return processdata

def SaveMetadata(vid_folder, processdata):

    csv_file = os.path.join(vid_folder, 'process_times.csv')

    keys_to_write = ['frameNumber', 'timeStamp', 'frameProcessTime', 'LoadOnQueue']
    length = len(processdata[keys_to_write[0]])

    with open(csv_file, 'w', newline='') as f:
        w = csv.writer(f)

        w.writerow(keys_to_write)
        for i in range(length):
            row = [processdata[key][i] for key in keys_to_write]

            w.writerow(row)

def ProcessFrames(process_params, ProcessQueues, BehaviorQueue, startQueues, stop_event):

    disable_preallocation()

    print('GPU initialized')

    vid_folder = process_params['video_folder']
    n_cams = process_params['n_cams']
    model_path = process_params['model_path']
    n_keypoints = process_params['num_keypoints']
    img_shape = process_params['img_shape']
    ser_port = process_params['serial_port']

    model = OptimizedModel(model_path)

    trigger_teensy = Teensy(ser_port)

    print('Model loaded')

    with tf.device('/GPU:0'):
        gpu_tensor = tf.Variable(initial_value=tf.zeros(img_shape, dtype=tf.float32), trainable=False)

    model.predict(gpu_tensor) #initialize graph

    print('Model loaded and initialized')

    cam_frames = [False]*n_cams
    cam_frame_numbers = [0,0,0]

    #behavior = BehaviorBuffer(buffer_size, n_keypoints, 3, template)

    processdata = ProcessData()

    #for sq in startQueues:
    #    sq.put("start")

    #trigger_teensy.send_single_trigger()

    #time.sleep(1)

    trigger_teensy.send_single_trigger()

    first_run_done = False
    framenumber = 0

    while not stop_event.is_set():
        try: 
            #TODO: parallelize reads from camera acquisition queue
            #TODO: Handle cases of dropped frames
            for cam in np.arange(n_cams):
                if cam_frames[cam] is False:
                    if not ProcessQueues[cam].empty():
                        try:
                            with tf.device('/GPU:0'):
                                data = tf.image.convert_image_dtype(ProcessQueues[cam].get(), tf.float32)
                                gpu_tensor[cam].assign(data)
                                #print('Image taken from queue')
                            cam_frames[cam] = True
                            cam_frame_numbers[cam] +=1
                        except Exception as e:
                            traceback.print_exc()
            
            if all(element for element in cam_frames):
                pre_predict = time.perf_counter()
                output = model.predict(gpu_tensor)

                peaks, peak_vals = find_global_peaks_rough(tf.transpose(output, perm=[0,2,3,1])) #peaks: 3x23x2

                peaks_numpy = peaks.numpy()
                peak_vals = peak_vals.numpy()

                peaks_numpy = peaks_numpy * 4 # Change based on input scaling and output stride
                peaks_numpy = (peaks_numpy / 0.5) + 0.5

                processed = time.perf_counter()

                BehaviorQueue.put((peaks_numpy, peak_vals))

                timeStamp = time.perf_counter()

                if not first_run_done:
                    framenumber = 0
                    first_run_done = True
                    time.sleep(0.05)
                    trigger_teensy.send_start_signal()

                else:
                    framenumber +=1

                    processdata['frameNumber'].append(framenumber)
                    processdata['timeStamp'].append(pre_predict)
                    processdata['frameProcessTime'].append(processed-pre_predict)
                    processdata['LoadOnQueue'].append(timeStamp-processed)

                cam_frames = [False]*n_cams

                if framenumber % 100 == 0:
                    print(f'Processed frame {framenumber}')

            time.sleep(0.001)

        except Exception as e:
            traceback.print_exc()
            break
            
        except KeyboardInterrupt:
            print(f"Processor Interrupted by user")
            stop_event.set()
            break

    print(output.shape)

    trigger_teensy.send_stop_signal()
    SaveMetadata(vid_folder, processdata)

        
