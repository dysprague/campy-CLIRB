
import numpy as np
import scipy.io as sio

from numpy.lib.format import open_memmap
import cv2

from time import perf_counter
import traceback

from collections import deque


import time
from campy.teensy import Teensy

import serial
import os, csv

class DataLogger:
    def __init__(self, filename, batch_size=500):
        self.filename   = filename
        self.batch_size = batch_size
        self.file       = open(filename, 'w', newline='', buffering=1)
        self.buffer     = []
        self.file.write(
            "frame,queue_time,head_height,triggered,trigger_time\n"
        )

    def log(self, frame, t_queue, head_height, triggered, t_trigger=None):
        """
        frame        : int
        t_queue      : float, when we got data from the queue
        head_height  : float
        triggered    : bool (0/1)
        t_trigger    : float or '' (timestamp of reward send)
        """
        trig_time = f"{t_trigger:.6f}" if triggered else ""
        line = (
            f"{frame},{t_queue:.6f},{head_height:.3f},{int(triggered)},{trig_time}\n"
        )
        self.buffer.append(line)
        if len(self.buffer) >= self.batch_size:
            self.file.write(''.join(self.buffer))
            self.file.flush()
            os.fsync(self.file.fileno())
            self.buffer.clear()

    def close(self):
        if self.buffer:
            self.file.write(''.join(self.buffer))
            self.file.flush()
            os.fsync(self.file.fileno())
            self.buffer.clear()
        self.file.close()

    def __del__(self):
        self.close()



class BehaviorRule:
    """
    Base class: keeps a rolling buffer of the last buffer_duration seconds of data,
    and enforces a refractory period between triggers.
    """
    def __init__(self, buffer_duration, refractory=0.0):
        self.buffer_duration  = buffer_duration
        self.refractory       = refractory
        self._last_trigger    = -np.inf
        self.buffer           = deque()   # holds (t, data) pairs

    def _can_trigger(self, t):
        return (t - self._last_trigger) >= self.refractory

    def update(self, data, t):
        # 1) Manage the timestamped buffer
        self.buffer.append((t, data))
        cutoff = t - self.buffer_duration
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

        # 2) Base always returns False; subclasses override this
        return False


class HeadHoldRule(BehaviorRule):
    """
    Fires once when head-height stays within [head_min, head_max]
    for at least hold_time seconds, then respects refractory.
    """
    def __init__(self, buffer_duration, head_min, head_max, hold_time, refractory=0.0):
        super().__init__(buffer_duration, refractory)
        self.head_min  = head_min
        self.head_max  = head_max
        self.hold_time = hold_time
        self._enter_t  = None

    def update(self, keypoints, t):
        super().update(keypoints, t)

        # compute current head height
        head_z = float(np.mean(keypoints[:2, 2]))

        fired = False
        if self.head_min <= head_z <= self.head_max:
            if self._enter_t is None:
                self._enter_t = t
            elif (t - self._enter_t) >= self.hold_time and self._can_trigger(t):
                fired = True
        else:
            self._enter_t = None

        if fired:
            self._last_trigger = t
        return fired


def normalize_skeleton(points_3d):

    SpineF = points_3d[3,:]  # shape: (n_frames, 3)
    SpineM = points_3d[4, :]  # shape: (n_frames, 3)

    rotangle = np.arctan2( -(SpineF[1] - SpineM[1]), (SpineF[0] - SpineM[0]) )

    global_rotmat = np.zeros((2, 2))

    global_rotmat[0, 0] = np.cos(rotangle)
    global_rotmat[0, 1] = -np.sin(rotangle)
    global_rotmat[1, 0] = np.sin(rotangle)
    global_rotmat[1, 1] = np.cos(rotangle) 

    markers_centered = points_3d - points_3d[4,:] #23x3

    markers_rotated = markers_centered 
    markers_rotated[:,:2] = np.transpose(global_rotmat @ np.transpose(markers_rotated[:,:2]))

    return markers_rotated

def undistort_points(points, K, dist_coeffs):
    """
    points: (N,2) array of pixel coordinates in one image
    K: (3,3) intrinsic matrix
    dist_coeffs: vector [k1,k2,p1,p2,(k3,…)] as OpenCV expects
    returns: (N,2) of normalized coordinates x',y' satisfying
             [x', y', 1]^T ∝ K^{-1} [u,v,1]^T after removing distortion
    """
    # OpenCV’s undistortPoints returns normalized coordinates if you omit P
    pts = points.reshape(-1,1,2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist_coeffs)  
    return undist.reshape(-1,2)

def build_projection_matrix(K, R, t):
    """Returns the 3×4 projection matrix P = K [R | t]."""
    return np.identity(3) @ np.hstack((R, t.reshape(3,1))) # Use identity for K because undistort already uses camera intrinsics

def triangulate_point_multiview(undist_points, P_list):
    """
    undist_points: list of (xi, yi) normalized coords, one per camera
    P_list:       list of corresponding 3×4 projection matrices
    returns:      X (3,) inhomogeneous 3D point
    """
    m = len(P_list)
    A = np.zeros((2*m, 4), dtype=np.float64)
    for i, ((x, y), P) in enumerate(zip(undist_points, P_list)):
        A[2*i    ] = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]

    # Solve A X = 0 via SVD:
    _, _, Vt = np.linalg.svd(A)
    X_homog = Vt[-1]        # last row of V^T
    X_homog /= X_homog[3]   # de-homogenize
    return X_homog[:3]


def triangulate(keypoints_2D, P_list, dist_coefs, K_list): #keys 3D is n_camsx23x2
    #keypoints_2D[:,:,1] = 1200 - keypoints_2D[:,:,1] # Flip y vals
        
    undist_pts = np.zeros(keypoints_2D.shape) # n_cams x n_keypoints x 2

    points_3d = np.zeros((keypoints_2D.shape[1], 3))

    try:
        for i in range(keypoints_2D.shape[0]):
            undist_pts[i,:,:] = undistort_points(keypoints_2D[i,:,:], K_list[i], np.array(dist_coefs[i]))

        for j in range(keypoints_2D.shape[1]):
            uv_list = undist_pts[:,j,:] 
            points_3d[j,:] = triangulate_point_multiview(uv_list, P_list)

        #points_3d[:,2] = 100 -points_3d[:,2] # correct for flipping

        return points_3d, undist_pts

    except Exception as e:
        #print(e)
        
        return points_3d, undist_pts

def correct_triangulations(points_3d, P_list, undist_pts, edges, bone_length_avg, w_bone=1.0):
    return points_3d #TODO: add triangulation corrections

def BehaviorData():

    behaviordata = {}

    behaviordata['frameNumber'] = []
    behaviordata['behaviorProcessTime'] = [] 
    behaviordata['finalTimeStamp'] = []

    return behaviordata

def SaveMetadata(vid_folder, behaviordata):

    csv_file = os.path.join(vid_folder, 'behavior_times.csv')

    keys_to_write = ['frameNumber', 'behaviorProcessTime', 'finalTimeStamp']
    length = len(behaviordata[keys_to_write[0]])

    with open(csv_file, 'w', newline='') as f:
        w = csv.writer(f)

        w.writerow(keys_to_write)
        for i in range(length):
            row = [behaviordata[key][i] for key in keys_to_write]

            w.writerow(row)


def ProcessBehavior(behavior_params, BehaviorQueue, stop_event):
    '''
    Args:

    Outputs:
        - Saves 3D keypoints to file
        - 
    '''
    print('Initializing behavior module')

    vid_folder = behavior_params["video_folder"]
    cam_calibration_path = behavior_params["calibration_path"]
    calibration_files = behavior_params["calibration_files"]
    skel_file = behavior_params["skeleton"]
    edge_lengths = behavior_params['edge_lengths']
    max_frames = behavior_params["numImagesToGrab"]+1
    n_cams = behavior_params['n_cams']
    save_path = behavior_params['save_path']

    skel_label = sio.loadmat(skel_file, simplify_cells=True)
    labels = skel_label['RP2']

    skeleton = skel_label['skeleton']
    # skeleton
    nodes = skeleton['joint_names']
    nodes = list(map(str, nodes))

    edges = skeleton['joints_idx']-1 # python indexing

    cam_extrinsics = []

    for file in calibration_files:
        params = sio.loadmat(f'{cam_calibration_path}/{file}', simplify_cells=True)
        cam_extrinsics.append({'K':params['K'], 'RDistort':params['RDistort'], 'TDistort':params['TDistort'], 'r':params['r'], 't':params['t']})

    P_list = []
    dist_coefs = []
    K_list = []

    for cam_vals in cam_extrinsics:
        K = np.transpose(cam_vals['K'])
        r = np.transpose(cam_vals['r'])
        t = -cam_vals['t']
        Rdist = cam_vals['RDistort']
        Tdist = cam_vals['TDistort']
        P_list.append(build_projection_matrix(K,r,t))
        K_list.append(K)
        dist_coefs.append([Rdist[0], Rdist[1], Tdist[0], Tdist[1]])


    #template_path = behavior_params["template_path"]
    #PCA_path = behavior_params["PCA_path"]

    #cam_calbration = sio.loadmat(cam_calibration_path, simplify_cells=True)
    #template = sio.loadmat(tempate_path, simplify_cells=True)
    #PCA_mat = sio.loadmat(PCA_path, simplify_cells=True)

    mm_peaks_and_vals = open_memmap(f'{save_path}/sleap_keys_2D.npy', mode='w+',
                           dtype=np.float64,
                           shape = (max_frames, n_cams, 23, 3)) # xy positions of keypoints from each camera and peak_val confidence levels
    mm_keys_3D = open_memmap(f'{save_path}/triang_keys_3D.npy', mode='w+',
                             dtype=np.float64,
                             shape = (max_frames, 23, 3))


    print("Behavior analysis module initialized and ready")

    behaviordata = BehaviorData()

    # parameters - should eventually be set in config file
    buffer_secs = 5.0
    head_min    = 115.0
    head_max    = 500.0
    hold_time   = 0.5     # must hold for 500 ms
    refractory  = 2.0     # then wait 2 s before next reward

    opcon_teensy = Teensy('/dev/ttyACM0') # add to config file
    logger = DataLogger(os.path.join(save_path, 'behavior_log.csv'))
    rule = HeadHoldRule(buffer_secs, head_min, head_max, hold_time, refractory)
    
    start_time = perf_counter()

    first_run_done = False
    frameNumber = 0

    print('Behavior initialized')

    while not stop_event.is_set():
        if not BehaviorQueue.empty():
            try:

                # GET AND STORE RAW DATA
                keypoints_2D, peak_vals  = BehaviorQueue.get() # 3x23x2, 3x23 matrices of peak locations and confidence
                keys_obtained = perf_counter()
                # Queue get is blocking if empty
                mm_peaks_and_vals[frameNumber, :,:,:2] = keypoints_2D 
                mm_peaks_and_vals[frameNumber, :,:,2] = peak_vals


                # PRE-PROCESS KEYPOINTS TO GET TRIANGULATED EGOCENTRIC POINTS
                keypoints_3D, undist_pts = triangulate(keypoints_2D, P_list, dist_coefs, K_list)
                #TODO: add triangulation correction

                mm_keys_3D[frameNumber,:,:] = keypoints_3D
                points_rotated = normalize_skeleton(keypoints_3D)
                beh_processed = perf_counter() # sorry moved this around, which step was this meant to correspond to? -kh

                points_rotated[:, 2] = 100 - points_rotated[:,2]

                if not first_run_done:
                    frameNumber=0
                    first_run_done=True
                    print('First frame fully processed')

                # REWARD (OR NOT) BASED ON BEHAVIOR
                reward = rule.update(points_rotated, keys_obtained) #TODO: get camera acquisition time from camera process
                if reward:
                    print(f'Triggered reward on frame {frameNumber}')
                    opcon_teensy.send_reward()
                    t_trigger = perf_counter()
                else:
                    t_trigger = None

                # test to trigger reward every 100 frames
                #if frameNumber%100 == 0:
                #    print(f'Triggered reward on frame {frameNumber}')
                #    opcon_teensy.send_reward()
                #    t_trigger = perf_counter()
                #else:
                #    t_trigger = None

                # LOG 
                head_height = float(np.mean(points_rotated[:2,2])) # this is clunky.. 
                logger.log(frameNumber, keys_obtained, head_height, reward, t_trigger)
                behaviordata['frameNumber'].append(frameNumber)
                behaviordata['behaviorProcessTime'].append(beh_processed-keys_obtained)
                behaviordata['finalTimeStamp'].append(beh_processed)

                # HOUSEKEEPING
                if (frameNumber%100) == 0:
                    mm_peaks_and_vals.flush() #Flush buffer of memory maps every 100 frames or so
                    mm_keys_3D.flush()

                frameNumber += 1

            except Exception as e:
                traceback.print_exc()

        else:
            time.sleep(0.005)

    SaveMetadata(vid_folder, behaviordata)

        
    

