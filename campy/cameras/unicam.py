"""
Unicam unifies camera APIs with common syntax to simplify multi-camera acquisition and 
reduce redundancy in campy code.
"""

import os, sys, time, csv, logging
import numpy as np
from collections import deque
import tensorflow as tf
from scipy import io as sio
import cv2
import traceback
from time import perf_counter


def ImportCam(make):
	if make == "basler":
		from campy.cameras import basler as cam
	elif make == "flir":
		from campy.cameras import flir as cam
	elif make == "emu":
		from campy.cameras import emu as cam
	else:
		print('Camera make is not supported by CamPy. Check config.', flush=True)
	return cam


def LoadSystems(params):
	try:
		systems = {}
		makes = GetMakeList(params)
		for m in range(len(makes)):
			systems[makes[m]] = {}
			cam = ImportCam(makes[m])
			systems[makes[m]]["system"] = cam.LoadSystem(params)
	except Exception as e:
		logging.error('Caught exception at camera/unicam.py LoadSystems. Check cameraMake: {}'.format(e))
		raise
	return systems


def LoadDevice(systems, params, cam_params):
	try:
		cam = ImportCam(cam_params["cameraMake"])
		cam_params = cam.LoadDevice(systems, params, cam_params)
	except Exception as e:
		logging.error('Caught exception at camera/unicam.py LoadSystems. Check cameraMake: {}'.format(e))
		raise
	return cam_params


def OpenCamera(cam_params, stopWriteQueue):
	# Import the cam module
	cam = ImportCam(cam_params["cameraMake"])

	try:
		camera, cam_params = cam.OpenCamera(cam_params)

		print("Opened {}: {} {} serial# {}".format( \
			cam_params["cameraName"],
			cam_params["cameraMake"], 
			cam_params["cameraModel"],
			cam_params["cameraSerialNo"]))

	except Exception as e:
		logging.error("Caught error at cameras/unicam.py OpenCamera: {}".format(e))
		stopWriteQueue.append('STOP')

	return cam, camera, cam_params


def GetDeviceList(systems, params):
	makes = GetMakeList(params)
	for m in range(len(makes)):
		cam = ImportCam(makes[m])
		system = systems[makes[m]]["system"]
		deviceList = cam.GetDeviceList(system)
		serials = [cam.GetSerialNumber(deviceList[i]) for i in range(len(deviceList))]
		systems[makes[m]]["serials"] = serials
		systems[makes[m]]["deviceList"] = deviceList
	return systems


def GetMakeList(params):
	if type(params["cameraMake"]) is list:
		cameraMakes = [params["cameraMake"][m] for m in range(len(params["cameraMake"]))]
	elif type(params["cameraMake"]) is str:
		cameraMakes = [params["cameraMake"]]
	makes = list(set(cameraMakes))
	return makes


def GrabData(cam_params):
	grabdata = {}
	grabdata["timeStamp"] = []
	grabdata["frameNumber"] = []
	grabdata["GrabTime"] = []
	grabdata["PreprocessTime"] = []
	grabdata["LoadOnWrite"] = []
	grabdata["ProcessQueueTimeStamp"] = []
	grabdata['startTime'] = []
	grabdata["cameraName"] = cam_params["cameraName"]

	# Calculate display rate
	if cam_params["displayFrameRate"] <= 0:
		grabdata["frameRatio"] = float('inf')
	elif cam_params["displayFrameRate"] > 0 and cam_params["displayFrameRate"] <= cam_params['frameRate']:
		grabdata["frameRatio"] = int(round(cam_params["frameRate"]/cam_params["displayFrameRate"]))
	else:
		grabdata["frameRatio"] = cam_params["frameRate"]

	# Calculate number of images and chunk length
	grabdata["numImagesToGrab"] = int(round(cam_params["recTimeInSec"]*cam_params["frameRate"]))
	grabdata["chunkLengthInFrames"] = int(round(cam_params["chunkLengthInSec"]*cam_params["frameRate"]))

	return grabdata


def StartGrabbing(camera, cam_params, cam):
	grabbing = cam.StartGrabbing(camera)
	if grabbing:
		print(cam_params["cameraName"], "ready to trigger.")
	return grabbing

def convert_rgb_to_bgr(image: tf.Tensor) -> tf.Tensor: 
    """
	SLEAP function ported into campy

	Convert an RGB image to BGR format by reversing the channel order.

    Args:
        image: Tensor of any dtype with shape (..., 3) in RGB format. If grayscale, the
            image will be converted to RGB first.

    Returns:
        The input image with the channels axis reversed.
    """
    return tf.reverse(image, axis=[-1])

def CountFPS(grabdata, frameNumber, timeStamp):
	if frameNumber % grabdata["chunkLengthInFrames"] == 0:
		timeElapsed = timeStamp - grabdata["timeStamp"][0]
		fpsCount = round(frameNumber / timeElapsed, 1)
		print('{} collected {} frames at {} fps for {} sec.'\
			.format(grabdata["cameraName"], frameNumber, fpsCount, round(timeElapsed)))
		
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

		
def SimulateFrames(n_cam, writeQueue, frameQueue, startQueue, stopReadQueue, stopWriteQueue, stop_event):

	print('initializing cameras')
	
	# Create dictionary for appending frame number and timestamp information
	#grabdata = GrabData(cam_params)
	grabdata = {'frameNumber':[], 'timeStamp':[], 'readTime':[], 'preprocess':[], 'loadonqueue':[]}

	#frames = np.ones((200, 1200, 1920, 3))

	frameNumber = 0

	print(f'Setup camera {n_cam}')

	startQueue.get(block=True) #block until receive start signal from processing module

	print('Start camera acquisition')

	while(not stopReadQueue) and (not stop_event.is_set()):
		try:
			# Append numpy array to writeQueue for writer to append to file
			#img = frames[frameNumber, :,:,:]

			preread = time.perf_counter()

			frame = read_frames('./test/example.mp4', fidxs=[frameNumber], grayscale=False) #load frame

			prepreprocess = time.perf_counter()
			
			with tf.device('/GPU:0'):
				#frame_use = frame/255
				frame_use = tf.image.convert_image_dtype(frame, tf.float32)
				imresized = tf.image.resize(frame_use, size=[600,960], method='bilinear', preserve_aspect_ratio=False, antialias=False,)
				imbgr = convert_rgb_to_bgr(imresized)
				imtranspose = tf.transpose(imbgr, perm=[0,3,1,2])

			timeStamp = time.perf_counter() # frame acquisition time

			#TODO: move img to tensorflow and convert to float32 before sending to processer
			#with tf.device('/GPU:0'):
				# Create a tensor on the GPU
			#	gpu_tensor = tf.convert_to_tensor(img)
			#gpu_tensor = tf.cast(gpu_tensor, tf.float32)
			
			frameQueue.put(imtranspose)

			post_put_on_queue = time.perf_counter()

			#CountFPS(grabdata, frameNumber, timeStamp)

			img = frame.astype(np.float32)
			writeQueue.append(img)

			frameNumber += 1

			# Append timeStamp and frameNumber to grabdata
			grabdata['frameNumber'].append(frameNumber) # first frame = 1
			grabdata['timeStamp'].append(timeStamp)
			grabdata['readTime'].append(prepreprocess-preread)
			grabdata['preprocess'].append(timeStamp-prepreprocess)
			grabdata['loadonqueue'].append(post_put_on_queue-prepreprocess)
			


		except Exception as e:
			traceback.print_exc()
			time.sleep(0.001)
			break

		except KeyboardInterrupt:
			print(f"[Cam {n_cam}] Interrupted by user")
			stop_event.set()
			break


		if frameNumber >= 1000:
			break

		#time.sleep(0.05) #sleep for 50 msec to simulate camera acquisition

	# Close the camaera, save metadata, and tell writer and display to close
	SaveSimulation(n_cam, grabdata)
	stopWriteQueue.append('STOP')

def resize_image(image: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    """
	SLEAP function ported into campy
	
	Rescale an image by a scale factor.

    This function is primarily a convenience wrapper for `tf.image.resize` that
    calculates the new shape from the scale factor.

    Args:
        image: Single image tensor of shape (height, width, channels).
        scale: Factor to resize the image dimensions by, specified as either a float
            scalar or as a 2-tuple of [scale_x, scale_y]. If a scalar is provided, both
            dimensions are resized by the same factor.

    Returns:
        The resized image tensor of the same dtype but scaled height and width.

    See also: tf.image.resize
    """
    height = tf.shape(image)[-3]
    width = tf.shape(image)[-2]
    new_size = tf.reverse(
        tf.cast(
            tf.cast([width, height], tf.float32) * tf.cast(scale, tf.float32), tf.int32
        ),
        [0],
    )
    return tf.cast(
        tf.image.resize(
            image,
            size=new_size,
            method="bilinear",
            preserve_aspect_ratio=False,
            antialias=False,
        ),
        image.dtype,
    )

def GrabFrames(cam_params, writeQueue, frameQueue, startQueue, stopReadQueue, stopWriteQueue):
	# Open the camera object
	cam, camera, cam_params = OpenCamera(cam_params, stopWriteQueue)

	# Create dictionary for appending frame number and timestamp information
	grabdata = GrabData(cam_params)

	print(f'Setup camera')

	#startQueue.get(block=True) #block until receive start signal from processing module

	print('Start camera acquisition')

	# Start grabbing frames from the camera
	grabbing = StartGrabbing(camera, cam_params, cam)

	first_run_done = False
	frameNumber = 0

	while(not stopReadQueue):
		try:
			pre_grab = perf_counter()
			# Grab image from camera buffer if available
			grabResult = cam.GrabFrame(camera, frameNumber)
			img = cam.GetImageArray(grabResult)

			#if frameNumber == 0: #Img read from camera is in rgb 
			#	np.save('./test/CamGrabSave.npy', img)

			post_grab = perf_counter()

			with tf.device('/CPU:0'):
				#frame_use = frame/255
				#frame_use = tf.image.convert_image_dtype(grabResult.Array, tf.float32)
				imresized = tf.cast(tf.image.resize(img, size=[600,960], method='bilinear', preserve_aspect_ratio=False, antialias=False,), img.dtype)
			#	imresized = resize_image(img, 0.5)
				imtranspose = tf.transpose(imresized, perm=[2,0,1])
				imbgr = convert_rgb_to_bgr(imtranspose)

			frameQueue.put(imbgr) #testing without channel flipping

			preprocess = perf_counter()

			# Append numpy array to writeQueue for writer to append to file
			#img = cam.GetImageArray(grabResult)
			writeQueue.put(img)
			# Append timeStamp and frameNumber to grabdata

			post_write = perf_counter()

			if not first_run_done: #do single first run to intialize
				frameNumber=0
				first_run_done=True
				print('First image acquired')

			else:
				frameNumber += 1

				# Display converted, downsampled image in the Window
				#if frameNumber % grabdata["frameRatio"] == 0:
					#print(imresized.shape)
					#img = cam.DisplayImage(cam_params, dispQueue, grabResult)

				grabdata['frameNumber'].append(frameNumber) # first frame = 1
				timeStamp = cam.GetTimeStamp(grabResult)
				grabdata['timeStamp'].append(timeStamp)
				grabdata['startTime'].append(pre_grab)
				grabdata["GrabTime"].append(post_grab-pre_grab)
				grabdata["PreprocessTime"].append(preprocess-post_grab)
				grabdata["LoadOnWrite"].append(post_write-preprocess)
				grabdata["ProcessQueueTimeStamp"].append(preprocess)

			CountFPS(grabdata, frameNumber, timeStamp)

			cam.ReleaseFrame(grabResult)

			if frameNumber >= grabdata["numImagesToGrab"]:
				break

		except Exception as e:
			if cam_params["cameraDebug"]:
				logging.error('Caught exception at cameras/unicam.py GrabFrames: {}'.format(e))
			time.sleep(0.001)

	writeQueue.put(None)

	# Close the camaera, save metadata, and tell writer and display to close
	cam.CloseCamera(cam_params, camera)
	SaveMetadata(cam_params, grabdata)
	stopWriteQueue.append('STOP')

def SaveSimulation(n_cam, grabdata):
	
	# Get the frame and time counts to save into metadata
	frame_count = grabdata['frameNumber'][-1]
	time_count = grabdata['timeStamp'][-1]-grabdata['timeStamp'][0]
	fps_count = int(round(frame_count/time_count))

	# Save frame data to numpy file
	npy_filename = os.path.join('./test', f'camera_{n_cam}_frametimes.npy')
	x = np.array([grabdata['frameNumber'], grabdata['timeStamp'], grabdata['readTime'], grabdata['preprocess'], grabdata['loadonqueue']])
	np.save(npy_filename,x)

def SaveMetadata(cam_params, grabdata):
	full_folder_name = os.path.join(cam_params["videoFolder"], cam_params["cameraName"])

	try:
		# Zero timeStamps
		#timeFirstGrab = grabdata["timeStamp"][0]
		#grabdata["timeStamp"] = [i - timeFirstGrab for i in grabdata["timeStamp"]]

		# Get the frame and time counts to save into metadata
		frame_count = grabdata['frameNumber'][-1]
		time_count = grabdata['timeStamp'][-1]-grabdata['timeStamp'][0]
		fps_count = int(round(frame_count/time_count))
		print('{} saved {} frames at {} fps.'.format(cam_params["cameraName"], frame_count, fps_count))

		meta = cam_params

		# Save frame data to numpy file
		npy_filename = os.path.join(full_folder_name, 'frametimes.npy')
		x = np.array([grabdata['frameNumber'], grabdata['timeStamp']])
		np.save(npy_filename,x)

		# Also save frame data to MATLAB file
		mat_filename = os.path.join(full_folder_name, 'frametimes.mat')
		matdata = {};
		matdata['frameNumber'] = grabdata['frameNumber']
		matdata['timeStamp'] = grabdata['timeStamp']
		sio.savemat(mat_filename, matdata, do_compression=True)

		# Save parameters and recording metadata to csv spreadsheet
		csv_filename = os.path.join(full_folder_name, 'metadata.csv')
		meta['totalFrames'] = grabdata['frameNumber'][-1]
		meta['totalTime'] = grabdata['timeStamp'][-1]
		
		with open(csv_filename, 'w', newline='') as f:
			w = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
			for row in meta.items():
				# Print items that are not objects or dicts
				if isinstance(row[1],(list,str,int,float)):
					w.writerow(row)

		csv_timing = os.path.join(full_folder_name, 'timestamps.csv')

		keys_to_write = ['frameNumber', 'timeStamp', 'startTime', 'GrabTime', 'PreprocessTime', 'LoadOnWrite', 'ProcessQueueTimeStamp']
		length = len(grabdata[keys_to_write[0]])

		with open(csv_timing, 'w', newline='') as f:
			w = csv.writer(f)

			w.writerow(keys_to_write)
			for i in range(length):
				row = [grabdata[key][i] for key in keys_to_write]

				w.writerow(row)

		print('Saved metadata for {}.'.format(cam_params['cameraName']))

	except Exception as e:
		logging.error('Caught exception: {}'.format(e))


def CloseSystems(systems, params):
	print('Closing systems...')
	makes = GetMakeList(params)
	for m in range(len(makes)):
		cam = ImportCam(makes[m])
		cam.CloseSystem(systems[makes[m]]["system"], systems[makes[m]]["deviceList"])
	print('Exiting campy...')
