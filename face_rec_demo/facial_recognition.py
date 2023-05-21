import cv2
import tensorrt

import numpy as np

import time
import pdb

from trt_utils import initialize_trt_net, infer_trt, print_binding_info
from nms import detector_post_process


def run_face_detection(frame):
	
	# Resize frame to match model input.
	small_frame = cv2.resize(frame, (640, 480))

	# Normalize to match training distribution. 
	small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
	mean = np.array([127, 127, 127])
	small_frame = (small_frame - mean) / 128.0
	small_frame = np.transpose(small_frame, [2,0,1])
	small_frame = small_frame.astype(np.float32)

	# Run inference.
	det_inputs[0].host = small_frame.ravel()
	det_outputs = infer_trt(det_inputs, det_output_objs, det_bindings, det_stream, det_context)
	
	# Reshape outputs.
	scores = np.array(det_outputs[0]).reshape((17640, 2))
	boxes = np.array(det_outputs[1]).reshape((17640,4))
	
	# Run NMS and post-processing.  
	nms_boxes, labels, nms_confs = detector_post_process(frame.shape[1], frame.shape[0], scores, boxes, 0.7)
	
	return nms_boxes


def run_face_recognition(cropped_face):

	# Resize frame to match model input.
	rsz_face = cv2.resize(cropped_face, (160, 160))

	# Normalize to match training distribution.
	rsz_face = cv2.cvtColor(rsz_face, cv2.COLOR_BGR2RGB)
	mean = np.array([127, 127, 127])
	rsz_face = (rsz_face - mean) / 128.0
	rsz_face = np.transpose(rsz_face, [2,0,1])
	rsz_face = rsz_face.astype(np.float32)

	# Run inference.
	rec_inputs[0].host = rsz_face.ravel()
	rec_outputs = infer_trt(rec_inputs, rec_output_objs, rec_bindings, rec_stream, rec_context)
	face_vec = np.array(rec_outputs[0])

	return face_vec 


def cosine_similarity(vec1, vec2):
	
	sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

	return sim


# Construct GStreamer pipeline for Arducam IMX219 camera module. 
gst_pipe = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

# Initialize OpenCV VideoCapture object using the GStreamer pipeline. 
cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)

# Initialize TensorRT components. 
trtlogger = tensorrt.Logger()
# Face detection network components. 
det_inputs, det_output_objs, det_bindings, det_stream, det_context, det_engine = initialize_trt_net("RFB-640.engine", trtlogger, 1)
# Face feature extraction network components. 
rec_inputs, rec_output_objs, rec_bindings, rec_stream, rec_context, rec_engine = initialize_trt_net("FaceEmbNet.engine", trtlogger, 1)

# Preload an image to generate a face vector to recognize.
# Note: This assumes there is only one face detection in the preloaded image. 
preload_img = cv2.imread("brandon.jpg")
box = run_face_detection(preload_img)[0]
preload_face = preload_img[box[1]:box[3], box[0]:box[2]]
preload_face_vec = run_face_recognition(preload_face)

# Set a match threshold for cosine similarity. 
match_threshold = 0.6


while True:

	# Grab the most recent frame from the VideoCapture object.  
	ret, frame = cap.read()

	# Run face detection. 
	face_boxes = run_face_detection(frame)

	# Run face feature extraction, recognitions, and draw boxes based on distance to preloaded face.  
	for box in face_boxes:
		
		# Run face embedding net. 
		face_vec = run_face_recognition(frame[box[1]:box[3], box[0]:box[2]])
		
		# Compute cosine similarity with preloaded face. 
		cossim = cosine_similarity(face_vec, preload_face_vec)
		
		# Determine if face matches preload and draw box. 
		if cossim > match_threshold:
			# Draw green box if match. 
			cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 4)
		else: 
			# Draw red box if not match. 
			cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), [0, 0, 255], 4)

	# Display the frame with face recognitions to a window. 
	cv2.imshow("capture", frame)
	
	# Clean exit with 'esc' key. 
	key = cv2.waitKey(1)
	if key == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
	

	

