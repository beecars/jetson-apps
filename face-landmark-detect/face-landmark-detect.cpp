// FACE-LANDMARK-DETECT.CPP 
// 1. Detect facial landmarks from a live camera stream.
// 2. Visualize facial landmarks in display window.

// Requirement: NVIDIA JETSON PLATFORM.
// -- (Requires "JetPack" libs: OpenCV, GStreamers, etc.)
// GStreamer Pipline based on IMX477 CAMERA SENSOR.
// -- (May require 3rd party driver for camera)

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>
using namespace cv;
 
// ---- DEBUG MODE.
/* 
	Setting DEBUG to true will disable the camera stream, and instead use a local
    image file as the network input.
*/ 
bool DEBUG = true;
std::string DEBUG_image_path = "/home/beecars/Projects/jetson-apps/face-landmark-detect/debug_image.jpg";

// ---- SETUP PARAMS. 
float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inputSize = 256;

// ---- INTIALIZE FUNCTIONS.
void processBoxes(cv::Mat& frame, const std::vector<cv::Mat>& outputBlobs);
void drawBox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

int main()
{
	// ---- INITIALIZE OPENCV VIDEO CAPTURE OBJECT.
	/* 
    	Define GStreamer pipeline based on following config:
		- Jetson Nano (L4T, JetPack).
		- IMX477 camera sensor (3rd party driver required).
		Tuned for "aggresive" latency (downstream framerate not enforced, sink can drop frames).
    */
   	cv::Mat frame;	// Current frame Mat object.
   	cv::VideoCapture cap;

	if (!DEBUG)		// If not DEBUG, assign videocapture object.
	{
		std::string rx_gstream_pipe{};
		rx_gstream_pipe = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate = 30/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true";
		cv::VideoCapture cap(rx_gstream_pipe, CAP_GSTREAMER);
	}

	// ---- INITIALIZE DNN MODULE.
	/*
		Custom face detection model from YOLOv4-tiny trained on Darknet:
		Used WIDER-FACE dataset for robust detection. 
	*/
	std::string yolo_face_cfg_Path{"/home/beecars/Projects/jetson-apps/face-landmark-detect/models/yolov4-tiny-face.cfg"};
	std::string yolo_face_weights_Path{"/home/beecars/Projects/jetson-apps/face-landmark-detect/models/yolov4-tiny-face_final.weights"};
	auto faceDetNet = dnn::readNetFromDarknet(yolo_face_cfg_Path, yolo_face_weights_Path);
	faceDetNet.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
	faceDetNet.setPreferableTarget(dnn::DNN_TARGET_CUDA);
	
	// Get output blob layer names (for dnn::Net::forward()).
	std::vector<std::string> outputBlobLayerNames = faceDetNet.getUnconnectedOutLayersNames();

	// ---- STREAM PROCESSING.

	// Define variables for frametime measurement.
	double fps{0.0};
	double fps_array[10] = {};
	auto frametime_start = std::chrono::system_clock::now();
	auto frametime_end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = {}; 

	// Begin reading stream.
	std::cout << "Processing frames...\nPress 'Q' to terminate capture.\n";
	for (;;)
	{
		frametime_start = std::chrono::system_clock::now();

		if (waitKey(1) == 'q')	// End reading stream.
		{
			break;
		}
		if (!DEBUG){
			cap.read(frame);	// Read current frame from stream.
		}
		else {
			frame = cv::imread(DEBUG_image_path);	// Read DEBUG image instead.
		}

		// ---- DETECT FACES
		// Create blob for network input (RGB image).
		Mat frameBlob = dnn::blobFromImage(frame, 
										   1/255.0, 
										   cv::Size(inputSize, inputSize), 
										   cv::Scalar(0,0,0), 
										   false, false);
		// Set blob as network input.
		faceDetNet.setInput(frameBlob);
		
		// Declare network output as blob (box coords, class names, confidences).
		std::vector<cv::Mat> outputBlobs;
		
		// Feedforward input through network to get output blobs.
		/* Details...
			"outputBlobs" come from the YOLO layers of the YOLOv4-tiny network. 
			Each blob has shape defined by:
				# rows = # YOLO grid subsets for that layer. 
				# cols = 5 + N (where N = # classes).
			Column headers are:
				| centerX | centerY | width | height | obj. score | class 0 conf. | ... | class N conf. |
		*/
		faceDetNet.forward(outputBlobs, outputBlobLayerNames);
		
		// Process and draw boxes on frame.
		processBoxes(frame, outputBlobs);
		
		// Show detections.
		imshow("Live Camera Stream", frame);

	// Print framerate to console. 
	frametime_end = std::chrono::system_clock::now();
	elapsed_seconds = frametime_end - frametime_start;
	fps = 1.0 / elapsed_seconds.count();
	std::cout << fps << '\n';
	}
}

void processBoxes(cv::Mat& frame, const std::vector<cv::Mat>& outputBlobs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	// Iterate through all blobs output from the network. 
	for (size_t i = 0; i < outputBlobs.size(); ++i)
	{
		// Get pointer for float data stored in <cv::Mat>outputBlobs.
		float* data = (float*)outputBlobs[i].data;
		// Iterate through each entry (row) in the <cv::Mat>outputBlob. 
		for (int j = 0; j < outputBlobs[i].rows; ++j, data += outputBlobs[i].cols)
		{
			cv::Mat scores = outputBlobs[i].row(j).colRange(5, outputBlobs[i].cols);
			cv::Point classIdPoint;
			double confidence;
			// Get value and location of max score.
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Non-maximum supperssion to remove redundant, lower-confidence boxes. 
	std::vector<int> indexes;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indexes);
	for (size_t i = 0; i < indexes.size(); ++i)
	{
		int idx = indexes[i];
		cv::Rect box = boxes[idx];
		drawBox(classIds[idx], confidences[idx], 
			    box.x, box.y, 
				box.x + box.width, 
				box.y + box.height, 
				frame);
	}
}

void drawBox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
	std::string label = format("%.2f", conf);
	
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	cv::rectangle(frame, 
				  cv::Point(left, top - round(1.5 * labelSize.height)), 
				  cv::Point(left + round(1.5 * labelSize.width), top + baseLine), 
				  cv::Scalar(255, 255, 255), FILLED);
	cv::putText(frame, label, Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}