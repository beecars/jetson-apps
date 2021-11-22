// FACE-LANDMARK-DETECT.CPP 
// 1. Detect facial landmarks from a live camera stream.
// 2. Visualize facial landmarks in display window.

// Requirement: NVidia Jetson platform with L4T and NVidia Jetpack.
// Requirement: OpenCV built from source with CUDA, GStreamer, DNN, 
//        		and "contrib" modules.
// 				
// GStreamer Pipline based on IMX477 CAMERA SENSOR.
// -- May require 3rd party driver.

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
 
// ---- SETUP PARAMS. 
float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inputSize = 256;

// ---- INTIALIZE FUNCTIONS.
void processBoxes(cv::Mat& frame, const std::vector<cv::Mat>& outputBlobs);
void drawBox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

int main()
{
	/* ---- INITIALIZE OPENCV VIDEO CAPTURE OBJECT.
       Define GStreamer pipeline based on following config:
	   - Jetson Nano (L4T, JetPack).
	   - IMX477 camera sensor (3rd party driver required).
	   Tuned for "aggresive" latency (downstream framerate not enforced, sink can drop frames).
    */
   	cv::Mat frame;	

	std::string rx_gstream_pipe{};
	rx_gstream_pipe = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate = 30/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true";
	cv::VideoCapture cap(rx_gstream_pipe, CAP_GSTREAMER);

	/* ---- INITIALIZE DNN MODULE.
	   Custom face detection model from YOLOv4-tiny trained on Darknet:
	   Used WIDER-FACE dataset for robust detection. 
	*/
	std::string yolo_face_cfg_Path{"../models/yolov4-tiny256-face.cfg"};
	std::string yolo_face_weights_Path{"../models/yolov4-tiny256-face.weights"};
	auto faceDetNet = dnn::readNetFromDarknet(yolo_face_cfg_Path, yolo_face_weights_Path);
	faceDetNet.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
	faceDetNet.setPreferableTarget(dnn::DNN_TARGET_CUDA);
	
	// Get output blob layer names (for dnn::Net::forward()).
	std::vector<std::string> outputBlobLayerNames = faceDetNet.getUnconnectedOutLayersNames();

	// ---- STREAM PROCESSING.
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
		if (waitKey(1) == 'q'){
			break;
		}

		cap.read(frame);	// Read current frame from stream.

		// ---- DETECT FACES
		// Create blob for network input (RGB image).
		Mat frameBlob = dnn::blobFromImage(frame, 
										   1/255.0, 
										   cv::Size(inputSize, inputSize), 
										   cv::Scalar(0,0,0), 
										   false, false);

		faceDetNet.setInput(frameBlob);
		
		// Feedforward input through network to get output blobs.
		/* Details...
			"outputBlobs" come from the YOLO layers of the YOLOv4-tiny network. 
			Each blob has shape defined by:
				# rows = # YOLO grid subsets for that layer. 
				# cols = 5 + N (where N = # classes).
			Column headers are:
				| centerX | centerY | width | height | obj. score | class 0 conf. | ... | class N conf. |
		*/
		std::vector<cv::Mat> outputBlobs;
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
	std::vector<float> detectedScores;
	std::vector<cv::Rect> detectedBoxes;
	// Iterate through all blobs output from the network. 
	for (size_t i = 0; i < outputBlobs.size(); ++i)
	{
		// Get pointer for float data stored in <cv::Mat>outputBlobs.
		float* data = (float*)outputBlobs[i].data;
		// Iterate through each entry (row) in the <cv::Mat>outputBlob. 
		for (int j = 0; j < outputBlobs[i].rows; ++j, data += outputBlobs[i].cols)
		{
			cv::Mat classScores = outputBlobs[i].row(j).colRange(5, outputBlobs[i].cols);
			cv::Point classId;
			double highestClassScore;

			/* Get value and location of max score. 
			   cv::minMaxLoc is a clever hack-y way to do this. It takes the cv::Mat of 
			   potential classScores, which is essentially a 1D array, and puts the 
			   highest score in highestClassScore and the location is stored as a "point",
			   where the x-coordinate represents the class index. For this single-class
			   case, this is all ridiculous, but the generality is important for multi-class
			   cases, so I leave it here. 
			*/
			cv::minMaxLoc(classScores, 0, &highestClassScore, 0, &classId);
			if (highestClassScore > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classId.x);
				detectedScores.push_back((float)highestClassScore);
				detectedBoxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Non-maximum supperssion to remove redundant, lower-confidence boxes. 
	std::vector<int> indexes;
	dnn::NMSBoxes(detectedBoxes, detectedScores, confThreshold, nmsThreshold, indexes);
	for (size_t i = 0; i < indexes.size(); ++i)
	{
		int idx = indexes[i];
		cv::Rect bestBox = detectedBoxes[idx];
		
		drawBox(classIds[idx], 
				detectedScores[idx], 
			    bestBox.x, 
				bestBox.y, 
				bestBox.x + bestBox.width, 
				bestBox.y + bestBox.height, 
				frame);
	}
}

void drawBox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, 
			  cv::Point(left, top), 
			  cv::Point(right, bottom), 
			  cv::Scalar(255, 178, 50), 3);

	std::string label = format("%.2f", conf);
	
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	
	cv::rectangle(frame, 
				  cv::Point(left, top - round(1.5 * labelSize.height)), 
				  cv::Point(left + round(1.5 * labelSize.width), top + baseLine), 
				  cv::Scalar(255, 255, 255), FILLED);
	
	cv::putText(frame, 
				label, 
				Point(left, top), 
				cv::FONT_HERSHEY_SIMPLEX, 
				0.75, 
				cv::Scalar(0, 0, 0), 1);
}