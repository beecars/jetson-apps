#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>

using namespace cv;

int main()
{
	Mat frame;
	
	// ---- INITIALIZE OPENCV VIDEO CAPTURE OBJECT
	
	// Construct GStreamer pipeline string.
	// Tuned for "aggresive" latency (downstream framerate not enforced, sink can drop frames).
	std::string rx_gstream_pipe{ };
	rx_gstream_pipe = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate = 30/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true";
	VideoCapture cap(rx_gstream_pipe, CAP_GSTREAMER);
	
	// ---- STREAM PROCESSING
	
	// Define variables for frametime measurement.
	double fps{ 0.0 };
	double fps_array[10] = { };
	auto frametime_start = std::chrono::system_clock::now();
	auto frametime_end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = { }; 

	// Begin reading stream.
	std::cout << "Processing frames...\nPress 'Q' to terminate capture.\n";
	for (;;)
	{
		frametime_start = std::chrono::system_clock::now();
		
		cap.read(frame);
		
		imshow("Live Camera Stream", frame);
		
		if (waitKey(1) == 'q')	// End reading stream.
		{
			break;
		}

	// ---- PRINT FRAMERATE 
	frametime_end = std::chrono::system_clock::now();
	elapsed_seconds = frametime_end - frametime_start;
	fps = 1.0 / elapsed_seconds.count();
	std::cout << fps << '\n';

	}

}
