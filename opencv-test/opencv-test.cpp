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
	
	// INITIALIZE VIDEO CAPTURE OBJECT.

	// Construct GStreamer pipeline string.
	std::string rx_gstream_pipe{ };
	rx_gstream_pipe = "nvarguscamerasrc ! \
					       video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, \
						   framerate = 30/1 ! \
			           nvvidconv flip-method=6 ! \
					       video/x-raw, width=(int)1280, height=(int)720, framerate = 30/1, \
						   format=(string)BGRx ! \
				 	   videoconvert ! \
					       video/x-raw, format=(string)BGR ! \
				       appsink";
	
	// GStreamer pipeline 2
	//VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM)! nvvidconv flip-method=6 ! appsink", CAP_GSTREAMER);

	VideoCapture cap(rx_gstream_pipe, CAP_GSTREAMER);

	if (!cap.isOpened())
	{
		std::cerr << "ERROR: Unable to open camera capture stream.";
		return -1;
	}
	
	// STREAM PROCESSING
	std::cout << "Processing frames...\nPress 'Q' to terminate capture.\n";
	
	double fps{ 0.0 };
	double fps_array[10] = { };
	auto frametime_start = std::chrono::system_clock::now();
	auto frametime_end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = { }; 

	// begin reading stream
	for (;;)
	{
		frametime_start = std::chrono::system_clock::now();
		
		cap.read(frame);
		
		imshow("Live Camera Stream", frame);
		
		if (waitKey(1) == 'q')	// end reading steam
		{
			break;
		}

	// PRINT FRAMERATE 
	frametime_end = std::chrono::system_clock::now();
	elapsed_seconds = frametime_end - frametime_start;
	fps = 1.0 / elapsed_seconds.count();
	std::cout << fps << '\n';

	}

}
