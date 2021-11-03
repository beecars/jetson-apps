#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace cv;

int main()
{
	Mat frame;
	
	// INITIALIZE VIDEO CAPTURE OBJECT
	VideoCapture cap;
	cap.open(0);
	cap.set(CAP_PROP_FRAME_WIDTH, 800);
	cap.set(CAP_PROP_FRAME_HEIGHT, 600);
	cap.set(CAP_PROP_FPS, 30);

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