gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=(int)1280, height=(int)720' ! nvvidconv flip-method=6 ! nvoverlaysink
