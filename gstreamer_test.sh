gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=(int)800, height=(int)600' ! nvvidconv flip-method=6 ! nvoverlaysink
