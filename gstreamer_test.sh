gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM)' ! nvvidconv flip-method=6 ! nvoverlaysink
