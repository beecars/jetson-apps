#!/bin/sh

/usr/src/tensorrt/bin/trtexec \
	--workspace=2500 \
	--onnx=RFB-640.onnx \
	--saveEngine=RFB-640.engine --fp16


