#!/bin/sh

/usr/src/tensorrt/bin/trtexec \
	--workspace=2500 \
	--onnx=FaceEmbNet.onnx \
	--saveEngine=FaceEmbNet.engine --fp16


