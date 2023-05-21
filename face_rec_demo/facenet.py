"""
    TensorRT implementation of the PyTorch Facenet implementation to
    extract facial feature embeddings
"""

import numpy as np

from ..trt.trt_utils import get_input_shape, initialize_trt_net, infer_trt, preprocess_image, allocate_buffers

import pdb

class FaceNet():
    """
    FaceNet object constructs model to be used for extracting
    facial feature embeddings

    Parameters
    ----------
    cfg : cfgNode
        A configuration node to define model parameters

    trt_runtime : tensorrt.tensorrt.Runtime
        Object that enables :class:'ICudaEngine' to be deserialized

    Attributes
    ---------
    cfg : cfgNode
        A configuration node to define model parameters

    batch_size : int
        Total number of images to process with single forward pass

    width : int
        Number of columns for input image

    height : int
        Number of rows for input image

    channels : int
        Number of channels for input image

    engine : ...
        Deserialized TensorRT engine

    inputs : TODO

    outputs : TODO

    bindings : TODO

    stream : TODO

    context : TODO


    """

    def __init__(self, cfg, trt_logger):

        self.cfg = cfg
        self.batch_size = 1
        self.width = self.cfg.IMAGE_WIDTH
        self.height = self.cfg.IMAGE_HEIGHT
        self.channels = self.cfg.CHANNELS
        self.out_dims = self.cfg.DIMENSIONALITY

        # Initialize TensorRT engine
        self.trt_logger = trt_logger
        self.engine_file = self.cfg.ENGINE_FILE
        inputs, outputs, bindings, self.stream, self.context, self.engine = \
            initialize_trt_net(self.engine_file, self.trt_logger, self.batch_size)
        self.input_shape = get_input_shape(self.engine)

    def infer_single_image(self, images):
        """
        Processes an image and returns a facial embedding vector.
        Assumes image of cropped face

        Parameters
        ----------
        image : np.ndarray, dtype=int, float, double (any precision)
            Array representing a cropped face in HxWxC format

        Returns
        -------
        : np.ndarray, dtype=np.float32
            Facial feature embedding vector of shape (512,)
        : float
            Total image processing time

        """

        images_np = np.array([preprocess_image(image, (self.input_shape[2], self.input_shape[3]), norm_type=1) for image in images])
        batch_size = len(images_np)

        # Allocate memory buffers.
        inputs, outputs, bindings, stream = allocate_buffers(self.engine, batch_size)
        inputs[0].host = images_np.ravel()

        self.context.set_binding_shape(0, (batch_size,self.channels,self.width,self.height))
        trt_outputs = infer_trt(inputs, outputs, bindings, stream, self.context, batch_size=batch_size)
        return trt_outputs[0].reshape(batch_size, self.out_dims)

if __name__ == '__main__':
    pass
