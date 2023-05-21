import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

class HostDeviceMem(object):
    """ A helper object for storing allocated host and device memory locations.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def initialize_trt_net(engine_file, logger, batch_size):
    """ Load and initialize an TRT net from an existing engine file.

    Parameters
    ----------
    engine_file: string (filepath)
        Path to TensorRT engine file.
    logger: Logger (TensorRT object)
        Initialized TensorRT Logger object.
    batch_size: int
        Integer representing batch size. In current implementation, this needs to be 1, even for
        multibatch processing. For multibatch processing, build the TensorRT engine with the
        "batch-size" flag and initialize the engine here with "batch_size=1".

    Returns
    -------
    inputs: list of HostDeviceMem object(s)
    outputs: list of HostDeviceMem objects(s)
    bindings: list of TensorRT bindings
    stream: CUDA Stream object
    context: CUDA Context object
    engine: TensorRT Engine object
    """
    # Load custom TRT plugins.
    trt.init_libnvinfer_plugins(logger, namespace="")
    # Initialize TRT runtime.
    trt_runtime = trt.Runtime(logger)
    # Load TRT engine.
    engine = load_engine(trt_runtime, engine_file)
    # Allocate memory buffers.
    inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size)
    # Create exectution context.
    context = engine.create_execution_context()

    return inputs, outputs, bindings, stream, context, engine


def preprocess_image_batch(batch, model_size, norm_type=0):
    """ Preprocess image batch for TRT processing.

    Parameters
    ----------
    batch: list of nparray (3d) (H, W, C)
        Batch list of images to run inference on. H, W dimensions do not need to be sized for YOLOv7
        model. Will be resized as a preprocessing step.
    model_size: tuple(int) (H, W)
        Input (H, W) of the model the batch is being sent to. Batch (H, W) will be resized to this
        dimension.
    norm_type: int (0 or 1)
        0 - Linear transform of elements to [0, 1] range.
        1 - Linear transform of elements to [-1, 1] range.

    Returns
    -------
    batch: nparray (4d) (N, C, W_resized, H_resized)
        Preprocessed batch.
    """
    # Get resized shape.
    resize_h, resize_w = model_size
    resize_n, resize_c = len(batch), batch[0].shape[-1]
    resized_batch = np.zeros((resize_n, resize_h, resize_w, resize_c))

    # Resize each image and fill the initialized array.
    for bidx in range(resize_n):

        img = cv2.resize(batch[bidx], (resize_w, resize_h))
        resized_batch[bidx,:,:,:] = img

    # Permute the dimensions.
    resized_batch = resized_batch.transpose((0, 3, 1, 2)).astype(np.float32)   # N x H x W x C -->  N x C x H x W

    # Normalize based on normalization mode.
    if norm_type == 0:      # [0, 255] --> [0, 1]
        resized_batch /= 255.0
    elif norm_type == 1:    # [0, 255] --> [-1, 1]
        resized_batch = (img.astype(np.float32) - 127.5) / 128

    return resized_batch


def preprocess_image(img, size, norm_type=0):
    """ Preprocess image for TRT processing.

    Parameters
    ----------
    img: nparray (3d) (H, W, C)
        Image to run inference on. H, W dimensions do not need to be sized for YOLOv7
        model. Will be resized as a preprocessing step.
    model_size: tuple(int) (H, W)
        Input (H, W) of the model the batch is being sent to. Batch (H, W) will be resized to this
        dimension.
    norm_type: int (0 or 1)
        0 - Linear transform of elements to [0, 1] range.
        1 - Linear transform of elements to [-1, 1] range.

    Returns
    -------
    batch: nparray (3d) (C, W_resized, H_resized)
        Preprocessed image.
    """
    model_w, model_h = size
    img = cv2.resize(img, (model_w, model_h))

    # Check if grayscale image
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1)).astype(np.float32)

    if norm_type == 0:
        img /= 255.0
    elif norm_type == 1:
        img = (img.astype(np.float32) - 127.5) / 128

    return img


def infer_trt(inputs, outputs, bindings, stream, context, batch_size=1):
    """ Run inference sequence on TRT engine.

    Parameters
    ----------
    inputs: list of HostDeviceMem object(s)
    outputs: list of HostDeviceMem objects(s)
    bindings: list of TensorRT bindings
    stream: CUDA Stream object

    Returns
    -------
    host_outputs: list (nparray)
    """
    # Copy image to appropriate place in memory.
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize the stream
    stream.synchronize()

    # Isolate host outputs.
    host_outputs = [out.host for out in outputs]

    return host_outputs


def reshape_trt_outputs(outputs):
    """ Reshape YOLOv7 TensorRT ouputs from ravelled to column vector style.
    """
    num_dets = outputs[0][0]
    len_dets = len(outputs[2])
    boxes = np.array(outputs[1]).reshape((len_dets,4))[0:num_dets]
    confs = np.array(outputs[2])[0:num_dets]
    labels = np.array(outputs[3])[0:num_dets]

    return boxes, confs, labels

def predict(image, batch_size, model_size, inputs, outputs, bindings, stream, context, norm_type=0):
    """Infers model on batch of same sized images resized to fit the model.
    Args:
        image (str): paths to images, that will be packed into batch
            and fed into model
        batch_size : TBD
        model_size : TBD
        inputs : TBD
        outputs : TBD
        bindings : TBD
        stream : TBD
        context : TBD
    """

    img = preprocess_image(image, model_size, norm_type)
    # Copy it into appropriate place into memory
    # (self.inputs was returned earlier by allocate_buffers())

    np.copyto(inputs[0].host, img.ravel())

    # Fetch output from the model
    trt_outputs = infer_trt(inputs, outputs,bindings, stream, context)

    return trt_outputs


def rescale_boxes(boxes, h_sfac, w_sfac):
    """Scale YOLOv7 detection boxes to original image coordinate space.
    """
    boxes[:, 0] = boxes[:, 0] * w_sfac
    boxes[:, 1] = boxes[:, 1] * h_sfac
    boxes[:, 2] = boxes[:, 2] * w_sfac
    boxes[:, 3] = boxes[:, 3] * h_sfac
    return boxes


def draw_det_boxes(img, boxes, confs, labels, class_names, colors):
    for box, conf, label in zip(boxes, confs, labels):
        # Get class name and visualization color.
        name = class_names[label]
        color = colors[name]
        name += ' ' + str(round(float(conf), 3))
        # Draw.
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color, 1)
        cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness = 1)

    return img


def draw_track_boxes(img, boxes, confs, target_ids):
    """Convert tlwh boxes to x1y1x2y2 boxes.
    """
    boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        return img
    else:
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        for box, conf, target_id in zip(boxes, confs, target_ids):
            color = [0, 255, 255]
            box = box.astype(int)
            # Draw.
            cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color, 1)
            cv2.putText(img, str(target_id), (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness = 1)

    return img


def load_engine(trt_runtime, engine_path):
    """ Deserialize a TRT engine file.

    Parameters
    ----------
    trt_runtime: TensorRT Runtime object
    engine_path: string (filepath)
        Path to TensorRT engine file.

    Returns
    -------
    engine: TensorRT Engine object
    """
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine, batch_size):
    """ Allocate memory buffers for inputs and ouputs on CPU and GPU.

    Parameters
    ----------
    engine: TensorRT Engine object
    batch_size: int
        Batch size. Probably should be 1 unless a good reason for it not to be.

    Returns
    -------
    inputs: list of HostDeviceMem object(s)
    outputs: list of HostDeviceMem objects(s)
    bindings: list of TensorRT bindings
    stream: CUDA Stream object
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = abs(trt.volume(engine.get_binding_shape(binding)) * batch_size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def get_input_shape(engine):
    """Get input shape of the TRT engine.

    Parameters
    ----------
    engine: TensorRT engine object

    Returns
    -------
    binding_dims: nparray
        A numpy representing the [H, W] dimensions of the input to the TensorRT model.
    """
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)

    return binding_dims

def print_binding_info(trt_engine):
    """ Print binding information from deserialized TRT engine.

    Parameters
    ----------
    trt_engine: TensorRT Engine object
    """
    # Print engine binding info.
    print("---------------------------------------")
    print("Engine binding information:")
    num_bindings = trt_engine.num_bindings
    for idx in range(num_bindings):
        print(trt_engine[idx] + "\t" + str(trt_engine.get_binding_dtype(idx)) + "\t" + str(trt_engine.get_binding_shape(idx)))
    print("---------------------------------------")
