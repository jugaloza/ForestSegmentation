import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


img = cv2.imread(r"D:\Deep_Learning\Segmentation\Forest Segmentation\Forest_Segmented\images\3484_sat_34.jpg",cv2.IMREAD_COLOR)
img = cv2.resize(img,(128,128))
img = img.ravel()

#print(img.shape)

LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(LOGGER)

with open("unet_forest.engine","rb") as f:
    engine_data = f.read()

engine = runtime.deserialize_cuda_engine(engine_data) 

hostmem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),trt.nptype(engine.get_binding_dtype(0)))
hostoutput = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),trt.nptype(engine.get_binding_dtype(1)))

d_input_1 = cuda.mem_alloc(hostmem.nbytes)
d_output = cuda.mem_alloc(hostoutput.nbytes)

img = img / 255.

np.copyto(img,hostmem)

stream = cuda.Stream()

print("Executing context")

with engine.create_execution_context() as context:

    cuda.memcpy_htod_async(d_input_1,hostmem,stream)

    context.execute(1, bindings=[int(d_input_1),int(d_output)])

    cuda.memcpy_dtoh_async(hostoutput,d_output,stream)

    stream.synchronize()

    #out = hostoutput.reshape((batchsize,))

#hostoutput = hostoutput.reshape(128,128,1)
#cv2.imwrite("img.png",hostoutput)
print(hostoutput)