import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(None, "")
#trt_runtime = trt.Runtime(LOGGER)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network,LOGGER) as parser, trt.Runtime(LOGGER) as runtime:
    builder.max_batch_size = 1
    config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = (1 << 33)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    with open(r"D:\Deep_Learning\Segmentation\Forest Segmentation\forest_unet.onnx","rb") as model:
        parser.parse(model.read())

    network.get_input(0).shape = [1,3,128,128]

    plan = builder.build_serialized_network(network,config)
    #engine = runtime.deserialize_cuda_engine(plan)
    #plan = engine.serialize()

    with open("unet_forest.engine","wb") as f:
        f.write(plan)
