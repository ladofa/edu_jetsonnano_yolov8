#%%
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file('yolov8n.onnx')
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    pass # Error handling code here
config = builder.create_builder_config()

#%%
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
serialized_engine = builder.build_serialized_network(network, config)
#%%
with open("sample.engine", "wb") as f:
    f.write(serialized_engine)

# %%
