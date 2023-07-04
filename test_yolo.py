#%%
import common
import tensorrt as trt
# %%
logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)
trt.init_libnvinfer_plugins(logger, "")

with open('sample.engine', "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
# %%
context = engine.create_execution_context()
inputs, outputs, bindings, stream = common.allocate_buffers(engine)

#%%
import cv2
import numpy as np
import scipy

image = cv2.imread('test.jpg')
H, W = image.shape[:2]
resized = cv2.resize(image, (640, 640))
normalized = resized[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255

inputs[0].host = normalized.copy()

trt_outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
res = outputs[0].host.reshape(84, 8400).transpose()

max_prob = res[:, 4:].max(axis=1)
max_cat = res[:, 4:].argmax(axis=1)
index = max_prob > 0.3
filtered = res[index]
cx = filtered[:, 0]
cy = filtered[:, 1]
w2 = filtered[:, 2] / 2
h2 = filtered[:, 3] / 2
x1 = cx - w2
y1 = cy - h2
x2 = cx + w2
y2 = cy + h2
conf = max_prob[index]
cat = max_cat[index]

boxes = np.stack([x1, y1, x2, y2, conf, cat], axis=1)

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
results = []
for box in boxes:
    for r in results:
        if iou(r, box) > 0.7:
            break
    else:
        results.append(box)

dst = image.copy()
for row in results:
    x1, y1, x2, y2 = row[:4]
    x1 = int(x1 / 640 * W)
    y1 = int(y1 / 640 * H)
    x2 = int(x2 / 640 * W)
    y2 = int(y2 / 640 * H)
    
    cv2.rectangle(dst, (x1, y1), (x2, y2), (255, 0, 0), 3, -1)

cv2.imwrite('dst.jpg', dst)
# %%
