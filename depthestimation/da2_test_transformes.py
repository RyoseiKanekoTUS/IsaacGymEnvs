from transformers import pipeline
from PIL import Image
import requests
import time
from matplotlib import pyplot as plt


# keyerror : git install from source resolved

# load pipe on to gpu
pipe_small = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device='cuda')
# pipe_large = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device='cuda')
# pipe_metric = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device='cuda')
# load image
path = './images/test/handles_1.jpeg'
# path = './images/test/isaac_tests/0.2_wall.png'
# path = './images/test/real_wood_plane/wood_4.jpg'
image = Image.open(path)
image.show()

# inference
depth_small = pipe_small(image)['depth']
# depth_large = pipe_large(image)['depth']
# depth_metric = pipe_metric(image)['depth']

depth_small.show()
# depth_large.show()
# depth_metric.show()