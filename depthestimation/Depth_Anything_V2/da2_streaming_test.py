import pyrealsense2 as rs
import numpy as np
import torch
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

import time


# realsense config
fps = 90
raw_width, raw_height = (640, 480)
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.color, raw_width, raw_height, rs.format.rgb8, fps)

# depthanything config
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
encoder = 'vits'
# encoder = 'vitl'
MODEL = DepthAnythingV2(**model_configs[encoder])
MODEL.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))

MODEL = MODEL.to(DEVICE).eval()


# start streaming
pipeline.start(config)


def get_frame():

    # 
    frames = pipeline.wait_for_frames()
    RGB_frame = frames.get_color_frame()
    np_rgb_frame = np.asanyarray(RGB_frame.get_data(),dtype=np.uint8)

    return np_rgb_frame


def vis_rgb_cv2(np_rgb):

    cv2_rgb = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    # visualized by cv2
    cv2.namedWindow('RealSense', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('RealSense', cv2_rgb)


def vis_depth_cv2(raw_depth):

    depth_map = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
    depth_map = depth_map.astype(np.uint8)
    cv2.namedWindow('DepthAnything_V2', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('DepthAnything_V2', depth_map)


def depthanything_v2(raw_rgb):

    depth = MODEL.infer_image(raw_rgb)

    return depth

def stream():

    try:
        while True:
            t_0 = time.time()

            rgb = get_frame()
            depth = depthanything_v2(rgb)
            vis_depth_cv2(depth)

            if cv2.waitKey(1) & 0xff == 27:# killed by ESC key
                cv2.destroyAllWindows()
                break

            print('FPS :', 1.0 / (time.time() - t_0))

    finally:
        pipeline.stop()



if __name__ == '__main__':

    stream()
