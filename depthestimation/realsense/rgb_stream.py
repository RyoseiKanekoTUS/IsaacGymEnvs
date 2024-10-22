import pyrealsense2 as rs
import numpy as np
import cv2

import time


# image config
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

# start streaming
pipeline.start(config)


def get_frame():

    # 
    frames = pipeline.wait_for_frames()
    RGB_frame = frames.get_color_frame()
    np_rgb_frame = np.asanyarray(RGB_frame.get_data(),dtype=np.uint8)

    return np_rgb_frame


def vis_cv2(np_rgb):

    cv2_rgb = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    # visualized by cv2
    cv2.namedWindow('RealSense', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('RealSense', cv2_rgb)


def stream():

    try:
        while True:
            t_0 = time.time()

            rgb = get_frame()
            print(rgb.shape)
            vis_cv2(rgb)

            if cv2.waitKey(1) & 0xff == 27:# killed by ESC key
                cv2.destroyAllWindows()
                break

            print('FPS :', 1.0 / (time.time() - t_0))

    finally:
        pipeline.stop()



if __name__ == '__main__':

    stream()
