import json

import cv2
import numpy as np
import zstandard
from tqdm import tqdm


def pixel_to_point(pixel, depth, config):
    x = (pixel[0] - config['ppx']) / config['fx']
    y = (pixel[1] - config['ppy']) / config['fy']

    point = [depth * x, depth * y, depth]
    return point


def point_to_pixel(point, config):
    x = point[0] / point[2]
    y = point[1] / point[2]

    pixel = [x * config['fx'] + config['ppx'], y * config['fy'] + config['ppy']]
    return pixel


def depth_to_camera(depth_frame, depth_scale, depth_config, color_config):
    depth_in = depth_frame.flatten()
    depth_out = np.zeros_like(depth_in)

    for depth_y in range(depth_config['height']):
        depth_pixel_index = depth_y * depth_config['width']
        for depth_x in range(depth_config['width']):
            depth = depth_scale * depth_in[depth_pixel_index]
            depth_pixel_index += 1
            if depth <= 0:
                continue

            # Map the top-left corner of the depth pixel onto the color image
            depth_pixel = [depth_x - 0.5, depth_y - 0.5]
            color_point = pixel_to_point(depth_pixel, depth, depth_config)
            color_pixel = point_to_pixel(color_point, color_config)

            color_x0 = int(color_pixel[0] + 0.5)
            color_y0 = int(color_pixel[1] + 0.5)

            if color_x0 < 0 or color_y0 < 0:
                continue

            # Map the bottom-right corner of the depth pixel onto the color image
            depth_pixel = [depth_x + 0.5, depth_y + 0.5]
            color_point = pixel_to_point(depth_pixel, depth, depth_config)
            color_pixel = point_to_pixel(color_point, color_config)

            color_x1 = int(color_pixel[0] + 0.5)
            color_y1 = int(color_pixel[1] + 0.5)

            if color_x1 >= color_config['width'] or color_y1 >= color_config['height']:
                continue

            # Transfer between the depth pixels and the pixels inside the rectangle on the other image
            for y in range(color_y0, color_y1 + 1):
                for x in range(color_x0, color_x1 + 1):
                    color_pixel_index = y * color_config['width'] + x
                    if depth_out[color_pixel_index] == 0:
                        depth_out[color_pixel_index] = depth_in[depth_pixel_index]
                    else:
                        depth_out[color_pixel_index] = min(depth_out[color_pixel_index],
                                                           depth_in[depth_pixel_index])

    depth_out = depth_out.reshape((color_config['height'], color_config['width']))
    return depth_out


def read_camera_frame(data_dir, start_frame=None, end_frame=None):
    color_data_file = f'{data_dir}color.avi'

    cap = cv2.VideoCapture(color_data_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    color_data = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    for frame_idx in range(frame_count):
        _, frame = cap.read()
        color_data[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if start_frame is not None and end_frame is not None:
        color_data = color_data[start_frame:end_frame]

    return color_data


def load_color_config(data_dir):
    color_config_file = f'{data_dir}color_config.json'

    with open(color_config_file) as f:
        color_config_file = json.load(f)

    color_config = color_config_file['intrinsics']
    return color_config


def load_depth_config(data_dir):
    depth_config_file = f'{data_dir}depth_config.json'

    with open(depth_config_file) as f:
        depth_config_file = json.load(f)

    depth_scale, depth_config = depth_config_file['depth_units'], depth_config_file['intrinsics']
    return depth_scale, depth_config


def read_depth_frame(data_dir, start_frame=None, end_frame=None):
    depth_data_file = f'{data_dir}depth.zst'
    depth_scale, depth_config = load_depth_config(data_dir)

    decom = zstandard.ZstdDecompressor()
    with open(depth_data_file, 'rb') as f:
        depth_data = decom.stream_reader(f)
        depth_data = depth_data.read()

    depth_data = np.frombuffer(depth_data, dtype=np.uint16).copy()
    depth_data = depth_data.reshape((-1, depth_config['height'], depth_config['width']))

    if start_frame is not None and end_frame is not None:
        depth_data = depth_data[start_frame:end_frame]

    return depth_data


def align_depth_camera(data_dir, raw_depth_data, cpp):
    color_config = load_color_config(data_dir)
    depth_scale, depth_config = load_depth_config(data_dir)

    depth_data = raw_depth_data.copy()
    frame_count = raw_depth_data.shape[0]
    for frame_idx in tqdm(range(frame_count)):
        if not cpp:
            depth_data[frame_idx] = depth_to_camera(
                raw_depth_data[frame_idx], depth_scale, depth_config, color_config
            )
        else:
            depth_data[frame_idx] = depth_project.depth_to_camera(
                raw_depth_data[frame_idx], depth_scale, depth_config, color_config
            )

    return depth_data


def make_3d(fx, fy, depth, cx, cy, u, v):
    return np.stack(
        [(u - cx) * depth / fx, (v - cy) * depth / fy, depth], axis=-1
    )


def make_point_cloud(data_dir, rgbd_data):
    color_config = load_color_config(data_dir)

    w, h = color_config['width'], color_config['height']
    fx, fy, cx, cy = color_config['fx'], color_config['fy'], color_config['ppx'], color_config['ppy']
    u, v = np.arange(w), np.arange(h)
    u, v = np.meshgrid(u, v)

    pc_frames = []
    for rgbd_frame in rgbd_data:
        xyz, rgb = make_3d(fx, fy, rgbd_frame[:, :, 3], cx, cy, u, v), rgbd_frame[:, :, :-1]
        point_cloud = np.concatenate((xyz, rgb), axis=-1).reshape(-1, 6)
        pc_frames.append(point_cloud)

    pc_frames = np.array(pc_frames, dtype=np.float32)
    return pc_frames


def fast_normalize(img, dtype=np.uint8, alpha=0, beta=255):
    img_min = img.min()
    img_max = img.max()
    norm_img = (img - img_min) * (beta - alpha) / (img_max - img_min + 1e-8) + alpha
    return norm_img.astype(dtype)
