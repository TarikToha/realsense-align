import cv2
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error, structural_similarity
from tqdm import tqdm

from camera_utility import load_color_config, load_depth_config

np.set_printoptions(suppress=True)

obj_mapping = {
    'std_brick_object': 'mini_pc_std_obj_cascaded_dataset',
    'std_t_object': 'mini_pc_new_cascaded_dataset',
    'old_t_object': 'cascaded_dataset'
}


def compute_remap(depth_intrinsics, color_intrinsics, depth_shape):
    """
    Computes remapping matrices for OpenCV's cv2.remap() function to align depth to color image.

    Args:
        depth_intrinsics (dict): Intrinsics of the depth camera.
        color_intrinsics (dict): Intrinsics of the color camera.
        depth_shape (tuple): (height, width) of the depth image.

    Returns:
        map_x, map_y (np.ndarray): Remap matrices for cv2.remap().
    """
    h_c, w_c = color_intrinsics['height'], color_intrinsics['width']
    map_x = np.zeros((h_c, w_c), dtype=np.float32)
    map_y = np.zeros((h_c, w_c), dtype=np.float32)

    fx_d, fy_d = depth_intrinsics['fx'], depth_intrinsics['fy']
    cx_d, cy_d = depth_intrinsics['ppx'], depth_intrinsics['ppy']

    fx_c, fy_c = color_intrinsics['fx'], color_intrinsics['fy']
    cx_c, cy_c = color_intrinsics['ppx'], color_intrinsics['ppy']

    for v_c in range(h_c):
        for u_c in range(w_c):
            # Back-project to normalized coordinates in color camera
            x = (u_c - cx_c) / fx_c
            y = (v_c - cy_c) / fy_c
            z = 1.0  # arbitrary depth (unit ray)

            # Project to depth image
            u_d = fx_d * x / z + cx_d
            v_d = fy_d * y / z + cy_d

            map_x[v_c, u_c] = u_d
            map_y[v_c, u_c] = v_d

    return map_x, map_y


def register_depth_to_color(depth_image, map_x, map_y, depth_scale=1.0):
    """
    Registers depth image to color frame using precomputed maps and cv2.remap().

    Args:
        depth_image (np.ndarray): Input depth image (uint16 or float32).
        map_x (np.ndarray): X remap matrix from compute_remap().
        map_y (np.ndarray): Y remap matrix from compute_remap().
        depth_scale (float): Scale factor to convert raw depth to meters.

    Returns:
        registered_depth (np.ndarray): Warped depth image in color frame.
    """
    depth_float = depth_image.astype(np.float32) * depth_scale
    registered = cv2.remap(depth_float, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    return registered


base_dir = '/home/ttoha12/weapon/'
data = pd.read_csv('overall_selection_9.csv')
total_rows = data.shape[0]

results = []
for _, row in tqdm(data.iterrows(), total=total_rows, desc='align_eval'):
    capture_dir, capture_id = row['Dataset'], row['Capture_ID']
    capture_dir = obj_mapping[capture_dir]
    cam_dir = f'{base_dir}{capture_dir}/capture_{capture_id:05d}/realsense/'

    raw_depth_data = f'{cam_dir}raw_depth.npy'
    git_depth_data = f'examples/{capture_dir}_{capture_id}.npy'
    depth_data = f'{cam_dir}depth.npy'

    raw_depth_data = np.load(raw_depth_data)
    git_depth_data = np.load(git_depth_data)
    depth_data = np.load(depth_data)

    color_config = load_color_config(cam_dir)
    depth_scale, depth_config = load_depth_config(cam_dir)

    map_x, map_y = compute_remap(depth_config, color_config, raw_depth_data[0].shape)
    start_frame, end_frame = row['Start_Frame'], row['End_Frame']

    val = []
    for frame_id in range(start_frame, end_frame):
        D_pred0 = raw_depth_data[frame_id]
        D_pred1, D_pred2 = git_depth_data[frame_id] * 1e3, depth_data[frame_id]
        D_gt = register_depth_to_color(D_pred0, map_x, map_y, depth_scale=depth_scale * 1e3)

        rmse0 = np.sqrt(mean_squared_error(D_pred0, D_gt))
        rmse1 = np.sqrt(mean_squared_error(D_pred1, D_gt))
        rmse2 = np.sqrt(mean_squared_error(D_pred2, D_gt))
        # print(frame_id, 'rmse', rmse0, rmse1, rmse2)

        ssim0, _ = structural_similarity(D_pred0, D_gt, full=True, data_range=D_gt.max() - D_gt.min())
        ssim1, _ = structural_similarity(D_pred1, D_gt, full=True, data_range=D_gt.max() - D_gt.min())
        ssim2, _ = structural_similarity(D_pred2, D_gt, full=True, data_range=D_gt.max() - D_gt.min())
        # print(frame_id, 'ssim', ssim0, ssim1, ssim2)

        val.append(
            (rmse0, rmse1, rmse2, ssim0, ssim1, ssim2)
        )

    val = np.array(val).mean(axis=0)
    results.append(
        (capture_dir, capture_id, *val)
    )

results = pd.DataFrame(results, columns=[
    'capture_dir', 'capture_id', 'rmse0', 'rmse1', 'rmse2', 'ssim0', 'ssim1', 'ssim2'
])
results.to_csv('align_eval.csv', index=False)
print(results)
