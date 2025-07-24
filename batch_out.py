import pathlib
from threading import Thread

import pandas as pd

from camera_utility import *
from examples.play_depth_video import colors, depths

obj_mapping = {
    'std_brick_object': 'mini_pc_std_obj_cascaded_dataset',
    'std_t_object': 'mini_pc_new_cascaded_dataset',
    'old_t_object': 'cascaded_dataset'
}


def run_align(capture_dir, capture_id):
    cam_dir = f'{base_dir}{capture_dir}/capture_{capture_id:05d}/'
    cols = colors(pathlib.Path(cam_dir))
    _, aligned = depths(pathlib.Path(cam_dir), cols)

    out_file_name = f'examples/{capture_dir}_{capture_id}.npy'
    np.save(out_file_name, aligned)


base_dir = '/home/ttoha12/weapon/'
data = pd.read_csv('overall_selection_9.csv')
total_rows = data.shape[0]

thread_list = []
for _, row in tqdm(data.iterrows(), total=total_rows, desc='save_raw_depth'):
    capture_dir, capture_id = row['Dataset'], row['Capture_ID']
    capture_dir = obj_mapping[capture_dir]

    t = Thread(target=run_align, args=(capture_dir, capture_id))
    t.start()
    thread_list.append(t)

    if len(thread_list) == 8:
        for t in thread_list:
            t.join()
        thread_list.clear()
