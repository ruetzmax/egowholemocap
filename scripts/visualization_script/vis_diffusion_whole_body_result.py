#  Copyright Jian Wang @ MPI-INF (c) 2023.

import os
import pickle
from copy import deepcopy
import time

import numpy as np
import open3d
import torch
from mmpose.utils.visualization.draw import draw_skeleton_with_chain
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.data.keypoints_mapping.mano import mano_skeleton

def main(pkl_path, image_id):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    res_dir = os.path.join(os.path.dirname(pkl_path), 'diffusion_res')
    os.makedirs(res_dir, exist_ok=True)

    for image_id in range(len(data)):
        pred_left_hand = data[image_id]['left_hand_pred_motion']
        pred_right_hand = data[image_id]['right_hand_pred_motion']
        pred_body_pose = data[image_id]['mo2cap2_pred_motion']

        pred_right_hand += pred_body_pose[3] - pred_right_hand[0]
        pred_left_hand += pred_body_pose[6] - pred_left_hand[0]

        body_mesh = draw_skeleton_with_chain(pred_body_pose, mo2cap2_chain)
        left_hand_mesh = draw_skeleton_with_chain(pred_left_hand, mano_skeleton, keypoint_radius=0.01,
                                                        line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(pred_right_hand, mano_skeleton, keypoint_radius=0.01,
                                                        line_radius=0.0025)
        
        vis.clear_geometries()
        
        vis.add_geometry(body_mesh, reset_bounding_box=image_id==0) 
        vis.add_geometry(left_hand_mesh, reset_bounding_box=image_id==0)
        vis.add_geometry(right_hand_mesh, reset_bounding_box=image_id==0)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(res_dir, f'vis_frame_{image_id}.png'))
        
        time.sleep(0.05)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='visualize single frame whole body result')
    parser.add_argument('--pred_path', type=str, required=True, help='prediction output pkl file path')
    parser.add_argument('--image_id', type=int, required=True, help='the image id to visualize')
    args = parser.parse_args()

    main(args.pred_path, args.image_id)
