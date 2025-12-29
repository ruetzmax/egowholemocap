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

def main(joint_pkl_path, image_id, object_pkl_path=None):
    with open(joint_pkl_path, 'rb') as f:
        joint_data = pickle.load(f)
        
    if object_pkl_path:
        with open(object_pkl_path, 'rb') as f:
            object_data = pickle.load(f)
        
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    res_dir = os.path.join(os.path.dirname(joint_pkl_path), 'diffusion_res')
    os.makedirs(res_dir, exist_ok=True)

    for image_id in range(len(joint_data)):
        # draw joints
        pred_left_hand = joint_data[image_id]['left_hand_pred_motion']
        pred_right_hand = joint_data[image_id]['right_hand_pred_motion']
        pred_body_pose = joint_data[image_id]['mo2cap2_pred_motion']

        pred_right_hand += pred_body_pose[3] - pred_right_hand[0]
        pred_left_hand += pred_body_pose[6] - pred_left_hand[0]

        body_mesh = draw_skeleton_with_chain(pred_body_pose, mo2cap2_chain)
        left_hand_mesh = draw_skeleton_with_chain(pred_left_hand, mano_skeleton, keypoint_radius=0.01,
                                                        line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(pred_right_hand, mano_skeleton, keypoint_radius=0.01,
                                                        line_radius=0.0025)
        
        # draw object boxes
        if object_data and image_id < len(object_data):
            frame_data = object_data[image_id]
            
            object_centers = frame_data['pred_center_cam']
            object_dimensions = frame_data['pred_dimensions']
            object_rotations = frame_data['pred_pose']
            object_scores = frame_data['scores']

            
            # rotation_matrix = np.array([
            #     [1, 0, 0],
            #     [0, 0, -1],
            #     [0, 1, 0]
            # ])
            
            # object_centers = object_centers @ rotation_matrix.T

            
            object_boxes = []
            for object_idx in range(len(object_centers)):
                if object_scores[object_idx] < 0.3:
                    continue
                obj = open3d.geometry.OrientedBoundingBox(object_centers[object_idx], object_rotations[object_idx], object_dimensions[object_idx])
                obj = open3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obj)
                obj.paint_uniform_color([0.0, 0.0, 1.0])
                object_boxes.append(obj)
            
        
        vis.clear_geometries()
        
        vis.add_geometry(body_mesh, reset_bounding_box=image_id==0) 
        vis.add_geometry(left_hand_mesh, reset_bounding_box=image_id==0)
        vis.add_geometry(right_hand_mesh, reset_bounding_box=image_id==0)
        for object_box in object_boxes:
            vis.add_geometry(object_box, reset_bounding_box=image_id==0)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(res_dir, f'vis_frame_{image_id}.png'))
        
        time.sleep(0.05)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='visualize single frame whole body result')
    parser.add_argument('--pred_path', type=str, required=True, help='prediction output pkl file path')
    parser.add_argument('--image_id', type=int, required=True, help='the image id to visualize')
    parser.add_argument('--object_pred_path', type=str, default=None, help="ovmono3d predictions pkl file path")
    args = parser.parse_args()

    main(args.pred_path, args.image_id, args.object_pred_path)
