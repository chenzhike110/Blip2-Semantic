import torch
import os.path as osp
from PIL import Image
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from torch_geometric.data import Data
from pytorch3d.renderer import TexturesUV
from .transform import (
    matrix_to_rotation_6d,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    rotation_6d_to_matrix
)
"""
Get Joint Positions
"""
def joint_positions(positions, t, joints_name):
    # collect position
    position_t = positions.values.iloc[t, :]
    pos_X = {joint: None for joint in joints_name}
    pos_Y = {joint: None for joint in joints_name}
    pos_Z = {joint: None for joint in joints_name}
    for name, pos in position_t.items():
        joint, position = name.rsplit('_', 1)
        if joint not in joints_name:
            continue
        if position == 'Xposition':
            pos_X[joint] = pos
        elif position == 'Yposition':
            pos_Y[joint] = pos
        elif position == 'Zposition':
            pos_Z[joint] = pos
    # pos features
    pos = [None for _ in range(len(joints_name))]
    for joint in joints_name:
        try:
            pos[joints_name.index(joint)] = torch.Tensor([
                pos_X[joint],
                pos_Y[joint],
                pos_Z[joint]
            ])
        except:
            pos[joints_name.index(joint)] = torch.Tensor([0,0,0])
    pos = torch.stack(pos, dim=0)
    return pos

"""
Get Joint Angles
"""
def joint_angles(parsed_data, t, joints_name, rotation_mode='6d'):
    # collect joint angles
    rotation_t = parsed_data.values.iloc[t, :]
    joint_X = {joint: None for joint in joints_name}
    joint_Y = {joint: None for joint in joints_name}
    joint_Z = {joint: None for joint in joints_name}
    for name, rot in rotation_t.items():
        joint, rotation = name.rsplit('_', 1)
        if joint not in joints_name:
            continue
        if rotation == 'Xrotation':
            joint_X[joint] = rot*np.pi/180.0
        elif rotation == 'Yrotation':
            joint_Y[joint] = rot*np.pi/180.0
        else:
            joint_Z[joint] = rot*np.pi/180.0
    # joint features
    x = [None for _ in range(len(joints_name))]
    for joint in joints_name:
        euler_angles = torch.tensor([joint_X[joint], joint_Y[joint], joint_Z[joint]])
        if rotation_mode == '6d':
            rot_matrix = euler_angles_to_matrix(euler_angles, 'XYZ')
            x[joints_name.index(joint)] = matrix_to_rotation_6d(rot_matrix)
        else:
            x[joints_name.index(joint)] = euler_angles
    x = torch.stack(x, dim=0)
    return x

"""
Get Node Parent
"""
def node_parent(parsed_data, joints_name):
    parent = [joints_name.index(parsed_data.skeleton[joint]['parent']) if parsed_data.skeleton[joint]['parent'] in joints_name # root if no parent in joints name
              else -1 for joint in joints_name]
    return torch.LongTensor(parent)

"""
Get Offset
"""
def node_offset(parsed_data, joints_name):
    offset = torch.stack([torch.Tensor(parsed_data.skeleton[joint]['offsets']) for joint in joints_name], dim=0)
    return offset

"""
Get Joint name
"""
def get_topology_from_bvh(parsed_data):
    joints_name = []
    for joint in parsed_data.skeleton:
        if "_Nub" not in joint:
            joints_name.append(joint)
    return joints_name

"""
Parse BVH file
"""
def parse_bvh_to_frame(f, need_pose=True, skeleton_name=None, fbx_path=None, device=torch.device("cpu")):
    bvh_parser = BVHParser()
    parsed_data = bvh_parser.parse(f)
    if need_pose:
        mp = MocapParameterizer('position')
        positions = mp.fit_transform([parsed_data])[0] # list length 1
    else:
        positions = parsed_data

    if skeleton_name is None:
        skeleton_name = f.split('/')[-2]

    extern_data_path = osp.join(fbx_path, skeleton_name, "fbx")
    weight = np.load(osp.join(extern_data_path, "weights.npy"))
    verts = torch.from_numpy(np.load(osp.join(extern_data_path, "verts.npy"))).float().to(device)
    faces = torch.from_numpy(np.load(osp.join(extern_data_path, "faces.npy"))).to(device)
    with open(osp.join(extern_data_path, "labels.txt"), "r") as input:
        skinning_label = input.readlines()
        skinning_label = [name.split(':')[-1].strip() for name in skinning_label]
    uv = torch.from_numpy(np.load(osp.join(extern_data_path, "uv.npy"))).float().to(device)
    joints = np.load(osp.join(extern_data_path, "tjoints.npy"))
    with Image.open(osp.join(extern_data_path, "texture.png")) as image:
        np_image = torch.from_numpy(np.asarray(image.convert("RGB")).astype(np.float32) / 255.).to(device)
    texture = TexturesUV(maps=np_image[None], faces_uvs=faces[None], verts_uvs=uv[None])
    joints_all = get_topology_from_bvh(parsed_data)
    parent_all = node_parent(parsed_data, joints_all)
    offset_all = node_offset(parsed_data, joints_all)
    
    skeleton = Data(
        faces=faces,
        skinning_weights=weight,
        skinning_label=skinning_label,
        verts_origin=verts,
        joints_origin=joints,
        texture=texture,
        joints_all=joints_all,
        parent_all=parent_all,
        offset_all=offset_all,
    )

    data_list = []
    total_frames = parsed_data.values.shape[0]
    for t in range(0, total_frames):
        # joint features
        ang = joint_angles(parsed_data, t, joints_all)
        # pos features
        pos = joint_positions(positions, t, joints_all)
        # data all
        data = Data()
        data.ang = ang
        data.pos = pos
        data_list.append(data)
    
    return data_list, skeleton