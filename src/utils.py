import numpy as np

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

def transform_cloud(cloud, extrinsic):
    rotation_mat = extrinsic[:3, :3]
    translation = extrinsic[:3, 3:]
    xyz = cloud[:, :3]
    xyz = xyz @ rotation_mat.T + translation.T
    intensity = cloud[:, 3:]
    return np.hstack([xyz, intensity])

def project_cloud(cloud, projection, image_shape):
    rotation_mat = projection[:3, :3]
    translation = projection[:3, 3:]
    xyz = cloud[:, :3]
    xyz = xyz @ rotation_mat.T + translation.T
    
    depth_inliers = xyz[:, 2] > 0
    depth = xyz[depth_inliers, 2:]
    uv = xyz[depth_inliers, :2] / depth
    image_inliers = (uv[:, 0] > 0) & (uv[:, 0] < image_shape[1]) & (uv[:, 1] > 0) & (uv[:, 1] < image_shape[0])
    intensity = cloud[depth_inliers][image_inliers, 3:]
    return np.hstack([uv[image_inliers], depth[image_inliers], intensity])

DEG2RAD = np.pi / 180.0
RAD2DEG = 189.0 / np.pi

def random_rotation_matrix_from_euler(max_roll: float, max_pitch: float, max_yaw: float):
    roll = np.random.uniform(-max_roll * DEG2RAD, max_roll * DEG2RAD)
    pitch = np.random.uniform(-max_pitch * DEG2RAD, max_pitch * DEG2RAD)
    yaw = np.random.uniform(-max_yaw * DEG2RAD, max_yaw * DEG2RAD)
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def random_translation(max_x: float, max_y: float, max_z: float):
    x = np.random.uniform(-max_x, max_x)
    y = np.random.uniform(-max_y, max_y)
    z = np.random.uniform(-max_z, max_z)
    return np.array([x, y, z])

def random_transform(max_roll: float,  max_pitch: float, max_yaw: float,
                     max_x: float, max_y: float, max_z: float):
    transform = np.eye(4)
    transform[:3, :3] = random_rotation_matrix_from_euler(max_roll, max_pitch, max_yaw)
    transform[:3, 3] = random_translation(max_x, max_y, max_z)
    return transform