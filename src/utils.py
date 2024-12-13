import numpy as np

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