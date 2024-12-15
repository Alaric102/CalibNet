from src.dataset import KittiOdometryDataset
from src.utils import transform_cloud, project_cloud
from src.visualization import plot_projection

import cv2 as cv
import numpy as np

BASE_PATH = "/home/evouser/Workspace/Datasets/KITTI/"
VISUALOZATION_STEP = 10
VISUALOZATION_DELAY_MS = 100
VISUALOZATION_STOP_KEY = 'q'

if __name__ == "__main__":
    config = {
        'base_path' : BASE_PATH,
    }
    dataset = KittiOdometryDataset(config)

    cv.namedWindow("Projection", cv.WINDOW_AUTOSIZE)
    print(f"Press {VISUALOZATION_STOP_KEY} to stop.")

    for id in range(len(dataset) // VISUALOZATION_STEP):
        data = dataset[VISUALOZATION_STEP*id]
        cloud = data['cloud']
        image = np.array(data['rgb'])
        extrinsic = data['extrinsic']
        intrinsic = data['intrinsic']
        
        cloud_camera = transform_cloud(cloud, extrinsic)
        cloud_projection = project_cloud(cloud_camera, intrinsic, image.shape)
        projection = plot_projection(image, cloud_projection)
        cv.imshow('Projection', projection)
        if (cv.waitKey(VISUALOZATION_DELAY_MS) == ord(VISUALOZATION_STOP_KEY)):
            break
    cv.destroyAllWindows() 