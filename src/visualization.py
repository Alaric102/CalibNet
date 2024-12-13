import cv2 as cv

def plot_projection(image, cloud, cmap='depth', radius=1):
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    for point in cloud:
        u, v, depth, intensity = point
        if cmap == 'depth':
            color = (int(depth), 255, 255)
        elif cmap == 'intensity':
            color = (int(intensity * 255), 255, 255)
        else:
            raise ValueError(f"Unexpected color map. Possible values: 'depth', 'intensity'.")
        cv.circle(hsv_img, (int(u), int(v)), radius=radius, color=color, thickness=-1)
    return cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)