import numpy as np
import cv2
import os

def calculate_angles_and_display(depth_image_filename, px, py, hfov, vfov):
    depth_image = np.load(depth_image_filename)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # cv2.imshow('Depth Image', depth_colormap)

    # 특정 좌표의 depth 값 확인
    height, width = depth_image.shape
    cx = width // 2
    cy = height // 2
    depth_value_center = depth_image[cy, cx]

    # 특정 픽셀 좌표의 depth 값 확인
    depth_value_pixel = depth_image[py, px]

    # 수평 및 수직 각도 계산
    angle_x = (px - cx) / (width / 2) * (hfov / 2)
    angle_y = (py - cy) / (height / 2) * (vfov / 2)

    print(f'Depth value at center (cx, cy): {depth_value_center}')
    print(f'Depth value at pixel (px, py): {depth_value_pixel}')
    print(f'Horizontal angle to pixel (px, py): {angle_x} degrees')
    print(f'Vertical angle to pixel (px, py): {angle_y} degrees')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    hfov = 87  # 수평 FOV
    vfov = 58  # 수직 FOV

    depth_image_filename = 'captured_images/results_00000/depth_image_0000.npy'

    px = 320
    py = 240

    calculate_angles_and_display(depth_image_filename, px, py, hfov, vfov)

if __name__ == "__main__":
    main()
