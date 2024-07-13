import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
import os

def capture_and_save(pipeline, save_dir, image_counter, hfov, vfov, depth_scale):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return image_counter

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 중앙 점 좌표
    height, width = depth_image.shape
    cx = width // 2
    cy = height // 2

    # 깊이 값 얻기
    depth_in_meters = depth_image[cy, cx] * depth_scale

    # 깊이 이미지를 컬러맵으로 변환하여 시각화
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # 원 그리기 (시각화 용도로만)
    color_image_with_circle = color_image.copy()
    depth_image_with_circle = depth_colormap.copy()
    cv2.circle(color_image_with_circle, (cx, cy), 5, (0, 0, 255), -1)  # 중앙에 빨간색 점
    cv2.circle(depth_image_with_circle, (cx, cy), 5, (255, 255, 255), -1)  # 중앙에 흰색 점

    cv2.imshow('Color Image', color_image_with_circle)
    cv2.imshow('Depth Image', depth_image_with_circle)

    # 이미지와 메타데이터 저장
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # 파일 이름 생성
        color_image_filename = os.path.join(save_dir, f'color_image_{image_counter:04d}.png')
        depth_image_filename = os.path.join(save_dir, f'depth_image_{image_counter:04d}.npy')
        depth_image_visual_filename = os.path.join(save_dir, f'depth_image_visual_{image_counter:04d}.png')

        # 컬러 이미지 저장
        color_image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        color_image_pil.save(color_image_filename)

        # 깊이 이미지를 numpy 배열로 저장
        np.save(depth_image_filename, depth_image)

        # 깊이 이미지를 시각화하여 저장
        depth_image_pil = Image.fromarray(cv2.convertScaleAbs(depth_image, alpha=0.03))
        depth_image_pil.save(depth_image_visual_filename)

        print(f"Saved {color_image_filename}, {depth_image_filename}, and {depth_image_visual_filename}")
        image_counter += 1

    return image_counter

def main():
    # 카메라 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Depth scale 얻기
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print(f"Depth Scale: {depth_scale}")

    # 카메라의 FOV 설정 (RealSense D455의 경우)
    hfov = 87  # 수평 FOV
    vfov = 58  # 수직 FOV

    # 결과 저장 디렉토리 설정
    save_dir_base = 'captured_images'
    os.makedirs(save_dir_base, exist_ok=True)

    # 저장할 이미지의 초기 번호 설정
    run_counter = 0

    # 현재 디렉토리 내 폴더 이름을 확인하여 가장 높은 번호를 가져옴
    existing_folders = [f for f in os.listdir(save_dir_base) if os.path.isdir(os.path.join(save_dir_base, f))]
    if existing_folders:
        existing_folders.sort()
        last_folder = existing_folders[-1]
        run_counter = int(last_folder.split('_')[-1]) + 1

    image_counter = 0

    try:
        while True:
            save_dir = os.path.join(save_dir_base, f'results_{run_counter:05d}')
            os.makedirs(save_dir, exist_ok=True)
            image_counter = capture_and_save(pipeline, save_dir, image_counter, hfov, vfov, depth_scale)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
