import cv2
import os
import numpy as np



def read_rgb_binary_file(file_path, width, height):
    # 读取二进制文件数据
    with open(file_path, 'rb') as file:
        rgb_data = np.frombuffer(file.read(width * height), dtype=np.uint8)
    return rgb_data.reshape((height, width))

def display_rgb_binary_files(file_paths, start_index=0):
    current_index = start_index
    total_files = len(file_paths)
    print("total_files",total_files)
    width, height = 1024, 540 

    while True:
        # 读取当前图像文件 
        file_path = file_paths[current_index]
        yuv_img = read_rgb_binary_file(file_path, width, height)

        # 显示图像
        cv2.imshow('yuv Image', yuv_img)

        # 等待用户按键，等待时间为 0.5 秒
        key = cv2.waitKey(500)

        # 如果按下 'q' 键，退出循环
        if key == ord('q'):
            break
        # 如果按下 'n' 键，切换到下一个文件
        elif key == ord('d'):
            if(current_index != total_files - 1):
                current_index = (current_index + 1) % total_files  
        # 如果按下 'p' 键，切换到上一个文件
        elif key == ord('a'):
            if(current_index != 0):
                current_index = (current_index - 1) % total_files

    # 释放窗口资源
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # yuv_file_path = "/home/zheng/work/repo/amba-replay/udp_test/data/00baaf18-43b9-42b5-a55d-c1756ad04115/front_standard_60/00004646-1625636551817057874_960/c_y.bin"
    yuv_file_path = "/media/zheng/D124-C503/rece"

    binary_files = [os.path.join(yuv_file_path, file) for file in os.listdir(yuv_file_path)]
    binary_files.sort()

    if binary_files:
        display_rgb_binary_files(binary_files)
    else:
        print('No binary files found in the specified folder.')






