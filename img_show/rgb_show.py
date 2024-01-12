import cv2
import numpy as np
import os

def read_rgb_binary_file(file_path, width, height):
    # 读取二进制文件数据
    with open(file_path, 'rb') as file:
        # 读取 RGB 数据（假设每个像素三个字节表示 RGB）
        rgb_data = np.frombuffer(file.read(width * height * 3), dtype=np.uint8)

    # 将一维数组转换为二维数组（图像数据）
    return rgb_data.reshape((3, height, width))

def display_rgb_binary_files(file_paths, start_index=0):
    current_index = start_index
    total_files = len(file_paths)
    print("total_files",total_files)
    width, height = 960, 540

    while True:
        # 读取当前 RGB 图像文件 
        file_path = file_paths[current_index]
        rgb_image = read_rgb_binary_file(file_path, width, height)

        # 显示图像
        # 根据实际rgb数据排布进行调整，[C, H, W]
        rgb_image = rgb_image.transpose((1,2,0))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('RGB Image', rgb_image)

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
    # rgb_file_path = "/home/zheng/work/repo/amba-replay/udp_test/demo/result/yz_out"
    rgb_file_path = "/media/zheng/D124-C503/yz_out"
    # 获取文件夹下所有二进制文件的路径
    binary_files = [os.path.join(rgb_file_path, file) for file in os.listdir(rgb_file_path)]
    binary_files.sort()
    # 显示 RGB 图像
    if binary_files:
        display_rgb_binary_files(binary_files)
    else:
        print('No binary files found in the specified folder.')