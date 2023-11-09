#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#define MAXLINE (1450)
#define kSimUdpPort (5012)
#define YUV_BLK_SIZE (1400)

constexpr int width = 960;
constexpr int height = 540;
// constexpr int width = 680;
// constexpr int height = 382;

#pragma pack(push, 1)
struct ReqData {
  uint8_t header;
  uint8_t fov;
  uint16_t seq_num;
  uint32_t length;
  uint32_t frame_number;
  uint64_t captrue_time;
  char data[YUV_BLK_SIZE];
};
#pragma pack(pop)

bool ReadYuvData(const std::vector<std::string>& path_list,
                 std::vector<uint8_t>* data) {
  if (nullptr == data) {
    return false;
  }

  uint32_t offset = 0;

  for (const auto& element : path_list) {
    std::ifstream file(element.c_str(), std::ios::binary);

    if (!file) {
      std::cout << "can't open file: " << element.c_str() << std::endl;
      return false;
    }

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    file.read(reinterpret_cast<char*>(data->data() + offset), file_size);
    offset += file_size;

    if (!file) {
      std::cout << "read file fail" << std::endl;
      return false;
    }
  }

  if (offset != data->size()) {
    return false;
  }
  std::cout << "data_size: " << data->size() << std::endl;

  return true;
}

int main() {
  int sockfd;
  struct sockaddr_in servaddr;
  size_t datalen;
  socklen_t len;

  const auto blk_size = YUV_BLK_SIZE;

  const size_t yuv_size = width * height * 1.5;
  std::vector<uint8_t> yuv;
  yuv.resize(yuv_size);
  ReadYuvData({"/home/zheng/work/repo/amba-replay/udp_test/data/83e1e804-b1bf-4f77-bd25-646a1622f4bb/front_standard_60/yuv_bin/00090539-1698835671091750952/c_y.bin",
               "/home/zheng/work/repo/amba-replay/udp_test/data/83e1e804-b1bf-4f77-bd25-646a1622f4bb/front_standard_60/yuv_bin/00090539-1698835671091750952/c_uv.bin"},
              &yuv);

  // std::vector<uint8_t> yuv1;
  // yuv1.resize(yuv_size);
  // ReadYuvData({"./data/00171792-1635073319991082701_y.bin",
  //              "./data/00171792-1635073319991082701_uv.bin"},
  //             &yuv1);

  uint8_t* yuv_data2 = yuv.data();
  // uint8_t* yuv_data1 = yuv1.data();

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);

  bzero(&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  const char* ip = "192.168.42.100";
  servaddr.sin_addr.s_addr = inet_addr(ip);
  servaddr.sin_port = htons(kSimUdpPort);
  std::cout << "ip: " << ip << ", port: " << kSimUdpPort << std::endl;
  int cnt = 0;
  int loog_num = 10;
  while (loog_num > 0) {
    uint8_t* yuv_data;
    // if (cnt % 2 == 0) {
    //   yuv_data = yuv_data1;
    // } else {
    //   yuv_data = yuv_data2;
    // }
    yuv_data = yuv_data2;
    cnt++;
    for (uint16_t i = 0; i < yuv_size / blk_size; ++i) {
      ReqData data;
      if (i == 0) {
        data.header = 0x00FF;
      } else {
        data.header = 0x0001;
      }
      datalen = sizeof(data);
      data.fov = 0x1;
      data.seq_num = i;
      data.length = blk_size;
      data.frame_number = i;
      data.captrue_time = 12345678901234567890ULL;
      memcpy(data.data, &yuv_data[i * blk_size], blk_size);
      if (sendto(sockfd, &data, datalen, 0, (struct sockaddr*)&servaddr,
                 sizeof(servaddr)) < 0) {
        printf("sendto error\n");
      } else {
        // printf("send sucess");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    if (yuv_size % blk_size != 0) {
      ReqData data;
      data.header = 0x0001;
      uint16_t remaining_size = yuv_size % blk_size;
      data.seq_num = yuv_size / blk_size;
      data.fov = 0x1;
      data.length = remaining_size;
      data.frame_number = 3333;
      data.captrue_time = 12345678901234567890ULL;
      datalen = sizeof(data);
      memcpy(data.data, &yuv_data[yuv_size - remaining_size], remaining_size);
      if (sendto(sockfd, &data, datalen, 0, (struct sockaddr*)&servaddr,
                 sizeof(servaddr)) < 0) {
        printf("sendto error\n");
      } else {
        std::cout << "sucessed " << cnt << std::endl;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    loog_num -- ;
  }

  return 0;
}
