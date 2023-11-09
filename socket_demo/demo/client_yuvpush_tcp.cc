#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#define kSimUdpPort (5012)
#define YUV_BLK_SIZE (1450) // 67200 68600

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
  int clientSocket;
  struct sockaddr_in servaddr;
  size_t datalen;
  socklen_t len;

  const auto blk_size = YUV_BLK_SIZE;

  const size_t yuv_size = width * height * 1.5;
  std::vector<uint8_t> yuv;
  yuv.resize(yuv_size);
  ReadYuvData({"/home/zheng/work/repo/amba-replay/udp_test/data/"
               "83e1e804-b1bf-4f77-bd25-646a1622f4bb/front_standard_60/yuv_bin/"
               "00090539-1698835671091750952/c_y.bin",
               "/home/zheng/work/repo/amba-replay/udp_test/data/"
               "83e1e804-b1bf-4f77-bd25-646a1622f4bb/front_standard_60/yuv_bin/"
               "00090539-1698835671091750952/c_uv.bin"},
              &yuv);

  uint8_t* yuv_data2 = yuv.data();

  clientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (clientSocket < 0) {
    printf("Error Creating Socket");
    exit(EXIT_FAILURE);
  }
  unsigned long snd_size = 1024 * 1024;
  socklen_t optlen = sizeof(snd_size);
  int ret = setsockopt(clientSocket, SOL_SOCKET, SO_SNDBUF, &snd_size, optlen);
  if (ret) {
    std::cout << "setsockopt error!" << std::endl;
  }

  ret = getsockopt(clientSocket, SOL_SOCKET, SO_SNDBUF, &snd_size, &optlen);
  if (ret) {
    std::cout << " get sock opt error " << std::endl;
  } else {
    std::cout << "sock cache size : " << snd_size << std::endl;
  }

  bzero(&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  const char* ip = "192.168.42.100";
  // const char* ip = "127.0.0.1";
  servaddr.sin_addr.s_addr = inet_addr(ip);
  servaddr.sin_port = htons(kSimUdpPort);
  std::cout << "ip: " << ip << ", port: " << kSimUdpPort << std::endl;
  int res =
      connect(clientSocket, (struct sockaddr*)&servaddr, sizeof(sockaddr_in));
  if (res < 0) {
    perror("connect failed");
    close(clientSocket);
    exit(EXIT_FAILURE);
  }
  std::cout << "connectd, starting send data" << std::endl;
  int cnt = 0;
  int loog_num = 10;
  while (cnt < loog_num) {
    uint8_t* yuv_data;

    cnt++;
    yuv_data = yuv_data2;
    uint16_t i = 0;
    for (; i < yuv_size / blk_size; ++i) {
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
      data.frame_number = cnt;
      data.captrue_time = 12345678901234567890ULL;
      memcpy(data.data, &yuv_data[i * blk_size], blk_size);
      if (send(clientSocket, &data, datalen, 0) < 0) {
        printf("send error\n");
      } else {
        printf("send sucess, seq_num: %d\n", i);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50000));
    }

    if (yuv_size % blk_size != 0) {
      ReqData data;
      data.header = 0x0001;
      uint16_t remaining_size = yuv_size % blk_size;
      data.seq_num = i;
      data.fov = 0x1;
      data.length = remaining_size;
      data.frame_number = cnt;
      data.captrue_time = 12345678901234567890ULL;
      datalen = sizeof(data);
      memcpy(data.data, &yuv_data[yuv_size - remaining_size], remaining_size);
      if (send(clientSocket, &data, datalen, 0) < 0) {
        printf("send error\n");
      } else {
        printf("send sucess, seq_num: %d\n", i);
      }
    }
    std::cout << "send frame " << cnt << " data finish" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }
  close(clientSocket);
  return 0;
}
