#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "./mqtt_idl_0_8_7_1.h"
#define SVC_YUV_UDP_BLK_SIZE (67200)
#define MAXLINE (1450)
#define kSimUdpPort (5012)
#define SIM_YUV_MAX_HEIGHT (540u)
#define SIM_YUV_MAX_WIDTH (960u)
#define SIM_YUV_MAX_PITCH (1024u)
#define SIM_YUV_MAX_UDP_DATA_SIZE \
  (SIM_YUV_MAX_HEIGHT * SIM_YUV_MAX_WIDTH * 3 / 2)

static const uint32_t kYuvFrontHeight = 540u;
static const uint32_t kYuvFrontWidth = 960u;
static const uint32_t kYuvFrontPitch = 960u;
static const uint32_t kYuvRearHeight = 382u;
static const uint32_t kYuvRearWidth = 680u;
static const uint32_t kYuvRearPitch = 704u;
static const uint8_t kYuvFovFrontStandard = 1u;
static const uint8_t kYuvFovFrontTelephoto = 3u;
static const uint8_t kYuvFovLeftRear = 4u;
static const uint8_t kYuvFovRightRear = 6u;
static const uint8_t kYuvUdpPkgHeader = 0xff;

struct ReqData {
  uint8_t header;
  uint8_t fov;
  uint16_t seq_num;
  uint32_t length;
  uint32_t frame_number;
  uint64_t captrue_time;
  char data[SVC_YUV_UDP_BLK_SIZE];
};

uint32_t YuvTotalSizeCacl(const uint32_t height, const uint32_t width) {
  return height * width * 3 >> 1;
}

int main() {
  bool udp_data_valid;
  int data_len;
  int ret;
  int listenSocket, connectSocket;
  struct sockaddr_in sock_addr;
  struct sockaddr newaddr;
  socklen_t addr_len;
  uint32_t addrSize;
  uint16_t expected_seq_num = 0u;
  uint16_t blk_num = 0u;

  char buffer[sizeof(ReqData)];
  ReqData *p_request = (ReqData *)buffer;
  const uint16_t front_blk_num =
      (uint16_t)(YuvTotalSizeCacl(kYuvFrontHeight, kYuvFrontWidth) /
                 SVC_YUV_UDP_BLK_SIZE) +
      1u;
  const uint16_t rear_blk_num =
      (uint16_t)(YuvTotalSizeCacl(kYuvRearHeight, kYuvRearWidth) /
                 SVC_YUV_UDP_BLK_SIZE) +
      1u;

  listenSocket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listenSocket < 0) {
    std::cerr << "Error in socket" << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned long rcv_size = 1024 * 1024;
  socklen_t optlen = sizeof(rcv_size);

  ret = setsockopt(listenSocket, SOL_SOCKET, SO_RCVBUF, &rcv_size, optlen);
  if (ret) {
    std::cout << "setsockopt error!" << std::endl;
  }

  ret = getsockopt(listenSocket, SOL_SOCKET, SO_RCVBUF, &rcv_size, &optlen);
  if (ret) {
    std::cout << " get sock opt error " << std::endl;
  } else {
    std::cout << "sock cache size : " << rcv_size << std::endl;
  }

  memset(&sock_addr, 0, sizeof(sock_addr));
  sock_addr.sin_family = AF_INET;
  sock_addr.sin_port = htons(kSimUdpPort);
  sock_addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(listenSocket, (struct sockaddr *)&sock_addr, sizeof(sock_addr)) <
      0) {
    perror("error bind failed");
    close(listenSocket);
    exit(EXIT_FAILURE);
  }

  if (listen(listenSocket, 10) < 0) {
    perror("error listen failed");
    close(listenSocket);
    exit(EXIT_FAILURE);
  }

  addr_len = sizeof(connectSocket);
  while (true) {
    uint8_t temp_yuv_data[SIM_YUV_MAX_UDP_DATA_SIZE];
    uint32_t t1, t2;
    std::cout << "Waitting connect ...." << std::endl;
    connectSocket =
        accept(listenSocket, (struct sockaddr *)&newaddr, &addr_len);
    if (connectSocket < 0) {
      std::cerr << "accept() failed " << std::endl;
      close(connectSocket);
      continue;
    }
    std::cout << "********** connected **********" << std::endl;
    udp_data_valid = false;
    while (true) {
      data_len = recv(connectSocket, buffer, sizeof(ReqData), 0);
      if (data_len <= 0) {
        std::cout << "recv data_len: " << data_len << std::endl;
        break;
      }
      if (udp_data_valid == false && p_request->header != kYuvUdpPkgHeader) {
        continue;
      }
      if (p_request->header == kYuvUdpPkgHeader) {
        expected_seq_num = 0u;
        udp_data_valid = true;
        if (p_request->fov == kYuvFovFrontStandard ||
            p_request->fov == kYuvFovFrontTelephoto) {
          blk_num = front_blk_num;
        } else if (p_request->fov == kYuvFovLeftRear ||
                   p_request->fov == kYuvFovRightRear) {
          blk_num = rear_blk_num;
        } else {
          udp_data_valid = false;
          std::cout << " fov error " << p_request->fov << std::endl;
          break;
        }
        std::cout << " header recv() returned " << data_len << std::endl;
      }

      if (p_request->seq_num != expected_seq_num) {
        std::cout << " Package loss detected! Expecting package: "
                  << expected_seq_num
                  << ",but received package: " << p_request->seq_num
                  << " Data discarded " << std::endl;
        udp_data_valid = false;
        continue;
      } else {
        memcpy(&temp_yuv_data[expected_seq_num * SVC_YUV_UDP_BLK_SIZE],
               p_request->data, p_request->length);
        expected_seq_num++;
      }
      if (expected_seq_num == blk_num) {
        std::cout << " recevie complete yuv data, frame_num: "
                  << p_request->frame_number << std::endl;
        if (udp_data_valid == true) {
          // push data into queue
          std::cout << " push data into queue" << std::endl;
          expected_seq_num = 0;
          memset(temp_yuv_data, 0, sizeof(temp_yuv_data));
        }
      }
    }
    close(connectSocket);
  }
  close(listenSocket);
}