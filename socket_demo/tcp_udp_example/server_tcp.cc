
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
#define MAXLINE (1450)
#define kSimUdpPort (5024)
#define YUV_BLK_SIZE (1400)

struct udp_msg {
  int sequence = 0;
  int length = 0;
  char data[YUV_BLK_SIZE];
  bool close = false;
  bool server_close = false;
};

bool flag = true;


int main() {
  int listenSocket, connectSocket;
  struct sockaddr_in sock_addr, newAddr;
  socklen_t addrSize;
  char buffer[sizeof(udp_msg)];
  udp_msg *p_request = (udp_msg *)buffer;
  ssize_t recsize;

  listenSocket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listenSocket < 0) {
    std::cerr << "Error in socket" << std::endl;
    exit(EXIT_FAILURE);
  }

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

  int all_num = sizeof(M3D_MQTT_FRAME_RESULT_V0_8_7_1) / YUV_BLK_SIZE + 1;
  while (flag) {

    addrSize = sizeof(connectSocket);
    std::cout << "waitting connect ..." << std::endl;
    connectSocket = accept(listenSocket, (struct sockaddr *)&newAddr, &addrSize);
    if (connectSocket < 0) {
      perror("error accept failed");
      close(connectSocket);
      continue;
    }

    int expect_num = 0;
    char frame_result[sizeof(M3D_MQTT_FRAME_RESULT_V0_8_7_1)];
    memset(buffer, 0, sizeof(buffer));
    #if 0
    while (recv(connectSocket, buffer, sizeof(udp_msg), 0) > 0) {
      std::cout << "expect_num: " << expect_num << std::endl;
      std::cout << "data_sequence: " << p_request->sequence << std::endl;
      memcpy(&frame_result[expect_num * YUV_BLK_SIZE], p_request->data,
             p_request->length);
      expect_num++;
      if (expect_num == all_num) {
        expect_num = 0;
        M3D_MQTT_FRAME_RESULT_V0_8_7_1 *p_result =
            (M3D_MQTT_FRAME_RESULT_V0_8_7_1 *)frame_result;
        std::cout << "FrameNum: " << p_result->FrameNum << std::endl;
      }
    }
    #endif
    std::cout << "********** connected **********" << std::endl;
    while(recv(connectSocket, buffer, sizeof(udp_msg), 0) > 0){

        M3D_MQTT_FRAME_RESULT_V0_8_7_1 *p_result =
            (M3D_MQTT_FRAME_RESULT_V0_8_7_1 *)frame_result;
        memcpy(&frame_result, p_request->data, p_request->length);
        std::cout << "FrameNum: " << p_result->FrameNum << std::endl;
    }
    close(connectSocket);
  }
  std::cout << " close listen socket" << std::endl;
  close(listenSocket);
  return 0;
}