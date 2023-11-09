
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "./mqtt_idl_0_8_7_1.h"
#define MAXLINE (1450)
#define kSimUdpPort (5023)
#define YUV_BLK_SIZE (1400)
struct udp_msg {
  int sequence = 0;
  int length = 0;
  char data[YUV_BLK_SIZE];
};

int main() {
  int sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
  struct sockaddr_in sock_addr;
  char buffer[sizeof(udp_msg)];
  udp_msg *p_request = (udp_msg *)buffer;
  ssize_t recsize;
  socklen_t fromlen;
  //   M3D_MQTT_FRAME_RESULT_V0_8_7_1 frame_result;

  memset(&sock_addr, 0, sizeof(sock_addr));
  sock_addr.sin_family = PF_INET;
  sock_addr.sin_addr.s_addr = INADDR_ANY;
  sock_addr.sin_port = htons(kSimUdpPort);

  if (-1 == bind(sock, (struct sockaddr *)&sock_addr, sizeof(sock_addr))) {
    perror("error bind failed");
    close(sock);
    exit(EXIT_FAILURE);
  }
  int expect_num = 0;
  int all_num = sizeof(M3D_MQTT_FRAME_RESULT_V0_8_7_1) / YUV_BLK_SIZE + 1;

  std::cout << "receive test..." << std::endl;
  for (;;) {
    char frame_result[sizeof(M3D_MQTT_FRAME_RESULT_V0_8_7_1)];
    while (1) {
      recsize = recvfrom(sock, (void *)buffer, sizeof(udp_msg), 0,
                         (struct sockaddr *)&sock_addr, &fromlen);
      if (recsize < 0) fprintf(stderr, "%s\n", strerror(errno));
      std::cout << "expect_num: " << expect_num << std::endl;
      std::cout << "data_sequence: " << p_request->sequence << std::endl;
      memcpy(&frame_result[expect_num * YUV_BLK_SIZE], p_request->data,
             p_request->length);
      expect_num++;
      if(expect_num == all_num){
        expect_num = 0;
        break;
      }
    }
    M3D_MQTT_FRAME_RESULT_V0_8_7_1 *p_result =
        (M3D_MQTT_FRAME_RESULT_V0_8_7_1 *)frame_result;
    std::cout << "FrameNum: " << p_result->FrameNum << std::endl;
    // printf("recsize: %d\n ", recsize);
    // sleep(1);
    // printf("MsgCode: %d\n", p_request->MsgCode);
    // // printf("ObjectLength: %d\n", p_request->ObjectLength);
    // std::cout << "ObjectLength: " << p_request->ObjectLength << std::endl;
  }
  return 0;
}