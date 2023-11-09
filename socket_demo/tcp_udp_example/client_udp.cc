#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h> /* for close() for socket */
#include <thread>
#include <iostream>
#include <memory>
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
  int sock;
  struct sockaddr_in sa;
  socklen_t len;
  size_t buffer_length = sizeof(udp_msg);
  size_t msg_length = sizeof(M3D_MQTT_FRAME_RESULT_V0_8_7_1);
  int bytes_sent;
  sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (-1 == sock) /* if socket failed to initialize, exit */
  {
    printf("Error Creating Socket");
    exit(EXIT_FAILURE);
  }

  const auto blk_size = YUV_BLK_SIZE;
  memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(0x7F000001);
  sa.sin_port = htons(kSimUdpPort);
  int send_num = 1;
  int loop_num = 10;
  std::cout << "buffer_length: " << msg_length << std::endl;
  while (send_num <= loop_num) {
    std::shared_ptr<M3D_MQTT_FRAME_RESULT_V0_8_7_1> frame_result =
        std::make_shared<M3D_MQTT_FRAME_RESULT_V0_8_7_1>();
    frame_result->FrameNum = send_num;
    int i = 0;
    for (; i < msg_length / blk_size; ++i) {
      udp_msg buffer;
      buffer.sequence = i;
      buffer.length = YUV_BLK_SIZE;
      memcpy(buffer.data, (char*)frame_result.get() + (i * blk_size), blk_size);
      bytes_sent = sendto(sock, &buffer, buffer_length, 0,
                          (struct sockaddr *)&sa, sizeof(struct sockaddr_in));
      if (bytes_sent < 0) printf("Error sending packet: %s\n", strerror(errno));
    }
    if (msg_length % blk_size != 0) {
      udp_msg buffer;
      int remaining_length = msg_length % blk_size;
      buffer.sequence = i;
      buffer.length = remaining_length;
      memcpy(buffer.data, (char*)frame_result.get() + (i * blk_size), remaining_length);
      bytes_sent = sendto(sock, &buffer, buffer_length, 0,
                          (struct sockaddr *)&sa, sizeof(struct sockaddr_in));
      if (bytes_sent < 0) printf("Error sending packet: %s\n", strerror(errno));
    }
    std::cout << "send_num " << send_num << " completed" << std::endl;

    send_num++;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  close(sock);
  return 0;
}
