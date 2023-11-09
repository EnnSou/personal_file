#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h> /* for close() for socket */

#include <iostream>
#include <memory>
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

int main() {
  int client_socket;
  struct sockaddr_in sa;
  socklen_t len;
  size_t buffer_length = sizeof(udp_msg);
  size_t msg_length = sizeof(M3D_MQTT_FRAME_RESULT_V0_8_7_1);
  int bytes_sent;
  client_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (client_socket < 0) /* if socket failed to initialize, exit */
  {
    printf("Error Creating Socket");
    exit(EXIT_FAILURE);
  }

  const auto blk_size = YUV_BLK_SIZE;
  memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_port = htons(kSimUdpPort);
  sa.sin_addr.s_addr = inet_addr("127.0.0.1");
  int res = connect(client_socket, (struct sockaddr *)&sa, sizeof(sockaddr_in));
  if (res < 0) {
    perror("connect failed");
    close(client_socket);
    exit(EXIT_FAILURE);
  }
  int send_num = 0;
  int loop_num = 10;
#if 0
  while (send_num <= loop_num) {
    std::shared_ptr<M3D_MQTT_FRAME_RESULT_V0_8_7_1> frame_result =
        std::make_shared<M3D_MQTT_FRAME_RESULT_V0_8_7_1>();
    frame_result->FrameNum = send_num;
    int i = 0;
    for (; i < msg_length / blk_size; ++i) {
      udp_msg buffer;
      buffer.sequence = i;
      buffer.length = YUV_BLK_SIZE;
      memcpy(buffer.data, (char *)frame_result.get() + (i * blk_size),
             blk_size);
      bytes_sent = send(client_socket, &buffer, buffer_length, 0);
      if (bytes_sent < 0) printf("Error sending packet: %s\n", strerror(errno));

      // std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    if (msg_length % blk_size != 0) {
      udp_msg buffer;
      int remaining_length = msg_length % blk_size;
      buffer.sequence = i;
      buffer.length = remaining_length;
      memcpy(buffer.data, (char *)frame_result.get() + (i * blk_size),
             remaining_length);
      bytes_sent = send(client_socket, &buffer, buffer_length, 0);
      if (bytes_sent < 0) printf("Error sending packet: %s\n", strerror(errno));
    }
    std::cout << "send_num " << send_num << " completed" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    send_num++;
  }
#endif
  while (send_num < loop_num) {
    std::shared_ptr<M3D_MQTT_FRAME_RESULT_V0_8_7_1> frame_result =
        std::make_shared<M3D_MQTT_FRAME_RESULT_V0_8_7_1>();
    udp_msg buffer;
    frame_result->FrameNum = send_num;
    buffer.length = sizeof(frame_result);
    memcpy(buffer.data, (char *)frame_result.get(), sizeof(frame_result));
    bytes_sent = send(client_socket, &buffer, buffer_length, 0);
    if (bytes_sent < 0) printf("Error sending packet: %s\n", strerror(errno));
    std::cout << "send_num " << send_num << " completed" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    send_num++;
  }
  close(client_socket);
  std::cout << "=========================" << std::endl;

  return 0;
}
