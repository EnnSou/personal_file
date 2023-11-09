import socket

def start_udp_server(ip, port):
    # 创建一个UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定到指定的ip和端口
    s.bind((ip, port))
    print(f"UDP server started on {ip}:{port}, waiting for data...")
    
    buffer = b""
    cnt = 0
    while True:
        # 接收数据以及客户端的地址和端口
        data, addr = s.recvfrom(1450)
        # print(f"Received message:")
        
        buffer += data
        print("data len: ", len(data))
        if len(data) < 1450:
            # 当收到的数据小于1450时，认为数据接收完毕

            print("Received message compeled {}".format(cnt))
            cnt += 1
            with open('output_idl.bin', 'wb') as f:
                f.write(buffer)
            buffer = b""
        

# 运行服务器，监听特定的IP和端口
start_udp_server('', 5004)
