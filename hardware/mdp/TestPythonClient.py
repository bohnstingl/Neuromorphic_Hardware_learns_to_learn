import sys
import time
import socket

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception('Not exactly two arguments given!')
    
    HOST = 'localhost'        # The remote host
    PORT = 25555              # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(sys.argv[1].encode())
    data = s.recv(1024)
    s.close()
    print('Received' + data.decode())
