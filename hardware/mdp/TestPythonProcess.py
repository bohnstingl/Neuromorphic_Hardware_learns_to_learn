import sys
import time
import socket
import pickle
import NetworkMDP

if __name__ == '__main__':
            

    nStates = 5
    nActions = 4
    nInhibit = 10
    maxIteration = 1999
    useCalibration = False

    print 'Starting dummy process'

    HOST = ''                 # Symbolic name meaning all available interfaces
    PORT = 25555              # Arbitrary non-privileged port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    
    while True:

        try:
            conn, addr = s.accept()
            print 'Connected by', addr
            
            f = s.makefile('rb', 1024)
            data = pickle.load(f)
            f.close()

            if not data:
                conn.close()
                continue

            #Check the incomming data
            data = data.decode()

            if data == 'exit':
                conn.close()
                break

            if '=' not in data:
                conn.sendall('Error: Not a valid command!')
                continue
            
            #parameters = [(str(t.split('=')[0]), str(t.split('=')[1])) for t in data.split('#')]
            print data

            #Execute the Network and return the fitness function
            gamma = 0.95 # discount factor
            lam = 0.12
            eta = 0.012

            network = DLSNetwork(nStates, nActions, useCalibration)
            
            network.CreateNetwork(nInhibit)
            
            network.Upload(gamma, lam, eta, maxIteration)

            network.EvaluateNetwork(maxIteration)

            #Handle the data and respond to the command
            conn.sendall(1234)
        except:
            conn.sendall('Error: Error during execution!')

        finally:
            conn.close()
