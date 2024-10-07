import socket
import pickle
import struct
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ServerAPI:
    """
    ServerAPI handles a single client connection and facilitates bidirectional communication.
    """

    def __init__(self, host='0.0.0.0', port=8888):
        """
        Initializes the server with the specified host and port.
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_addr = None

    def start_server(self):
        """
        Starts the server and waits for a client connection.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        logging.info(f'Server listening on {self.host}:{self.port}')

        # Accept a single connection
        self.client_socket, self.client_addr = self.server_socket.accept()
        logging.info(f'Connected by {self.client_addr}')

    def stop_server(self):
        """
        Stops the server and closes the client connection.
        """
        if self.client_socket:
            self.client_socket.close()
            logging.info('Client socket closed.')
        if self.server_socket:
            self.server_socket.close()
            logging.info('Server socket closed.')

    def send(self, message_dict):
        """
        Sends a dictionary to the connected client.
        This method blocks until the message is sent.
        """
        try:
            serialized_data = pickle.dumps(message_dict)
            data_length = struct.pack('>I', len(serialized_data))
            message = data_length + serialized_data
            self.client_socket.sendall(message)
            logging.info(f"Sent data to {self.client_addr}")
        except Exception as e:
            logging.error(f"Failed to send data to {self.client_addr}: {e}")
            self.stop_server()

    def receive(self):
        """
        Receives a dictionary from the connected client.
        This method blocks until the data is received.
        """
        try:
            # First, receive the length of the incoming data (4 bytes)
            raw_length = self.receive_all(4)
            if not raw_length:
                logging.info(f"No data received. Closing connection with {self.client_addr}.")
                self.stop_server()
                return None
            data_length = struct.unpack('>I', raw_length)[0]
            logging.info(f"Expecting to receive {data_length} bytes of data from {self.client_addr}")

            # Now receive the actual data
            serialized_data = self.receive_all(data_length)
            if not serialized_data:
                logging.info(f"No serialized data received. Closing connection with {self.client_addr}.")
                self.stop_server()
                return None

            # Deserialize the data
            data_dict = pickle.loads(serialized_data)
            logging.info(f"Received data from {self.client_addr}: {data_dict}")

            return data_dict
        except Exception as e:
            logging.error(f"Error receiving data from {self.client_addr}: {e}")
            self.stop_server()
            return None

    def receive_all(self, length):
        """
        Helper function to receive exactly 'length' bytes from the socket.
        """
        data = b''
        while len(data) < length:
            more = self.client_socket.recv(length - len(data))
            if not more:
                raise EOFError('Socket closed before receiving all data')
            data += more
        return data

    def generate_data(self):
        """
        Generates a sample dictionary containing NumPy arrays.
        Modify this method to send your actual data.
        """
        data_dict = {
            'server_array1': np.random.rand(5),
            'server_array2': np.random.randint(0, 100, size=(3, 3)),
            'server_array3': np.linspace(0, 1, 10)
        }
        return data_dict

def main():
    server = ServerAPI(host='0.0.0.0', port=8888)
    server.start_server()

    try:
        while True:
            # Receive data from client
            data = server.receive()
            if data is None:
                break

            # Process the received data as needed
            # For example, print it
            logging.info(f"Processing data from client: {data}")

            # Send data to client
            data_to_send = server.generate_data()
            server.send(data_to_send)

    except KeyboardInterrupt:
        logging.info('Server shutting down.')
    finally:
        server.stop_server()

if __name__ == '__main__':
    main()
