import socket
import pickle
import struct
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClientAPI:
    """
    ClientAPI connects to the server, sends and receives serialized dictionaries containing NumPy arrays,
    and processes the received data.
    """

    def __init__(self, host='127.0.0.1', port=8888):
        """
        Initializes the client with the server's host and port.
        """
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """
        Connects to the server.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            logging.info(f'Connected to server at {self.host}:{self.port}')
        except Exception as e:
            logging.error(f'An error occurred while connecting: {e}')
            self.socket = None

    def disconnect(self):
        """
        Disconnects from the server and closes the socket.
        """
        if self.socket:
            self.socket.close()
            logging.info('Disconnected from server.')
            self.socket = None

    def send(self, message_dict):
        """
        Sends a dictionary containing NumPy arrays to the server.
        This method blocks until the data is sent.
        """
        print('trying to send...')
        if not self.socket:
            logging.warning('Cannot send data. Not connected to the server.')
            return

        try:
            serialized_data = pickle.dumps(message_dict)
            data_length = struct.pack('>I', len(serialized_data))
            message = data_length + serialized_data

            self.socket.sendall(message)
            logging.info(f"Sent data to server: {message_dict}")
        except Exception as e:
            logging.error(f"Failed to send data: {e}")
            self.disconnect()

    def receive(self):
        """
        Receives a dictionary from the server.
        This method blocks until the data is received.
        """
        print('trying to receive...')
        if not self.socket:
            logging.warning('Cannot receive data. Not connected to the server.')
            return None

        try:
            # First, receive the length of the incoming data (4 bytes)
            raw_length = self.receive_all(4)
            if not raw_length:
                logging.info('No data received. Closing connection.')
                self.disconnect()
                return None
            data_length = struct.unpack('>I', raw_length)[0]
            logging.info(f"Expecting to receive {data_length} bytes of data from server")

            # Now receive the actual data
            serialized_data = self.receive_all(data_length)
            if not serialized_data:
                logging.info('No serialized data received. Closing connection.')
                self.disconnect()
                return None

            # Deserialize the data back into a Python object
            data_dict = pickle.loads(serialized_data)
            logging.info(f"Received data from server: {data_dict.keys()}")
            return data_dict
        except Exception as e:
            logging.error(f'An error occurred while receiving data: {e}')
            self.disconnect()
            return None

    def receive_all(self, length):
        """
        Helper function to receive exactly 'length' bytes from the socket.
        """
        data = b''
        while len(data) < length:
            more = self.socket.recv(length - len(data))
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
            'client_array1': np.random.rand(3),
            'client_array2': np.random.randint(0, 50, size=(2, 2))
        }
        return data_dict

def main():
    client = ClientAPI(host='10.40.11.68', port=8888)
    client.connect()

    try:
        while True:
            # Receive data from server
            data = client.receive()

            if data is None:
                break

            # Send data to server
            data_to_send = client.generate_data()
            client.send(data_to_send)

            # Process the received data as needed
            # For example, print it
            logging.info(f"Processing data from server: {data}")

    except KeyboardInterrupt:
        logging.info('Client shutting down.')
    finally:
        client.disconnect()

if __name__ == '__main__':
    main()
