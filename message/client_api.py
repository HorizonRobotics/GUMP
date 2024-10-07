import socket
import pickle
import struct
import numpy as np
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClientAPI:
    """
    ClientAPI connects to the server, sends and receives serialized dictionaries containing NumPy arrays,
    and processes the received data.
    """

    def __init__(self, host='127.0.0.1', port=8888, on_receive=None):
        """
        Initializes the client with the server's host and port.
        """
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        self.on_receive = on_receive  # Callback function to handle received data
        self.send_lock = threading.Lock()
        self.receive_lock = threading.Lock()

    def connect(self):
        """
        Connects to the server and starts the receiving thread.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            logging.info(f'Connected to server at {self.host}:{self.port}')

            # Start a new thread to listen for incoming data
            receive_thread = threading.Thread(target=self.receive_data, daemon=True)
            receive_thread.start()
        except ConnectionRefusedError:
            logging.error(f"Connection refused by the server at {self.host}:{self.port}")
            self.is_connected = False
        except Exception as e:
            logging.error(f'An error occurred while connecting: {e}')
            self.is_connected = False

    def disconnect(self):
        """
        Disconnects from the server and closes the socket.
        """
        self.is_connected = False
        if self.socket:
            self.socket.close()
            logging.info('Disconnected from server.')

    def send_data(self, message_dict):
        """
        Sends a dictionary containing NumPy arrays to the server.
        """
        if not self.is_connected:
            logging.warning('Cannot send data. Not connected to the server.')
            return

        try:
            serialized_data = pickle.dumps(message_dict)
            data_length = struct.pack('>I', len(serialized_data))
            message = data_length + serialized_data

            with self.send_lock:
                self.socket.sendall(message)
                logging.info(f"Sent data to server: {message_dict}")
        except Exception as e:
            logging.error(f"Failed to send data: {e}")
            self.disconnect()

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

    def receive_data(self):
        """
        Continuously listens for incoming data from the server, deserializes it,
        and invokes the callback function if provided.
        """
        try:
            while self.is_connected:
                # First, receive the length of the incoming data (4 bytes)
                raw_length = self.receive_all(4)
                if not raw_length:
                    logging.info('No data received. Closing connection.')
                    break
                data_length = struct.unpack('>I', raw_length)[0]
                logging.info(f"Expecting to receive {data_length} bytes of data from server")

                # Now receive the actual data
                serialized_data = self.receive_all(data_length)
                if not serialized_data:
                    logging.info('No serialized data received. Closing connection.')
                    break

                # Deserialize the data back into a Python object
                data_dict = pickle.loads(serialized_data)
                logging.info(f"Received data from server: {data_dict}")

                # Invoke the callback function if provided
                if self.on_receive:
                    self.on_receive(data_dict)
                else:
                    # Default behavior: print the received data
                    self.process_data(data_dict)
        except EOFError:
            logging.info('Server closed the connection.')
        except ConnectionResetError:
            logging.error('Connection was reset by the server.')
        except Exception as e:
            logging.error(f'An error occurred while receiving data: {e}')
        finally:
            self.disconnect()

    def process_data(self, data_dict):
        """
        Processes the received data. Override this method or provide a callback to customize behavior.
        """
        logging.info('Received data:')
        for key, array in data_dict.items():
            logging.info(f"{key}: {array}")

def main():
    """
    Example usage of the ClientAPI.
    """

    def handle_received_data(data):
        """
        Custom callback function to handle received data.
        """
        logging.info('Custom Handler - Received data:')
        for key, array in data.items():
            logging.info(f"{key}: {array}")

    # Initialize the client with a custom callback
    client = ClientAPI(host='10.40.11.68', port=8888, on_receive=handle_received_data)
    client.connect()

    try:
        # Example: Sending data from client to server every 5 seconds
        while client.is_connected:
            # Create a sample dictionary to send
            data_to_send = {
                'client_array1': np.random.rand(3),
                'client_array2': np.random.randint(0, 50, size=(2, 2))
            }
            client.send_data(data_to_send)

            # Wait before sending the next message
            threading.Event().wait(5)  # Wait for 5 seconds
    except KeyboardInterrupt:
        logging.info('Interrupted by user.')
    finally:
        client.disconnect()

if __name__ == '__main__':
    main()
