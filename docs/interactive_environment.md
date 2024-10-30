# Quick Start Guide

Follow these steps to set up and run the application.

### 1. Clone the Repository
Clone the code to your local machine.

### 2. Environment Setup
- **Local Dependencies**: Install all dependencies locally using `pip`. These are straightforward installations.
- **Server Setup**: Follow the installation guide on the server end. Refer to [docs/install.md](docs/install.md).

### 3. Configure Server IP and Port
Modify the server’s IP address and port number. For example:
```python
client = ClientAPI(host='10.40.11.68', port=8888)
```
This modification is located in `message/client_ui/driving.py` at line 57.

### 4. Set Interactive Mode
Ensure the `is_interactive` flag is enabled in the following file:
```
nuplan_extent/planning/training/modeling/models/transition_models/smart_transition_model.py
```
Check line 117.

### 5. Start Docker
Verify the port setup and start the remote Docker by running:
```bash
bash docker/docker_run.sh
```

### 6. Launch the Server
Start the server by executing the train/validation script:
```bash
bash scripts/training/train_wod_smart.sh
```
Wait for the server to prompt:
```
Server listening on 0.0.0.0:8888
```

### 7. Start the Client
On your local machine, navigate to the `message` directory and run the client:
```bash
cd ./message
python3 client_ui/driving.py
```
> **Note**: Some folder paths have been adjusted, but they may not be fully tested. Minor path errors might occur and should be easy to resolve.

### 8. Start Driving
You’re all set! Use your keyboard to control the application.
