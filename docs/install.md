# Installation

To make usage easier, we have prepared a Docker image. Download it from the following link:  
[Download Docker Image](https://storage.googleapis.com/93935945854-us-central1-blueprint-config/hoplan_py39-cu121-pt230-gcc9-devel-nocudnn-wm.tar)

## Using Docker

1. Load the image:
   ```bash
   docker load -i IMAGE_DOWNLOAD_PATH
   ```

2. Edit the Docker mount point:
   Update the mount point in `./docker/docker_run.sh` as needed.

3. Run the Docker container:
   ```bash
   sh ./docker/docker_run.sh
   ```

## Install Nuplan Devkit

1. Navigate to the `third_party` directory:
   ```bash
   cd GUMP/third_party
   ```

2. Clone the Nuplan Devkit repository:
   ```bash
   git clone --branch feat-v1.3_gump git@github.com:HorizonRobotics/nuplan-devkit.git
   ```

3. Install the Nuplan Devkit:
   ```bash
   cd nuplan-devkit
   pip install -e .
   ```

4. Set the `NUPLAN_DEVKIT_PATH` environment variable:
   ```bash
   export NUPLAN_DEVKIT_PATH=$YOUR_PATH_TO_GUMP/GUMP/third_party/nuplan-devkit
   ```
5. install following packages:
   ```bash
   pip install aiofiles aioboto3 flatdict adjustText loralib easydict einops_exts
   pip install waymo-open-dataset-tf-2-12-0==1.6.4
   ``` 

## Install ALF

Follow these steps to install ALF while ignoring the dependencies of torch:

1. Navigate to the `third_party` directory:

    ```bash
    cd GUMP/third_party
    ```

2. Clone the ALF repository:

    ```bash
    git clone https://github.com/HorizonRobotics/alf.git
    ```

3. Edit the `setup.py` file to ignore the dependencies of torch:

    Comment out lines 52, 53, and 54 in the `setup.py` file:

    ```python
    # 'torch==2.2.0',
    # 'torchvision==0.17.0',
    # 'torchtext==0.17.0',
    ```

4. Install ALF:

    ```bash
    pip install -e .
    ```