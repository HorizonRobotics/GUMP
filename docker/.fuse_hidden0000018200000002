sudo docker run  --rm \
      --gpus all \
      --user root \
      -p 8888:8888 \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      -v /etc/shadow:/etc/shadow:ro \
      --shm-size=256gb \
      -v /mnt/nas20:/mnt/nas20 \
      -v /mnt/nas26:/mnt/nas26 \
      -v /mnt/nas25:/mnt/nas25 \
      -v /home/users/yihan01.hu/data:/home/users/yihan01.hu/data \
      -v /home/users/yihan01.hu/workspace/GUMP:/root/workspace/GUMP \
      -it hoplan:py39-cu121-pt230-gcc9-devel-nocudnn \
      /bin/bash
