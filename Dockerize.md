# Dockerize

## Getting Started
### Requirements
- Linux
- [Docker](https://docs.docker.com/desktop/install/linux-install/)
- [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- VRAM > 16.8Gb (Recomend 24GB)

### Build Image
```bash
sudo docker build . -t my_avatar_ai:latest
```

### Run Container
```bash
sudo docker run -it --gpus all my_avatar_ai 
```
If you want to assign gpus, you just change `all` to `''device=#,#...''`. But this model will use only one gpu.

And If you want to get result of this model, you have to mount local system.

example)
```bash
sudo docker run -it --gpus ''device=0'' -v $(pwd):/app my_avatar_ai 
```

### Run Model
```bash
python example.py
```