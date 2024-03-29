# **my.Avatar: Generating 3D Cartoon Avatars Using 2D Facial Images**

![1](https://user-images.githubusercontent.com/62093939/200192543-908ed3d8-8f65-4c1c-8865-3c9eb4a206c7.jpg)

## **Project Info**

[![Video Label](http://img.youtube.com/vi/nVqUi7qXRGc/0.jpg)](https://youtu.be/nVqUi7qXRGc?t=0s)

## **Abstract**

The 3D avatar generation methods have been largely attracting attention as the market of metaverse applications grows significantly. Previous work has studied the 3D reconstruction methods that map the 2D human face images to the 3D objects, however, research on generating 3D cartoon-like avatars has gained little attention. In this paper, we present a deep learning-based pipeline that (1) changes the domain of original images to the cartoon domain that is selected by a user and (2) generates 3D avatar objects using the cartoon 2D image whose domain is changed. Moreover, as we apply the face alignment method to the CartoonStyleGAN, we have improved the quality of generated 2D cartoon images which results in improved 3D head reconstruction. We hope that our work can be a milestone for the research on generating 3D cartoon avatar objects based on 2D human face images.

## **Inference Notebook**

<a href="https://colab.research.google.com/github/noheat61/my.Avatar-AI/blob/main/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## **Getting started**

Clone the repo:

```shell
https://github.com/noheat61/my_avatar_ai
cd my_avatar_ai
```

### **Requirements**

-   Python 3.9
-   CUDA 11.3
-   Linux(pytorch3d only support Linux)

```shell
# Install all packages(recommend using conda)
pip install -r requirements.txt
```

### **Prepare data**

You can download all the files you need just by running the script frame below.

```shell
# Download all CartoonStyleGAN networks
python download_cartoon_data.py

# Download DECA template models
bash download_deca_model.sh
```

### **Run**

Before run scripts below, input your own image in "images/".

```shell
# Make directories
mkdir cartoon_image avatar

# Inference
python example.py
```

## **Datasets**

| Dataset           | Path                                                   |
| :---------------- | :----------------------------------------------------- |
| FFHQ              | https://github.com/rosinality/stylegan2-pytorch        |
| StyleGAN2-Encoder | https://github.com/bryandlee/stylegan2-encoder-pytorch |
| Naver Webtoon     | https://github.com/bryandlee/naver-webtoon-data        |
| DISNEY            | https://github.com/justinpinkney/toonify               |
| COMICS            | https://github.com/Sxela/face2comics                   |
| ART               | https://github.com/NVlabs/metfaces-dataset             |

## **More: Improve the accuracy of CartoonStyleGAN**

We improved the accuracy of the projection by performing face-alignment preprocessing before putting the image as an input of CartoonStyleGAN.

![2](https://user-images.githubusercontent.com/62093939/200192553-d93284a7-3d8b-43e5-9514-6a0d6df20662.jpg)

For more details, please check our paper.

## **More: Connect with the body**

You can connect your own avatar head with 3D body we provide(modified from mixamo).

Due to the compatibility between packages, this function must be performed in a new environment from **<U>python 3.7</U>**.

```shell
# Install 3D packages(fbx, bpy)
pip install -r fbx_utils/requirements.txt
bpy_post_install
bash fbx_utils/install_fbx.sh

# Download 3D bodies
bash fbx_utils/download_body.sh

# Convert obj to fbx(3D heads)
python fbx_utils/obj2fbx.py

# Connect with body(default: 1.fbx)
python fbx_utils/example.py
```
