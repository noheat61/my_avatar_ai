# **my.Avatar: Generating 3D Cartoon Avatars Using 2D Facial Images**

![figure1](https://user-images.githubusercontent.com/62093939/195901755-32fb5ea7-b196-49ae-bcb1-efd12873835e.png)

## Abstract
> Recently, due to the rapid spread of metaverse, interest in 3D avatar generation is increasing significantly. Technology for modeling human face images as 3D objects has been studied a lot, but research on generating 3D avatars such as cartoon characters lacks research results compared to their demand. In this paper, we present a deep learning technology-based pipeline that generates ② 3D avatars after performing domain conversion of live images into the cartoon character style you want. In addition, in a previous study, CartoonStyleGAN, we improved the limitations of no natural domain conversion when user images are input using face image alignment. It is hoped that this study will be the cornerstone of the study of creating 3D cartoon avatar objects based on human face images.

## Inference Notebook
<a href="https://colab.research.google.com/gist/noheat61/062a03245cf495cf3674df7a6cddfada/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Getting started

Clone the repo:
```shell
https://github.com/noheat61/my_avatar_ai
cd my_avatar_ai
```

### **Requirements**

* Python 3.8
* CUDA 11.3
* Only Linux(pytorch3d only support Linux)
```shell
# install all packages(recommend using conda)
pip install -r requirements.txt
```

### **Prepare data**

```shell
# Download all CartoonStyleGAN networks
python download_cartoon_data.py

# Download DECA template models
bash download_deca_model.sh
```

### **Run**
Before run scripts below, input your own image in "image/"
```shell
mkdir cartoon_image avatar
python example.py
```
