# **my.Avatar - AI**

SW 마에스트로 "인공지능을 활용한 실사 이미지로 3D 아바타 생성" AI 파트입니다.

얼굴 이미지를 인식하여 만화 캐릭터 스타일로 도메인 변환한 후, 3D 얼굴 obj 파일을 생성합니다.

## Abstract
> Recently, due to the rapid spread of metaverse, interest in 3D avatar generation is increasing significantly. Technology for modeling human face images as 3D objects has been studied a lot, but research on generating 3D avatars such as cartoon characters lacks research results compared to their demand. In this paper, we present a deep learning technology-based pipeline that generates ② 3D avatars after performing domain conversion of live images into the cartoon character style you want. In addition, in a previous study, CartoonStyleGAN, we improved the limitations of no natural domain conversion when user images are input using face image alignment. It is hoped that this study will be the cornerstone of the study of creating 3D cartoon avatar objects based on human face images.

## Inference Notebook
<a href="https://colab.research.google.com/gist/noheat61/062a03245cf495cf3674df7a6cddfada/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## **Requirements**

- Python 3.8, CUDA 11.3
- `pip install -r requirements.txt`

## **Getting started**

- `python3 download_cartoon_data.py`
- `bash download_deca_model.sh`
- `python3 example.py`
