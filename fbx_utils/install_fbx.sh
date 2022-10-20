# #!/bin/bash

# 1. 실행 파일 다운로드
gdown --id 16zpBaYhmXYW8d4h1OojopxNzdPOhICoH -O fbx202021_fbxpythonsdk_linux.tar.gz
mkdir fbxpython
tar -zxvf fbx202021_fbxpythonsdk_linux.tar.gz -C fbxpython
rm fbx202021_fbxpythonsdk_linux.tar.gz

# 2. python library 생성
cd fbxpython
./fbx202021_fbxpythonsdk_linux

# 3. 파이썬 환경으로 옮기기
PIP="$(which pip)"

ENV="${PIP:0:-4}/../"
PACKAGE="lib/python3.7/dist-packages/"

if [ ! -d "$ENV$PACKAGE" ];
then
PACKAGE="lib/python3.7/site-packages/"
fi

cd lib/Python37_x64
mv * "$ENV$PACKAGE"
cd ../../..

# 4. 남은 파일 제거
rm -rf fbxpython
echo "SUCCESS!"