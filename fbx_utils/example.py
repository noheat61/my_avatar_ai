from fbxmerge import main as fbxmerge
import glob
import os

body_path = "fbx_utils/body/"
avatar_path = "avatar/"

body = "1.fbx"

avatar_list = glob.glob(avatar_path+"*")
for filename in avatar_list:
    fbxmerge(head = filename+"/"+os.path.basename(filename)+".obj", body = body_path+body, out=filename+"/"+os.path.basename(filename)+"-merge.fbx")