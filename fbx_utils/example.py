from mergehair import main as mergehair
from mergebody import main as mergebody
import glob
import os
import json

body_path = "fbx_utils/body/"
hair_path = "fbx_utils/hair/"
avatar_path = "avatar/"

body = "1.fbx"
hair = "mhair0.fbx"

with open("fbx_utils/config_body.json", "r") as json_file:
    body_name = os.path.splitext(body)[0]
    config_body = json.load(json_file)[body_name]
    
with open("fbx_utils/config_hair.json", "r") as json_file:
    hair_name = os.path.splitext(hair)[0]
    config_hair = json.load(json_file)[hair_name]

avatar_list = glob.glob(avatar_path+"*")
for filename in avatar_list:
    mergehair(head = filename+"/"+os.path.basename(filename)+".fbx", hair = hair_path+hair, config=config_hair, out=filename+"/"+os.path.basename(filename)+"-hair.fbx")
    mergebody(head = filename+"/"+os.path.basename(filename)+"-hair.fbx", body = body_path+body, config=config_body, out=filename+"/"+os.path.basename(filename)+"-merge.fbx")