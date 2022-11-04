import fbx
from FbxCommon import InitializeSdkObjects, LoadScene, SaveScene
from pathlib import Path
import json
import os

def main(body, head, hair, out):
    manager, body_scene = InitializeSdkObjects()

    head_scene = fbx.FbxScene.Create(manager, "")
    hair_scene = fbx.FbxScene.Create(manager, "")
    print(head)
    with open("fbx_utils/config.json", "r") as json_file:
        cfx = json.load(json_file)
        body_name=os.path.splitext(os.path.basename(body))[0]
        hair_name=os.path.splitext(os.path.basename(hair))[0]
        head_config = cfx[body_name]["head"]
        hair_config = cfx[body_name][hair_name]
        print(head_config)

    try:
        LoadScene(manager, body_scene, str(Path(body).resolve()))
        LoadScene(manager, head_scene, str(Path(head).resolve()))
        LoadScene(manager, hair_scene, str(Path(hair).resolve()))
        
        # 머리 붙이기
        destination_node_name = "mixamorig:Head"
        destination_node = find(body_scene.GetRootNode(), destination_node_name)
        if destination_node is None:
            print("없는데?")
            return

        for i in range(head_scene.GetRootNode().GetChildCount()):
            child = head_scene.GetRootNode().GetChild(i)
            print(child)
            destination_node.AddChild(child)
            
            child.LclScaling.Set(fbx.FbxDouble3(head_config["s"], head_config["s"], head_config["s"]))
            child.LclTranslation.Set(fbx.FbxDouble3(head_config["x"], head_config["y"], head_config["z"]))

        head_scene.GetRootNode().DisconnectAllSrcObject()

        for i in range(head_scene.GetSrcObjectCount()):
            obj = head_scene.GetSrcObject(i)
            if obj == head_scene.GetRootNode() or obj.GetName() == 'GlobalSettings':
                continue
            obj.ConnectDstObject(body_scene)

        head_scene.DisconnectAllSrcObject()

        # 머리카락 붙이기
        # destination_node_name = "mixamorig:HeadTop_End"
        # destination_node = find(body_scene.GetRootNode(), destination_node_name)
        # if destination_node is None:
        #     print("없는데?")
        #     return

        # # SaveScene(manager, body_scene, str(Path(out.replace("merge", "merge1")).resolve()), pEmbedMedia=True)

        # for i in range(hair_scene.GetRootNode().GetChildCount()):
        #     child = hair_scene.GetRootNode().GetChild(i)
        #     destination_node.AddChild(child)
            
        #     child.LclScaling.Set(fbx.FbxDouble3(hair_config["s"], hair_config["s"], hair_config["s"]))
        #     child.LclTranslation.Set(fbx.FbxDouble3(hair_config["x"], hair_config["y"], hair_config["z"])) # blender 보고 수치 수정

        # # SaveScene(manager, body_scene, str(Path(out.replace("merge", "merge2")).resolve()), pEmbedMedia=True)

        # for i in range(hair_scene.GetSrcObjectCount()):
        #     obj = hair_scene.GetSrcObject(i)
        #     if obj == hair_scene.GetRootNode() or obj.GetName() == 'GlobalSettings':
        #         continue
        #     obj.ConnectDstObject(body_scene)

        # hair_scene.DisconnectAllSrcObject()

        SaveScene(manager, body_scene, str(Path(out).resolve()), pEmbedMedia=True)
    finally:
        head_scene.Destroy()
        hair_scene.Destroy()
        body_scene.Destroy()
        manager.Destroy()


def find(node, name):
    if node.GetName() == name:
        return node

    for i in range(node.GetChildCount()):
        found = find(node.GetChild(i), name)
        if found is not None:
            return found

    return None


if __name__ == '__main__':
    main()
