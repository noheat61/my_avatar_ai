import fbx
from FbxCommon import InitializeSdkObjects, LoadScene, SaveScene
from pathlib import Path
import os

def main(body, head, config, out):
    manager, body_scene = InitializeSdkObjects()

    head_scene = fbx.FbxScene.Create(manager, "")

    try:
        LoadScene(manager, body_scene, str(Path(body).resolve()))
        LoadScene(manager, head_scene, str(Path(head).resolve()))
        
        # 머리 붙이기
        destination_node_name = "mixamorig:Head"
        destination_node = find(body_scene.GetRootNode(), destination_node_name)
        if destination_node is None:
            print("없는데?")
            return

        for i in range(head_scene.GetRootNode().GetChildCount()):
            child = head_scene.GetRootNode().GetChild(i)
            destination_node.AddChild(child)
            
            child.LclScaling.Set(fbx.FbxDouble3(config["s"], config["s"], config["s"]))
            child.LclTranslation.Set(fbx.FbxDouble3(config["x"], config["y"], config["z"]))

        head_scene.GetRootNode().DisconnectAllSrcObject()

        for i in range(head_scene.GetSrcObjectCount()):
            obj = head_scene.GetSrcObject(i)
            if obj == head_scene.GetRootNode() or obj.GetName() == 'GlobalSettings':
                continue
            obj.ConnectDstObject(body_scene)

        head_scene.DisconnectAllSrcObject()

        SaveScene(manager, body_scene, str(Path(out).resolve()), pEmbedMedia=True)
    finally:
        body_scene.Destroy()
        head_scene.Destroy()
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
