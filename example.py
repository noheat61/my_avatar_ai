from infer import Model2D, Model3D
import glob

model2D = Model2D()
model3D = Model3D()

input_path = "images/*"

cartoon_path = "cartoon_image/"
avatar_path = "avatar/"

file_list = glob.glob(input_path)
for filename in file_list:
    model2D.inference("AMERICAN", filename, make_all=False, output_path=cartoon_path)

model3D.inference(input_path=cartoon_path, output_path=avatar_path)