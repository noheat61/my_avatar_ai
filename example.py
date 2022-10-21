from infer import Model2D, Model3D
import glob

model2D = Model2D()
model3D = Model3D()

input_path = "images/*"

cartoon_path = "cartoon_image/"
avatar_path = "avatar/"

file_list = glob.glob(input_path)
for filename in file_list:
    model2D.inference(input_path = filename, output_path = cartoon_path,
                    make_all = True, style = "DISNEY") # "DISNEY", "COMICS", "ART", "여신강림", "외모지상주의"

model3D.inference(input_path=cartoon_path, output_path=avatar_path, get_full = True)