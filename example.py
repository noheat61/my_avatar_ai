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
                    make_all = False, style = "COMICS") # "DISNEY_w", "DISNEY_s", "여신강림_w", "여신강림_s" "COMICS", "ART"

model3D.inference(input_path=cartoon_path, output_path=avatar_path, get_full = True)