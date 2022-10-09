from abc import ABC, abstractclassmethod, abstractmethod
import os
import torch
import argparse
import glob

from ffhq_align import image_align_68
import face_alignment

from CartoonStyleGAN.model import Generator
from CartoonStyleGAN.utils import tensor2image, save_image
from crop import main as crop
from CartoonStyleGAN.projector import main as projector

from DECA.demos.demo_reconstruct import main as DECA_run

class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def inference(self, model_input, kwargs=None):
        pass


class Model2D(Model):
    def __init__(self, path = "CartoonStyleGAN/networks"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

        self.model_dict = dict()
        file_list_pt = glob.glob(path+"/*.pt")
        for filename in file_list_pt:
            tokens = os.path.basename(filename)[:-3].split("_")
            if len(tokens)==1:
                tokens.append("generator")

            if tokens[0] not in self.model_dict:
                self.model_dict[tokens[0]]=dict()
            self.model_dict[tokens[0]][tokens[1]] = torch.load(filename)

    def make_image(self, nation, style, aligned_path, output_path, swap_layer_num, truncation):
        generator2 = Generator(256, 512, 8, channel_multiplier=2).to(self.device)
        generator2.load_state_dict(self.model_dict[nation][style]["g_ema"], strict=False)
        trunc2 = generator2.mean_latent(4096)

        with torch.no_grad():
            imgs_gen, _ = generator2(
                [self.latent],
                input_is_latent=True,
                truncation=truncation,
                truncation_latent=trunc2,
                swap=True,
                swap_layer_num=swap_layer_num,
                swap_layer_tensor=self.save_swap_layer,
                randomize_noise=True,
            )

            # 이미지 저장하고 path 전달
            cartoonized_path = (
                output_path + os.path.splitext(os.path.basename(aligned_path))[0] + f"-{style}.png"
            )
            save_image(tensor2image(imgs_gen), out=cartoonized_path)
            return cartoonized_path

    def inference(self, nation, input_path, output_path, make_all=True, style="DISNEY", swap_layer_num=4, truncation=0.6):
        
        self.generator = self.model_dict[nation]["generator"]
        self.encoder = self.model_dict[nation]["encoder"]

        # cropped_path = crop(input_path)
        # if cropped_path is None:
        #     return None

        face_landmarks = self.landmarks_detector.get_landmarks(input_path)
        if face_landmarks is None:
            return None
        img_name, extension = os.path.splitext(input_path)
        aligned_path = img_name + "-align" + extension
        image_align_68(input_path, aligned_path, face_landmarks[0])


        projector_out = projector(
            "networks/factor",
            self.generator,
            self.encoder,
            [aligned_path],
        )
        if projector_out is False:
            return None

        project = torch.load("project.pt")
        self.latent = project[os.path.splitext(os.path.basename(aligned_path))[0]]["latent"]
        self.latent = self.latent.to(self.device)

        generator1 = Generator(256, 512, 8, channel_multiplier=2).to(self.device)
        generator1.load_state_dict(self.generator["g_ema"], strict=False)
        trunc1 = generator1.mean_latent(4096)
        
        with torch.no_grad():
            _, self.save_swap_layer = generator1(
                [self.latent],
                input_is_latent=True,
                truncation=0.6,
                truncation_latent=trunc1,
                swap=True,
                swap_layer_num=swap_layer_num,
                randomize_noise=False,
            )

        if make_all is True:
            for key in self.model_dict[nation]:
                if key == "generator" or key == "encoder":
                    continue
                self.make_image(nation, key, aligned_path, output_path, swap_layer_num, truncation)
        else:
            self.make_image(nation, style, aligned_path, output_path, swap_layer_num, truncation)
        return output_path


class Model3D(Model):
    def __init__(self):
        pass

    def inference(self, input_path, output_path):
        parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

        parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                            help='path to the test data, can be image folder, image path, image list, video')
        parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                            help='path to the output directory, where results(obj, txt files) will be stored.')
        parser.add_argument('--device', default='cuda', type=str,
                            help='set device, cpu for using cpu' )
        # process test images
        parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to crop input image, set false only when the test image are well cropped' )
        parser.add_argument('--sample_step', default=10, type=int,
                            help='sample images from video data for every step' )
        parser.add_argument('--detector', default='fan', type=str,
                            help='detector for cropping face, check decalib/detectors.py for details' )
        # rendering option
        parser.add_argument('--rasterizer_type', default='standard', type=str,
                            help='rasterizer type: pytorch3d or standard' )
        parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to render results in original image size, currently only works when rasterizer_type=standard')
        # save
        parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to use FLAME texture model to generate uv texture map, \
                                set it to True only if you downloaded texture model' )
        parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
        parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save visualization of output' )
        parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save 2D and 3D keypoints' )
        parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save depth image' )
        parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                                Note that saving objs could be slow' )
        parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save outputs as .mat' )
        parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save visualization output as seperate images' )
        args = parser.parse_args()

        args.inputpath = input_path
        args.savefolder = output_path
        args.rasterizer_type = "pytorch3d"
        args.saveObj = True
        args.saveVis = False

        DECA_run(args)
        return os.path.join(output_path,os.path.splitext(os.path.basename(input_path))[0])
