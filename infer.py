from abc import ABC, abstractclassmethod, abstractmethod
import os
import torch
import argparse
import glob

from ffhq_align import image_align_68
import face_alignment

from CartoonStyleGAN.model import Generator
from CartoonStyleGAN.utils import tensor2image, save_image
from CartoonStyleGAN.projector import main as projector

from DECA.demos.demo_reconstruct import main as DECA_run


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def inference(self, model_input, kwargs=None):
        pass

# Mdel2D: 이미지 도메인 변환 모델(CartoonStyleGAN)
class Model2D(Model):
    def __init__(self, path="CartoonStyleGAN/networks"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.landmarks_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, flip_input=False
        )

        # 모델을 부를 때 모든 실행 가능한 모든 네트워크를 load함
        # 국가(현재 AMERICAN만 지원), 스타일별로 딕셔너리에 저장
        self.model_dict = dict()
        file_list_pt = glob.glob(path + "/*.pt")
        for filename in file_list_pt:
            tokens = os.path.basename(filename)[:-3].split("_")
            if len(tokens) == 1:
                tokens.append("generator")

            if tokens[0] not in self.model_dict:
                self.model_dict[tokens[0]] = dict()
            self.model_dict[tokens[0]][tokens[1]] = torch.load(filename)

    def make_image(
        self, nation, style, aligned_path, output_path, swap_layer_num, truncation
    ):

        # 각각의 네트워크(생성자, 만화 생성자)를 가지고 생성 모델 만들기
        # generator1: 실사 이미지 생성 모델
        # generator2: 만화 이미지 생성 모델
        generator1 = Generator(256, 512, 8, channel_multiplier=2).to(self.device)
        generator1.load_state_dict(self.generator["g_ema"], strict=False)
        trunc1 = generator1.mean_latent(4096)

        generator2 = Generator(256, 512, 8, channel_multiplier=2).to(self.device)
        generator2.load_state_dict(
            self.model_dict[nation][style]["g_ema"], strict=False
        )
        trunc2 = generator2.mean_latent(4096)

        # 생성 모델에 추출한 잠재 벡터를 입력으로 넣어 이미지 만들기
        # truncation: 이미지를 얼마나 평균에 가까운 이미지로 보정할지(0에 가까울수록 더 큰 보정)
        # layer swapping을 통해 원본 이목구비 정보를 더욱 유지한 이미지로 생성
        # swap_layer_num: swap할 layer(낮을수록 만화에 가까운 이미지, 높을수록 실사에 가까운 이미지)
        with torch.no_grad():
            _, save_swap_layer = generator1(
                [self.latent],
                input_is_latent=True,
                truncation=truncation,
                truncation_latent=trunc1,
                swap=True,
                swap_layer_num=swap_layer_num,
                randomize_noise=False,
            )

            imgs_gen, _ = generator2(
                [self.latent],
                input_is_latent=True,
                truncation=truncation,
                truncation_latent=trunc2,
                swap=True,
                swap_layer_num=swap_layer_num,
                swap_layer_tensor=save_swap_layer,
                randomize_noise=True,
            )

            # 이미지 저장하고 path 전달
            cartoonized_path = (
                output_path
                + os.path.splitext(os.path.basename(aligned_path))[0]
                + f"-{style}-{swap_layer_num}.png"
            )
            save_image(tensor2image(imgs_gen), out=cartoonized_path)
            # Image.fromarray(make_image(imgs_gen)[0]).save(cartoonized_path)
            return cartoonized_path

    def inference(
        self, input_path, output_path, nation="AMERICAN", make_all=True, style="DISNEY"
    ):

        self.generator = self.model_dict[nation]["generator"]
        self.encoder = self.model_dict[nation]["encoder"]

        # 얼굴 이미지 조정(face-alignment) 전처리 수행
        try:
            face_landmarks = self.landmarks_detector.get_landmarks(input_path)
        except:
            print(f"Cannot detect face of {input_path}")
            return None

        img_name, extension = os.path.splitext(input_path)
        aligned_path = img_name + "-align" + extension
        image_align_68(input_path, aligned_path, face_landmarks[0])

        # 이미지 인코딩 -> project.pt에 잠재 벡터 저장
        projector_out = projector(
            "networks/factor",
            self.generator,
            self.encoder,
            [aligned_path],
        )
        if projector_out is False:
            return None

        # project.pt에서 인코딩된 잠재 벡터(1 * 14 * 512) 불러오기
        project = torch.load("project.pt")
        self.latent = project[os.path.splitext(os.path.basename(aligned_path))[0]][
            "latent"
        ]
        self.latent = self.latent.to(self.device)

        # 입력된 정보들로 make_image 실행하여 이미지 도메인 변환
        # (swap_layer_num, truncation)
        # -> (4, 0.6): weak cartoonization
        # -> (3, 0.4): strong cartoonization
        if make_all is True:
            for key in self.model_dict[nation]:
                if key == "generator" or key == "encoder":
                    continue
                self.make_image(
                    nation,
                    key,
                    aligned_path,
                    output_path,
                    swap_layer_num=4,
                    truncation=0.6,
                )
                self.make_image(
                    nation,
                    key,
                    aligned_path,
                    output_path,
                    swap_layer_num=3,
                    truncation=0.6,
                )
        else:
            self.make_image(
                nation,
                style,
                aligned_path,
                output_path,
                swap_layer_num=4,
                truncation=0.6,
            )
            self.make_image(
                nation,
                style,
                aligned_path,
                output_path,
                swap_layer_num=3,
                truncation=0.6,
            )
        return output_path

# Model3D: 3D 객체 생성 모델(DECA)
class Model3D(Model):
    def __init__(self):
        pass

    def inference(self, input_path, output_path, get_full):

        # DECA의 demo_reconstruct.py 실행
        parser = argparse.ArgumentParser(
            description="DECA: Detailed Expression Capture and Animation"
        )

        parser.add_argument(
            "-i",
            "--inputpath",
            default="TestSamples/examples",
            type=str,
            help="path to the test data, can be image folder, image path, image list, video",
        )
        parser.add_argument(
            "-s",
            "--savefolder",
            default="TestSamples/examples/results",
            type=str,
            help="path to the output directory, where results(obj, txt files) will be stored.",
        )
        parser.add_argument(
            "--device", default="cuda", type=str, help="set device, cpu for using cpu"
        )
        # process test images
        parser.add_argument(
            "--iscrop",
            default=True,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to crop input image, set false only when the test image are well cropped",
        )
        parser.add_argument(
            "--sample_step",
            default=10,
            type=int,
            help="sample images from video data for every step",
        )
        parser.add_argument(
            "--detector",
            default="fan",
            type=str,
            help="detector for cropping face, check decalib/detectors.py for details",
        )
        # rendering option
        parser.add_argument(
            "--rasterizer_type",
            default="standard",
            type=str,
            help="rasterizer type: pytorch3d or standard",
        )
        parser.add_argument(
            "--render_orig",
            default=True,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to render results in original image size, currently only works when rasterizer_type=standard",
        )
        # save
        parser.add_argument(
            "--useTex",
            default=False,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to use FLAME texture model to generate uv texture map, \
                                set it to True only if you downloaded texture model",
        )
        parser.add_argument(
            "--extractTex",
            default=True,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode",
        )
        parser.add_argument(
            "--saveVis",
            default=True,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to save visualization of output",
        )
        parser.add_argument(
            "--saveKpt",
            default=False,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to save 2D and 3D keypoints",
        )
        parser.add_argument(
            "--saveDepth",
            default=False,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to save depth image",
        )
        parser.add_argument(
            "--saveObj",
            default=False,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                                Note that saving objs could be slow",
        )
        parser.add_argument(
            "--saveMat",
            default=False,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to save outputs as .mat",
        )
        parser.add_argument(
            "--saveImages",
            default=False,
            type=lambda x: x.lower() in ["true", "1"],
            help="whether to save visualization output as seperate images",
        )
        args, _ = parser.parse_known_args()

        args.inputpath = input_path
        args.savefolder = output_path
        args.useTex = get_full  # 뒷부분 텍스처 추가 생성 여부

        args.rasterizer_type = "pytorch3d"
        args.saveObj = True
        args.saveVis = False

        # DECA 실행하여 아바타 생성 후 path 전달
        DECA_run(args)
        return os.path.join(
            output_path, os.path.splitext(os.path.basename(input_path))[0]
        )
