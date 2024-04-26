import gradio as gr
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import spaces
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from torchvision.transforms.functional import to_pil_image


def process_image(base64_image):
    # 将 base64 编码的图像字符串转换为图像
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

    # 如果图像不是灰度图像，转换为灰度图像
    if image.mode != "L":
        image = image.convert("L")

    # 将图像转换为 numpy 数组
    inpaint_mask = np.array(image) / 255.0

    # 创建 mask
    mask = Image.fromarray((inpaint_mask * 255).astype(np.uint8))

    # 创建 mask_gray
    mask_gray = Image.fromarray((inpaint_mask * 127).astype(np.uint8))

    return mask, mask_gray


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = "yisol/IDM-VTON"
example_path = os.path.join(os.path.dirname(__file__), "example")

# 图像分割
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
# 分词器
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)

# 离散去噪调度器 Denoising Diffusion Probabilistic Models Scheduler
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

# Contrastive Language-Image Pre-Training 对比 文本-图像 预训练 图像和文本的互相转换
# 文本 转 向量，使得模型能够理解
text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
# 添加投影头，增强模型文本特征的表达能力
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
# 图像 转 向量，使得模型能够理解，从而与文本信息进行匹配
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
# 变分自编码器 Variational Auto-encoder 增强图形
vae = AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

from flask import Flask, request
import sys
import json
import io
import base64

app = Flask(__name__)


def render(data):
    result_dict = {}
    result_dict["code"] = "SUCCESS"
    result_dict["msg"] = "success"
    result_dict["data"] = data
    return json.dumps(result_dict)


@app.route("/", methods=["POST"])
def handler():
    for k, v in request.headers.items():
        if k.startswith("HTTP_"):
            # process custom request headers
            pass

    request_body = request.data
    request_method = request.method
    path_info = request.path
    content_type = request.content_type
    query_string = request.query_string.decode("utf-8")

    # print("request_body: {}".format(request_body))
    # print(
    #     "method: {} path: {} query_string: {}".format(
    #         request_method, path_info, query_string
    #     )
    # )
    body = json.loads(request_body)

    # args
    steps = body.get("steps", 30)
    seed = body.get("seed", -1)
    scale = body.get("scale", 2.0)
    # "model is wearing " + garment_des | "a photo of " + garment_des
    garment_des = body.get("garment_des", "clothing")
    prompt = body.get("prompt", None)
    negative_prompt = body.get(
        "negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality"
    )
    # garment base64
    garment_image = body["garment_image"]
    # human base64
    human_image = body["human_image"]
    # mask base64
    mask_image = body.get("mask_image", None)
    # Use auto-crop & resizing
    is_checked_crop = body.get("is_checked_crop", False)
    # OOTD
    model_type = body.get("model_type", "hd")  # "hd" or "dc"
    category = body.get("category", 0)  # 0:upperbody; 1:lowerbody; 2:dress

    # initialize
    model_prompt = prompt if prompt else "model is wearing " + garment_des
    photo_prompt = prompt if prompt else "a photo of " + garment_des

    category_dict_utils = ["upper_body", "lower_body", "dresses"]

    # do something here
    device = "cuda"

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = (
        Image.open(io.BytesIO(base64.b64decode(garment_image)))
        .convert("RGB")
        .resize((768, 1024))
    )
    human_img_orig = Image.open(io.BytesIO(base64.b64decode(human_image))).convert(
        "RGB"
    )

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if not mask_image:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            model_type, category_dict_utils[category], model_parse, keypoints
        )
        mask = mask.resize((768, 1024))
    else:
        # mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
        mask, mask_gray = process_image(mask_image).convert("RGB").resize((768, 1024))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            "cuda",
        )
    )
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = model_prompt
                negative_prompt = negative_prompt
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = photo_prompt
                    negative_prompt = negative_prompt
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = (
                        tensor_transfrom(pose_img)
                        .unsqueeze(0)
                        .to(device, torch.float16)
                    )
                    garm_tensor = (
                        tensor_transfrom(garm_img)
                        .unsqueeze(0)
                        .to(device, torch.float16)
                    )
                    generator = (
                        torch.Generator(device).manual_seed(seed)
                        if seed is not None
                        else None
                    )
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(
                            device, torch.float16
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(
                            device, torch.float16
                        ),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                            device, torch.float16
                        ),
                        num_inference_steps=steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=scale,
                    )[0]

    image_idx = 0
    data = []
    for image in images:
        if is_checked_crop:
            out_img = image.resize(crop_size)
            image.paste(out_img, (int(left), int(top)))
        image.save("./images_output/out_" + model_type + "_" + str(image_idx) + ".png")
        image_idx += 1
        # 将图片转换为 BytesIO 对象
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # 将 BytesIO 对象转换为 base64 编码
        base64_image = base64.b64encode(img_byte_arr).decode("utf-8")

        # 将 Base64 字符串包装在 {"image": <base64>} 的格式中
        image_data = {"image": base64_image}

        # 将包含图像数据的字典附加到 data 列表中
        data.append(image_data)

    return render(data), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run()
