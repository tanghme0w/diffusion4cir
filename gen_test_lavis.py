import torch

from PIL import Image
from lavis.models import load_model_and_preprocess

model, vis_pp, txt_pp = load_model_and_preprocess("blip_diffusion", "base", device="cuda", is_eval=True)

cond_subject = "dress"
tgt_subject = "dress"
# prompt = "painting by van gogh"
text_prompt = "is solid black with no sleeves"

cond_subjects = [txt_pp["eval"](cond_subject)]
tgt_subjects = [txt_pp["eval"](tgt_subject)]
text_prompt = [txt_pp["eval"](text_prompt)]

cond_image = Image.open("/home/me0w/Desktop/CIR/fashioniq/images/B003FGW7MK.png").convert("RGB")
cond_image.show()

cond_images = vis_pp["eval"](cond_image).unsqueeze(0).cuda()

samples = {
    "cond_images": cond_images,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
}

num_output = 4

iter_seed = 88888
guidance_scale = 7.5
num_inference_steps = 100
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

for i in range(num_output):
    output = model.generate(
        samples,
        seed=iter_seed + i,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )

    output[0].show()
