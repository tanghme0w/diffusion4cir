import torch
from tqdm import tqdm
from functools import partial
import numpy as np
import json
import os
from diffusers.utils import load_image
from blip_diff_pipeline import BlipDiffusionPipeline
import torch


# def evaluate_fashion(model, img2text, args, source_loader, target_loader):
#     model.eval()
#     img2text.eval()
#     all_target_paths = []
#     all_answer_paths = []
#     all_image_features = []
#     all_query_image_features = []
#     all_composed_features = []
#     all_caption_features = []
#     all_mixture_features = []
#     all_reference_names = []
#     all_captions = []
#     m = model.module if args.distributed or args.dp else model
#     logit_scale = m.logit_scale.exp()
#     logit_scale = logit_scale.mean()
#
#     # get target image features and store in all_image_features
#     with torch.no_grad():
#         for batch in tqdm(target_loader):
#             target_images, target_paths = batch
#             if args.gpu is not None:
#                 target_images = target_images.cuda(args.gpu, non_blocking=True)
#             image_features = m.encode_image(target_images)
#             image_features = image_features / image_features.(dim=-1, keepdim=True)
#             all_image_features.append(image_features)
#             for path in target_paths:
#                 all_target_paths.append(path)
#
#     with torch.no_grad():
#         for batch in tqdm(source_loader):
#             ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
#             for path in answer_paths:
#                 all_answer_paths.append(path)
#             all_reference_names.extend(ref_names)
#             all_captions.extend(captions)
#             ref_images = ref_images.cuda(args.gpu, non_blocking=True)
#             target_images = target_images.cuda(args.gpu, non_blocking=True)
#             target_caption = target_caption.cuda(args.gpu, non_blocking=True)
#             caption_only = caption_only.cuda(args.gpu, non_blocking=True)
#             image_features = m.encode_image(target_images)
#             query_image_features = m.encode_image(ref_images)
#             id_split = tokenize(["*"])[0][1]
#             caption_features = m.encode_text(target_caption)
#             query_image_tokens = img2text(query_image_features)
#             composed_feature = m.encode_text_img_retrieval(target_caption, query_image_tokens, split_ind=id_split,
#                                                            repeat=False)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#             caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
#             query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)
#             mixture_features = query_image_features + caption_features
#             mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
#             composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
#
#             all_caption_features.append(caption_features)
#             all_query_image_features.append(query_image_features)
#             all_composed_features.append(composed_feature)
#             all_mixture_features.append(mixture_features)
#
#         metric_func = partial(get_metrics_fashion,
#                               image_features=torch.cat(all_image_features),
#                               target_names=all_target_paths, answer_names=all_answer_paths)
#         feats = {'composed': torch.cat(all_composed_features),
#                  'image': torch.cat(all_query_image_features),
#                  'text': torch.cat(all_caption_features),
#                  'mixture': torch.cat(all_mixture_features)}
#
#         for key, value in feats.items():
#             metrics = metric_func(ref_features=value)
#             print(f"Eval {key} Feature" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
#     return metrics


def get_metrics_fashion(image_features, ref_features, target_names, answer_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_gallery(model, image_path, split_path, split, cls, save_path):
    # build gallery: target image with null text prompt
    dress_target_images = json.load(open(os.path.join(split_path, f"split.{cls}.{split}.json")))
    target_matrix = []
    with torch.no_grad():
        for image_name in tqdm(dress_target_images, desc=f"Build Emb Gallery - {cls}: "):
            image = load_image(os.path.join(image_path, f"{image_name}.png"))
            feat = model.get_unified_embed(
                prompt=[""],
                reference_image=image,
                source_subject_category=f"{cls}",
                target_subject_category=f"{cls}",
            )  # Tensor: [1, 768]
            target_matrix.append(feat.cpu())
    target_matrix = np.asarray(target_matrix, dtype=np.float32).squeeze()
    np.save(os.path.join(save_path, f"{cls}_{split}_all_features.npy"), target_matrix)
    return target_matrix


if __name__ == '__main__':
    classes = ["dress", "shirt", "toptee"]

    # build target image gallery
    bd_model = BlipDiffusionPipeline.from_pretrained(
        "Salesforce/blipdiffusion", torch_dtype=torch.float16
    ).to(device)
    for cls in classes:
        build_gallery(
            model=bd_model,
            image_path="../fashioniq/images",
            split_path="../fashioniq/image_splits",
            split="test",
            cls=cls,
            save_path="feature_save"
        )

    # get scores
