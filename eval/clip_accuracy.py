import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import json
from tqdm.auto import tqdm
import pandas as pd
import argparse
import clip

class CLIPMetric:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model, preprocess = clip.load(model_name, device=self.device)
        self.model = model.eval()
        self.preprocess = preprocess

    def score(self, images, text):
        images = images.to(self.device)
        if isinstance(text, list):
            text = clip.tokenize(text).to(self.device)
        else:
            text = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(images, text)

        return logits_per_image


class InferenceDataset(Dataset):
    def __init__(self, inference_dir, test_scene, clip_preprocess, eval_resolution=256):
        self.inference_dir = inference_dir
        self.file_names = os.listdir(inference_dir)
        self.clip_preprocess = clip_preprocess
        self.eval_resolution = eval_resolution
        with open(test_scene) as fp:
            self.test_scene = json.load(fp)
        self.ids = {file_name.split("_")[0] for file_name in self.file_names}
        self.ids = sorted(self.ids)

        self.collect_all_classes()

    def __len__(self):
        return len(self.ids)

    def read_instruction(self, path):
        with open(path) as fp:
            return fp.read()

    def scale_box(self, box, scale_ratio):
        return list(map(lambda x: int(x * scale_ratio), box))

    def get_cropped_boundary(self, object_bbox, image_size_orig):
        width, height = image_size_orig
        min_size = min(image_size_orig)
        object_bbox[0] -= (width - min_size) // 2
        object_bbox[1] -= (height - min_size) // 2
        object_bbox[2] -= (width - min_size) // 2
        object_bbox[3] -= (height - min_size) // 2
        object_bbox = np.clip(object_bbox, 0, min_size)
        return object_bbox

    def get_scaled_boundary(self, object_bbox, scale_ratio):
        object_bbox = np.array(self.scale_box(object_bbox, scale_ratio))
        return object_bbox

    def read_image(self, path):
        img = Image.open(path).resize((self.eval_resolution,self.eval_resolution), Image.BILINEAR)
        return img

    def collect_all_classes(self):
        classes = set()
        for scene_id in self.ids:
            img_id, obj_id = scene_id.split("-")
            img_id = img_id[1:]
            obj_id = obj_id[1:]
            classes.add(self.test_scene[img_id]["objects"][obj_id]["name"])
        self.classes = list(classes)

    def add_padding(self, image):	
        padding_color = (0,0,0)
        width, height = image.size	
        if width > height:	
            padded_image = Image.new(image.mode, (width, width), padding_color)	
            padded_image.paste(image, (0, (width - height) // 2))	
        else:	
            padded_image = Image.new(image.mode, (height, height), padding_color)	
            padded_image.paste(image, ((height - width) // 2, 0))	
        return padded_image

    def __getitem__(self, idx):
        scene_id = self.ids[idx]
        img_id, obj_id = scene_id.split("-")
        img_id = img_id[1:]
        obj_id = obj_id[1:]

        source_image = self.read_image(f"{self.inference_dir}/{scene_id}_source_image.jpg")
        target_image = self.read_image(f"{self.inference_dir}/{scene_id}_target_image.jpg")
        inpainted_image = self.read_image(f"{self.inference_dir}/{scene_id}_inpainted.jpg")

        instruction = self.read_instruction(f"{self.inference_dir}/{scene_id}_instruction.txt")

        object_bbox = self.test_scene[img_id]["objects"][obj_id]["bbox"]
        object_name = self.test_scene[img_id]["objects"][obj_id]["name"]
        
        image_size_orig = (self.test_scene[img_id]["width"], self.test_scene[img_id]["height"])
        object_bbox = self.get_cropped_boundary(object_bbox, image_size_orig)
        
        scale_ratio =  self.eval_resolution / min(image_size_orig)
        object_bbox = np.array(self.scale_box(object_bbox, scale_ratio))
        return (
            self.clip_preprocess(self.add_padding(source_image.crop(object_bbox))),	
            self.clip_preprocess(self.add_padding(target_image.crop(object_bbox))),	
            self.clip_preprocess(self.add_padding(inpainted_image.crop(object_bbox))),
            instruction,
            object_name,
            scene_id,
        )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_scene",
        type=str,
        default="/userfiles/hpc-byildirim/gqa-inpaint/test_scenes.json",
        help="path of the test scene",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the inference results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gqa_inpaint_eval/",
        help="Directory of evaluation outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Bath size of CLIP forward pass",
    )
    args = parser.parse_args()

    clip_metric = CLIPMetric(model_name="ViT-B/32")

    dataset = InferenceDataset(args.inference_dir, args.test_scene, clip_metric.preprocess, eval_resolution=256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    def get_dict():
        scores = {}
        scores.update({c: [] for c in dataset.classes})
        scores["ground_truth"] = []
        return scores

    scene_ids = []
    source_scores = get_dict()
    inference_scores = get_dict()

    prompts = list(map(lambda x: f"a photo of a {x}", dataset.classes))

    for idx, (source_img, target_img, inpainted_img, instruction, object_names, scene_id) in enumerate(tqdm(dataloader)):
        scene_ids.extend(list(scene_id))
        src_scores_list = clip_metric.score(source_img, prompts)
        prd_scores_list  = clip_metric.score(inpainted_img, prompts)
        tgt_scores_list = clip_metric.score(target_img, prompts)
        for src_scores, prd_scores, tgt_scores, object_name in zip(src_scores_list, prd_scores_list, tgt_scores_list, object_names):
            object_idx = dataset.classes.index(object_name)
            for (c_name, s_score, p_score, t_score) in zip(dataset.classes, src_scores, prd_scores, tgt_scores):
                source_scores[c_name].append(s_score.item())
                inference_scores[c_name].append(p_score.item())
            source_scores["ground_truth"].append(src_scores[object_idx].item())
            inference_scores["ground_truth"].append(prd_scores[object_idx].item())

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    df_source = pd.DataFrame.from_dict(source_scores).set_index([scene_ids])
    df_source.to_csv(f"{output_dir}/source_scores.csv")

    df_inference = pd.DataFrame.from_dict(inference_scores).set_index([scene_ids])
    df_inference.to_csv(f"{output_dir}/inference_scores.csv")

    classes = list(df_source.columns)
    classes.remove("ground_truth")

    out_str_list = []
    for top_n in [1,3,5]:
        diff_count = 0
        for im_id in range(len(df_source)):
            class_source = pd.to_numeric(df_source.iloc[im_id][classes]).nlargest(1).index[0]
            top_classes_inpainted = pd.to_numeric(df_inference.iloc[im_id][classes]).nlargest(top_n).index
            if class_source not in top_classes_inpainted:
                diff_count += 1
        score = diff_count/len(df_source)
        output = f"CLIP@{top_n}: {score}"
        out_str_list.append(output)
        print(output)

    f = open(f"{output_dir}/clip_256.txt", "w")
    f.write("\n".join(out_str_list))
    f.close()