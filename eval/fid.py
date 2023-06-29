import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision.models.inception import inception_v3
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from tqdm import tqdm
from PIL import Image
import argparse
import os

def get_frechet_inception_distance(dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception_model = inception_v3(pretrained=True, transform_input=True).to(device)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    resize_images = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
    
    def get_inception_features(image_batch):
        image_batch = resize_images(image_batch)
        inception_output = inception_model(image_batch)
        return inception_output.data.cpu().numpy()

    inception_feature_batches_fake = []
    for _, inpainted_image in tqdm(dataloader, desc=f'FID - Fake Data Feature Extraction', total=len(dataloader)):
        image_batch = torch.Tensor(inpainted_image).to(device)
        inception_feature_batch = get_inception_features(image_batch)
        inception_feature_batches_fake.append(inception_feature_batch)
    inception_features_fake = np.concatenate(inception_feature_batches_fake)

    inception_feature_batches_real = []
    for target_image, _ in tqdm(dataloader, desc=f'FID - Real Data Feature Extraction', total=len(dataloader)):
        image_batch = torch.Tensor(target_image).to(device)
        inception_feature_batch = get_inception_features(image_batch)
        inception_feature_batches_real.append(inception_feature_batch)
    inception_features_real= np.concatenate(inception_feature_batches_real)
    
    mu_fake, sigma_fake = inception_features_fake.mean(axis=0), cov(inception_features_fake, rowvar=False)
    mu_real, sigma_real = inception_features_real.mean(axis=0), cov(inception_features_real, rowvar=False)
    ssdiff = np.sum((mu_fake - mu_real)**2.0)
    cov_mean = sqrtm(sigma_fake.dot(sigma_real))
    if iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    frechet_inception_distance = ssdiff + trace(sigma_fake + sigma_real - 2.0 * cov_mean)
    return frechet_inception_distance

class InferenceDataset(Dataset):
    def __init__(self, inference_dir, eval_resolution=256):
        self.inference_dir = inference_dir
        self.eval_resolution = eval_resolution
        self.file_names = os.listdir(inference_dir)
        self.ids = list({file_name.split("_")[0] for file_name in self.file_names})

    def __len__(self):
        return len(self.ids)

    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.eval_resolution,self.eval_resolution), Image.BILINEAR)
        img = np.array(img, dtype=float) / 255
        img = np.moveaxis(img, [0,1,2], [1,2,0])
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, idx):
        im_id = self.ids[idx]
        target_image = self.read_image(f"{self.inference_dir}/{im_id}_target_image.jpg")
        inpainted_image = self.read_image(f"{self.inference_dir}/{im_id}_inpainted.jpg")
        return target_image, inpainted_image
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        default=64,
        help="Bath size of Inception v3 forward pass",
    )
    args = parser.parse_args()

    dataset = InferenceDataset(args.inference_dir, eval_resolution=256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    fid = get_frechet_inception_distance(dataloader)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    fid_str = f"FID: {fid}"
    output_path = os.path.join(output_dir, f"fid_{dataset.eval_resolution}.txt")
    f = open(output_path, "w")
    f.write(fid_str)
    f.close()

    print(fid_str)





