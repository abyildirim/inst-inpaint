import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from  torchvision.transforms import ToPILImage

import argparse
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion/gqa-inpaint-ldm-vq-f8-256x256.yaml",
        help="Path of the model config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/gqa_inpaint/ldm/model.ckpt",
        help="Path of the model checkpoint file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the inference results",
    )
    parser.add_argument(
        "--on_cpu",
        action='store_true',
        help="Running the inference code on CPU",
    )
    args = parser.parse_args()

    device = "cpu" if args.on_cpu else "cuda"
    outdir = args.outdir

    if os.path.isdir(outdir):
        raise Exception("Output directory already exists!")
    os.makedirs(outdir)

    parsed_config = OmegaConf.load(args.config)
    model = instantiate_from_config(parsed_config["model"])
    model_state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)

    dataset = instantiate_from_config(parsed_config["data"])
    dataset.setup()
    test_dataset = dataset.datasets["test"]
    dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False)

    to_pil = ToPILImage()

    for batch in tqdm(dataloader):
        source_image = batch["source_image"].to(device)
        target_image = batch["target_image"]
        instruction = batch["text"][0]
        im_id = batch["id"][0]
        
        inpainted_image = model.inpaint(source_image, instruction, num_steps=50, device=device, return_pil=True, seed=0)
        
        target_image_pil = to_pil((target_image[0] + 1) / 2)
        source_image_pil = to_pil((source_image[0] + 1) / 2)

        outpath = os.path.join(outdir, im_id)
        source_image_pil.save(outpath + "_source_image.jpg")
        target_image_pil.save(outpath + "_target_image.jpg")
        inpainted_image.save(outpath + "_inpainted.jpg")
        f = open(outpath + "_instruction.txt", "w")
        f.write(instruction)
        f.close()



