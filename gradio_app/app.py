import argparse
import gradio as gr
import numpy as np
import torch
from PIL import Image
import gradio_app.constants as constants
import gradio_app.utils as utils
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

MODEL, DEVICE = None, None

def inference(image: np.ndarray, instruction: str, center_crop: bool):
    if not instruction.lower().startswith("remove the"):
        raise gr.Error("Instruction should start with 'Remove the' !")
    image = Image.fromarray(image)
    cropped_image, image = utils.preprocess_image(image, center_crop=center_crop)
    output_image = MODEL.inpaint(image, instruction, num_steps=50, device=DEVICE, return_pil=True, seed=0)
    return cropped_image, output_image

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
        "--on_cpu",
        action='store_true',
        help="Running the inference code on CPU",
    )
    args = parser.parse_args()

    DEVICE = "cpu" if args.on_cpu else "cuda"

    parsed_config = OmegaConf.load(args.config)
    MODEL = instantiate_from_config(parsed_config["model"])
    model_state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    MODEL.load_state_dict(model_state_dict)
    MODEL.eval()
    MODEL.to(DEVICE)

    sample_image, sample_instruction, sample_step = constants.EXAMPLES[3]

    gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(type="numpy", value=sample_image, label="Source Image").style(
                height=256
            ),
            gr.Textbox(
                label="Instruction",
                lines=1,
                value=sample_instruction,
            ),
            gr.Checkbox(value=True, label="Center Crop", interactive=False),
        ],
        outputs=[
            gr.Image(type="pil", label="Cropped Image").style(height=256),
            gr.Image(type="pil", label="Output Image").style(height=256),
        ],
        allow_flagging="never",
        examples=constants.EXAMPLES,
        cache_examples=False,
        title=constants.TITLE,
        description=constants.DESCRIPTION,
    ).launch()
