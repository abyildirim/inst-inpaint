from typing import Tuple
from PIL import Image
from torchvision.transforms import ToTensor

to_tensor = ToTensor()

def preprocess_image(
    image: Image, resize_shape: Tuple[int, int] = (256, 256), center_crop=True
):
    pil_image = image

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        pil_image = image.crop((left, top, right, bottom))

    pil_image = pil_image.resize(resize_shape)
    
    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0) * 2 - 1

    return pil_image, tensor_image