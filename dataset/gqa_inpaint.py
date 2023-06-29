import json
import os
import random
from typing import Tuple
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset

RELATIONS = {
    "above",
    "at",
    "behind",
    "below",
    "beneath",
    "beside",
    "by",
    "in",
    "in front of",
    "inside",
    "near",
    "next to",
    "on",
    "on the back of",
    "on the front of",
    "on the side of",
    "on top of",
    "to the left of",
    "to the right of",
    "under",
    "underneath",
}

ADD_RELATIONS = {relation: relation for relation in RELATIONS}

REMOVE_RELATIONS = {
    **ADD_RELATIONS,
    "to the left of": "at the left of",
    "to the right of": "at the right of",
    "to the left": "at the left",
    "to the center": "at the center",
    "to the right": "at the right",
}

INSTRUCTION_RELATIONS = {
    "add": ADD_RELATIONS,
    "remove": REMOVE_RELATIONS,
}

class GQAInpaintBase(Dataset):
    def __init__(
        self,
        images_root, # Directory of the original images.
        images_inpainted_root, # Directory of the inpainted images.
        masks_root, # Directory of the mask images.
        scene_json_path, # Path of the scene json file.
        test_instructions_path=None, # If provided, instructions are not generated randomly (for evaluating the models). 
        size=256, # Size of the images for resizing.
        interpolation="bilinear", # Interpolation method used for resizing. Options: ["linear","bilinear","bicubic","lanczos"]
        instruction_type="remove", # Type of the instructions. The pair of source and target images are generated accordingly. Options: ["add","remove"]
        irrelevant_text_prob=None, # Probability of generating irrelevant instructions (source and target images are kept the same).
        max_relations=1, # Maximum number of relations to be included in the generated instructions.
        simplify_augment=True # Generating instructions without any relation if possible with 0.5 probability.
    ):
        assert instruction_type in ["add", "remove"]
        assert max_relations >= 0

        self._scene_json_path = scene_json_path
        self.images_root = images_root
        self.images_inpainted_root = images_inpainted_root
        self.masks_root = masks_root

        self.instruction_type = instruction_type
        self.relations_mapping = INSTRUCTION_RELATIONS[self.instruction_type]
        
        # GQA-Inpaint dataset is generated originally for add instructions.
        # If instruction_type is set to remove, then the instructions are updated accordingly
        # and the pair of target and source images are swapped. 
        self.construct_scenes()

        self._length = len(self.target_images)
        self.labels = {
            "target_image_path": self.target_images,
            "source_image_path": self.source_images,
            "mask_path": self.masks
        }

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.max_relations = max_relations
        self.irrelevant_text_prob = irrelevant_text_prob
        self.simplify_augment = simplify_augment

        self.use_fixed_instructions = False
        if test_instructions_path is not None:
            self.use_fixed_instructions = True
            with open(test_instructions_path) as json_file:
                self.test_instructions = json.load(json_file)

    def construct_scenes(self):
        assert self._scene_json_path

        with open(self._scene_json_path) as texts_fp:
            texts = json.load(texts_fp)

        if self.instruction_type == "remove":
            texts = self.filter_scene_texts(texts)

        self.count_object_occurrences(texts)
        self.texts = texts

        self.construct_image_dirs()

    def count_object_occurrences(self, texts):
        object_occurrences = {}
        for image_id, image_graph in texts.items():
            object_occurrences[image_id] = {}
            for object_id, object_graph in image_graph["objects"].items():
                object_name = object_graph["name"]
                if object_name not in object_occurrences[image_id]:
                    object_occurrences[image_id][object_name] = 0
                object_occurrences[image_id][object_name] += 1
        self.object_occurrences = object_occurrences

    def construct_image_dirs(self):
        assert self.texts
        assert self.images_root
        assert self.images_inpainted_root

        source_images = []
        target_images = []
        masks = []
        ids = []

        for image_id in self.texts:
            objects_graph = self.texts[image_id]["objects"]
            objects = dict(filter(lambda object_tuple: object_tuple[1]["bidirected"], objects_graph.items()))

            for object_id, object_attrs in objects.items():
                if object_attrs["bidirected"]:
                    source_image_path = os.path.join(self.images_inpainted_root, image_id, object_id + ".jpg")
                    target_image_path = os.path.join(self.images_root, image_id + ".jpg")
                    mask_path = os.path.join(self.masks_root, image_id, object_id + ".jpg")

                    source_images.append(source_image_path)
                    target_images.append(target_image_path)
                    masks.append(mask_path)
                    ids.append(f"i{image_id}-o{object_id}")

        self.source_images = source_images
        self.target_images = target_images
        self.masks = masks
        self.ids = ids

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        w_start, w_end, h_start, h_end = self.get_cropped_dims(width=image.shape[1], height=image.shape[0])
        image = image[h_start:h_end, w_start:w_end]
        image = Image.fromarray(image)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.uint8).transpose(2,0,1)
        return image

    def preprocess_mask(self, mask_path):
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.uint8)
        w_start, w_end, h_start, h_end = self.get_cropped_dims(width=mask.shape[1], height=mask.shape[0])
        mask = mask[h_start:h_end, w_start:w_end]
        mask = Image.fromarray(mask)
        if self.size is not None:
            mask = mask.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        mask = np.array(mask).astype(np.float32) / 255
        mask = mask.round(decimals=0)
        return mask[None,:,:]

    # Returns object name enriched by a random attribute if any attribute is available.
    def add_attributes(self, object_id, objects_data):
        name = objects_data[object_id]["name"]
        attributes = objects_data[object_id].get("attributes")
        if attributes:
            random_attr = random.choice(attributes)
            if random_attr:
                name = random_attr + " " + name
        return name

    def __getitem__(self, i):
        sample = dict((k, self.labels[k][i]) for k in self.labels)
        target_image = self.preprocess_image(sample["target_image_path"])
        target_image = target_image.astype(np.float32) / 127.5 - 1.0
        source_image = self.preprocess_image(sample["source_image_path"])
        source_image = source_image.astype(np.float32) / 127.5 - 1.0
        mask = self.preprocess_mask(sample["mask_path"])
        im_id = self.ids[i]

        if self.use_fixed_instructions:
            text = self.test_instructions[im_id]
            if self.instruction_type=="remove":
                source_image, target_image = target_image, source_image
        else:
            is_irrelevant_text = False
            if self.irrelevant_text_prob and self.irrelevant_text_prob > random.uniform(0, 1):
                is_irrelevant_text = True
            
            # Changing source image path to select a random instruction
            if is_irrelevant_text:
                source_image_path = self.source_images[random.randint(0, self.__len__() - 1)]
            else:
                source_image_path = self.source_images[i]

            image_id, object_id = source_image_path.split(os.sep)[-2:]
            object_id = os.path.splitext(object_id)[0]
            scene = self.texts[image_id]
            scene_texts = []

            base_obj_name = scene["objects"][object_id]["name"]
            is_object_unique = self.object_occurrences[image_id][base_obj_name] == 1
            target_obj_name = self.add_attributes(object_id, scene["objects"])

            simplify_unique = False
            # Discarding the object relations with 0.5 probability
            if self.simplify_augment:
                simplify_unique = random.choice((True, False))
            simplify = is_object_unique and simplify_unique

            if simplify:
                relations = []
            else:
                relations = scene["objects"][object_id]["relations"].copy()
                random.shuffle(relations)
                # Restricting long instructions
                relations = relations[:self.max_relations]

            for relation in relations:
                ref_obj_name = self.add_attributes(relation["object"], scene["objects"])
                relation_name = self.relations_mapping.get(relation["name"], relation["name"])
                text = f"{relation_name} the {ref_obj_name}"
                scene_texts.append(text)

            if not len(scene_texts) and not simplify:
                image_relative_relation = scene["image_relative_positions"][object_id]
                scene_texts.append(image_relative_relation)

            # Combining the relations
            text = " and ".join(scene_texts)

            if self.instruction_type=="remove":
                source_image, target_image = target_image, source_image
                text = np.str_(f"{self.instruction_type} the {target_obj_name} {text}")
            else:
                text = np.str_(f"{self.instruction_type} a {target_obj_name} {text}")

            if is_irrelevant_text:
                target_image = np.copy(source_image)

        sample["target_image"] = target_image
        sample["source_image"] = source_image
        sample["mask"] = mask
        sample["text"] = text.strip()
        sample["id"] = im_id

        return sample

    def get_object_region(self, image_width: int, object_box: Tuple[int, int, int, int]) -> str:
        split_point_1 = int(image_width // 3)
        split_point_2 = 2 * split_point_1

        x_min, _, x_max, _ = object_box
        obj_w = x_max - x_min

        if x_max <= split_point_1:
            return "left"
        if x_min <= split_point_1:
            if x_max >= split_point_2:
                return "center"
            rem = split_point_1 - x_min
            if rem >= obj_w / 2:
                return "left"
            else:
                return "center"
        if x_max <= split_point_2:
            return "center"
        if x_min <= split_point_2:
            rem = split_point_2 - x_min
            if rem >= obj_w / 2:
                return "center"
            else:
                return "right"
        return "right"

    def get_cropped_dims(self, width: int, height: int) -> Tuple[int, int, int, int]:
        crop_size = min(width, height)
        w_start = (width - crop_size) // 2
        w_end = (width + crop_size) // 2
        h_start = (height - crop_size) // 2
        h_end = (height + crop_size) // 2
        return w_start, w_end, h_start, h_end

    def get_cropped_bbox(self, image_scene, object_id) -> Tuple[int, int, int, int]:
        object_bbox = image_scene["objects"][object_id]["bbox"].copy()
        image_size_orig = (image_scene["width"], image_scene["height"])
        width, height = image_size_orig
        min_size = min(image_size_orig)
        object_bbox[0] -= (width - min_size) // 2
        object_bbox[1] -= (height - min_size) // 2
        object_bbox[2] -= (width - min_size) // 2
        object_bbox[3] -= (height - min_size) // 2
        object_bbox = np.clip(object_bbox, 0, min_size)
        return object_bbox

    def get_image_relative_position(self, image_scene, object_id) -> str:
        w_start, w_end, _, _ = self.get_cropped_dims(width=image_scene["width"], height=image_scene["height"])
        image_w = w_end - w_start
        bbox = self.get_cropped_bbox(image_scene, object_id)
        object_region = self.get_object_region(image_width=image_w, object_box=bbox)
        image_relative_relation = f"to the {object_region}"
        image_relative_relation = self.relations_mapping.get(image_relative_relation, image_relative_relation)
        return image_relative_relation

    def get_by_id(self, img_id):
        i = None
        for img_index in range(self.__len__()):
            if self.ids[img_index] == img_id:
                i = img_index
                break
        if i is None:
            raise Exception("Image is not found!")
        return self.__getitem__(i)

    # Filtering the scene texts based on object uniquness. The following filters are applied to the scene text:
    #     1.  If number of instance types for a given object type is greater than 2, then the scene is deleted.
    #     2.  If number of instance types are equal to 2, then their image relative position defines their uniqueness. 
    #         If their position is different, the scene is kept, else it is deleted.
    #     3.  If number of instance type is smaller than 2, it is considered as unique and the scene is kept.
    def filter_scene_texts(self, scene_texts):
        def filter_instances(image_id, item):
            _, object_ids = item
            if len(object_ids) < 2:
                return True
            elif len(object_ids) > 2:
                return False
            else:
                pos_1 = scene_texts[image_id]["image_relative_positions"][object_ids[0]]
                pos_2 = scene_texts[image_id]["image_relative_positions"][object_ids[1]]
                return pos_1 != pos_2
            
        filtered_texts = {}
        for image_id, scene in scene_texts.items():
            object_occurrences = {}
            object_id_to_name = {}
            included_objects = []

            for object_id, object_data in scene["objects"].items():
                object_name = object_data["name"]
                if object_name not in object_occurrences:
                    object_occurrences[object_name] = []

                object_occurrences[object_name].append(object_id)
                object_id_to_name[object_id] = object_name

            object_occurrences = dict(filter(lambda item: filter_instances(image_id, item), object_occurrences.items()))

            for object_name, object_ids in object_occurrences.items():
                included_objects.extend(object_ids)

            objects = {
                object_id: {
                    **scene["objects"][object_id],
                    "relations": list(
                        filter(
                            lambda x: (
                                x["object"] in included_objects
                                and object_id_to_name[x["object"]]
                                != object_id_to_name[object_id]
                            ),
                            scene["objects"][object_id]["relations"]
                        )
                    ),
                    "bbox": self.get_cropped_bbox(scene, object_id)
                }
                for object_id in included_objects
            }

            image_relative_positions = dict(
                filter(
                    lambda x: x[0] in included_objects,
                    scene_texts[image_id]["image_relative_positions"].items(),
                )
            )

            image_relative_positions = {
                object_id: self.get_image_relative_position(scene, object_id)
                for object_id in image_relative_positions
            }

            assert len(image_relative_positions) == len(objects)

            if included_objects:
                filtered_texts[image_id] = {
                    **scene,
                    "objects": objects,
                    "image_relative_positions": image_relative_positions,
                }

        return filtered_texts


class GQAInpaintTrain(GQAInpaintBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GQAInpaintTest(GQAInpaintBase):
    def __init__(self, **kwargs):
        super().__init__(irrelevant_text_prob=None, **kwargs)
