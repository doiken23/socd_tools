import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = {1: 1, 5: 2, 10: 3, 12: 4}  # Building  # Pole  # Car  # Traffic sign

LABEL_NAMES = ["building", "pole", "car", "traffic_sign"]

COLOR_PALETTE = {
    1: (77, 196, 255),  # buildings -> sky blue
    2: (3, 175, 122),  # pole -> green
    3: (255, 128, 130),  # car -> pink
    4: (153, 0, 153),  # traffic sign -> purple
}


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: str,
        split: str = "train",
        viewpoint: int = 1,
        mask: bool = False,
        transform: Optional[Any] = None,
    ):
        r"""
        Args:
            data (str): path of dataset root directory
            split (str): [train, val, test]
            viewpoint (int): viewpoint id ([1, 2, 3, 4])
            mask (bool): use instance mask
            transform (Optional[Any]): transform object
        """
        # set root directory path
        self.data = Path(data)

        # load annotation
        if split in ["train", "val", "test"]:
            file_name = "{}{}.json".format(split, viewpoint)
        else:
            raise ValueError("split should be chosen from [train, val, test].")

        # open annotation file
        with self.data.joinpath("annotations/", file_name).open("r") as f:
            annotations = json.load(f)["images"]

        # set annotation
        self.img_paths = []
        self.annotations = []
        for annotation in annotations:
            # check there is at least 1 bounding boxes
            anns = annotation["annotations"]
            for i, ann in enumerate(anns):
                box = np.array(ann["bbox"])
                if (box != 0).any() and ann["category_id"] in LABELS:
                    self.img_paths.append(annotation["path"])
                    self.annotations.append(annotation["annotations"])
                    break

        # set mask
        self.mask = mask

        # set transform
        self.transform = transform

        # label names
        self.label_names = LABEL_NAMES

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], Dict[str, torch.Tensor]]:
        # load image
        img_path = self.data.joinpath(self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")
        if self.mask:
            inst_path = self.data.joinpath(
                "instmasks",
                img_path.parent.stem,
                img_path.stem.replace("rgb", "inst") + "_processed.png",
            )
            inst_img = np.array(Image.open(inst_path).convert("RGB"), dtype=np.int64)
            inst_img = inst_img[:, :, 0] * 256 * 256 + inst_img[:, :, 1] * 256 + inst_img[:, :, 2]

        # set annotations
        anns = self.annotations[idx]
        instance_ids = []
        labels = []
        boxes = []
        areas = []
        for ann in anns:
            # boxes
            box = np.array(ann["bbox"])

            # old annotations
            if (box == 0).all() or ann["category_id"] not in LABELS:
                continue
            else:
                instance_ids.append(ann["id"])
                labels.append(LABELS[ann["category_id"]])
                boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                areas.append(box[2] * box[3])

        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        if self.mask:
            masks = (inst_img == np.array(instance_ids, dtype=np.int32)[:, None, None]).astype(
                np.uint8
            )
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # set annotation dictionary
        targets = {}
        targets["image_id"] = torch.tensor([idx])
        targets["labels"] = labels
        targets["boxes"] = boxes
        targets["area"] = areas
        targets["iscrowd"] = iscrowd
        if self.mask:
            targets["masks"] = masks

        # transform
        if self.transform is not None:
            img, targets = self.transform(img, targets)

        return img, targets


class CarlaObjcdDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: str,
        split: str = "train",
        viewpoint: int = 1,
        mask: bool = False,
        use_change_labels: bool = False,
        transform: Optional[Any] = None,
    ):
        r"""
        Args:
            data (str): path of dataset root directory
            split (str): train, val, test
            viewpoint (int): viewpoint id ([1, 2, 3, 4])
            mask (bool): use mask
            use_fundamental_matrix (bool): use F matrix or not
            transform (object): transform object
        """
        # set root directory path
        self.data = Path(data)
        self.viewpoint = viewpoint

        # load annotation
        if split in ["train", "val", "test"]:
            file_name = "{}{}.json".format(split, viewpoint)
        else:
            raise ValueError("split should be chosen from [train, val, test].")

        # load annotation file
        with self.data.joinpath("annotations", file_name).open("r") as f:
            annotations = json.load(f)["images"]

        # set annotation
        self.old_paths = []
        self.new_paths = []
        self.old_annotations = []
        self.new_annotations = []
        old_obj = False
        new_obj = False
        for i, annotation in enumerate(annotations):
            if i % 2 == 0:
                pair_id = annotation["pair_id"]

                # check old objects
                for ann in annotation["annotations"]:
                    box = np.array(ann["bbox"])
                    if (box != 0).any() and ann["category_id"] in LABELS:
                        old_obj = True
                        break

                # old_obj = len(annotation['annotations']) != 0
                old_path = annotation["path"]
                old_annotations = annotation["annotations"]
            else:
                # check pair id is same between old and new
                assert (
                    annotation["pair_id"] == pair_id
                ), "pair id is wrong (old = {}, new = {})".format(pair_id, annotation["pair_id"])

                # check new objects
                for ann in annotation["annotations"]:
                    box = np.array(ann["bbox"])
                    if (box != 0).any() and ann["category_id"] in LABELS:
                        new_obj = True
                        break

                # new_obj = len(annotation['annotations']) != 0
                new_path = annotation["path"]
                new_annotations = annotation["annotations"]

                # if both annotations are empty, skip
                if split == "train":
                    if old_obj and new_obj:
                        self.old_paths.append(old_path)
                        self.old_annotations.append(old_annotations)
                        self.new_paths.append(new_path)
                        self.new_annotations.append(new_annotations)
                else:
                    if old_obj or new_obj:
                        self.old_paths.append(old_path)
                        self.old_annotations.append(old_annotations)
                        self.new_paths.append(new_path)
                        self.new_annotations.append(new_annotations)

                # reset the trigger
                old_obj = False
                new_obj = False
        assert len(self.old_paths) == len(self.new_paths)
        assert len(self.old_annotations) == len(self.new_annotations)

        # set mask
        self.mask = mask

        # use change labels or not
        self.use_change_labels = use_change_labels

        # set transform
        self.transform = transform

    def __len__(self) -> int:
        return len(self.old_paths)

    def __getitem__(self, idx: int):
        # load image
        old_img_path = self.data.joinpath(self.old_paths[idx])
        new_img_path = self.data.joinpath(self.new_paths[idx])
        old_img = Image.open(old_img_path).convert("RGB")
        new_img = Image.open(new_img_path).convert("RGB")
        if self.mask:
            old_inst_path = self.data.joinpath(
                "instmasks",
                old_img_path.parent.stem,
                old_img_path.stem.replace("rgb", "inst") + "_processed.png",
            )
            old_inst_img = np.array(Image.open(old_inst_path).convert("RGB"), dtype=np.int64)
            old_inst_img = (
                old_inst_img[:, :, 0] * 256 * 256
                + old_inst_img[:, :, 1] * 256
                + old_inst_img[:, :, 2]
            )
            new_inst_path = self.data.joinpath(
                "instmasks",
                new_img_path.parent.stem,
                new_img_path.stem.replace("rgb", "inst") + "_processed.png",
            )
            new_inst_img = np.array(Image.open(new_inst_path).convert("RGB"), dtype=np.int64)
            new_inst_img = (
                new_inst_img[:, :, 0] * 256 * 256
                + new_inst_img[:, :, 1] * 256
                + new_inst_img[:, :, 2]
            )

        # set annotations
        old_anns = self.old_annotations[idx]
        new_anns = self.new_annotations[idx]
        assert len(old_anns) == len(new_anns), "length of old and new annotations"
        old_node_num = 0
        new_node_num = 0
        matched_pairs = []
        old_instance_ids = []
        new_instance_ids = []
        old_labels = []
        new_labels = []
        old_boxes = []
        new_boxes = []
        old_area = []
        new_area = []
        old_change_labels = []
        new_change_labels = []
        for old_ann, new_ann in zip(old_anns, new_anns):
            assert old_ann["match_id"] == new_ann["match_id"], "old id: {}, new id: {}".format(
                old_ann["match_id"], new_ann["match_id"]
            )

            if old_ann["category_id"] not in LABELS and new_ann["category_id"] not in LABELS:
                continue

            # boxes
            old_box = np.array(old_ann["bbox"])
            new_box = np.array(new_ann["bbox"])

            # match or not
            if old_ann["match"] and new_ann["match"]:
                assert (not (old_box == 0).all()) and (not (new_box == 0).all())
                matched_pairs.append((old_node_num, new_node_num))
                if old_ann["is_changed"] and new_ann["is_changed"]:
                    old_change_labels.append(1)
                    new_change_labels.append(1)
                else:
                    old_change_labels.append(0)
                    new_change_labels.append(0)
            elif (old_box != 0).any():
                matched_pairs.append((old_node_num, -1))
                if old_ann["is_changed"]:
                    old_change_labels.append(1)
                else:
                    old_change_labels.append(0)
            elif (new_box != 0).any():
                matched_pairs.append((-1, new_node_num))
                if new_ann["is_changed"]:
                    new_change_labels.append(1)
                else:
                    new_change_labels.append(0)

            # old annotations
            if (old_box != 0).any():
                old_node_num += 1
                old_instance_ids.append(old_ann["id"])
                old_labels.append(LABELS[old_ann["category_id"]])
                old_boxes.append(
                    [old_box[0], old_box[1], old_box[0] + old_box[2], old_box[1] + old_box[3]]
                )
                old_area.append(old_box[2] * old_box[3])

            # new annotations
            if (new_box != 0).any():
                new_node_num += 1
                new_instance_ids.append(new_ann["id"])
                new_labels.append(LABELS[new_ann["category_id"]])
                new_boxes.append(
                    [new_box[0], new_box[1], new_box[0] + new_box[2], new_box[1] + new_box[3]]
                )
                new_area.append(new_box[2] * new_box[3])

        old_labels = torch.as_tensor(old_labels, dtype=torch.int64)
        new_labels = torch.as_tensor(new_labels, dtype=torch.int64)
        old_boxes = torch.as_tensor(old_boxes, dtype=torch.float32)
        new_boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
        old_area = torch.as_tensor(old_area, dtype=torch.float32)
        new_area = torch.as_tensor(new_area, dtype=torch.float32)
        # old_change_labels = torch.as_tensor(old_change_labels, dtype=torch.int64)
        # new_change_labels = torch.as_tensor(new_change_labels, dtype=torch.int64)
        if self.mask:
            old_masks = (
                old_inst_img == np.array(old_instance_ids, dtype=np.int32)[:, None, None]
            ).astype(np.uint8)
            new_masks = (
                new_inst_img == np.array(new_instance_ids, dtype=np.int32)[:, None, None]
            ).astype(np.uint8)
            old_masks = torch.as_tensor(old_masks, dtype=torch.uint8)
            new_masks = torch.as_tensor(new_masks, dtype=torch.uint8)

        # set annotation dictionary
        old_targets = {}
        old_targets["image_id"] = torch.tensor([2 * idx])
        old_targets["labels"] = old_labels
        old_targets["boxes"] = old_boxes
        old_targets["area"] = old_area
        if self.mask:
            old_targets["masks"] = old_masks
        new_targets = {}
        new_targets["image_id"] = torch.tensor([2 * idx + 1])
        new_targets["labels"] = new_labels
        new_targets["boxes"] = new_boxes
        new_targets["area"] = new_area
        if self.mask:
            new_targets["masks"] = new_masks

        # set permutation matrix
        s = torch.zeros(old_node_num + 1, new_node_num + 1, dtype=torch.int64)
        for matched_pair in matched_pairs:
            s[matched_pair[0], matched_pair[1]] = 1

        # transform
        if self.transform is not None:
            old_img = self.transform(old_img)
            new_img = self.transform(new_img)

        outputs = [old_img, new_img, old_targets, new_targets, s]
        if self.use_change_labels:
            assert len(old_change_labels) == old_node_num
            assert len(new_change_labels) == new_node_num
            outputs.append(old_change_labels)
            outputs.append(new_change_labels)
        return tuple(outputs)


class CarlaCdDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: str,
        split: str = "train",
        viewpoint: int = 1,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        r"""
        Args:
            data (str): path of dataset root directory
            split (str): train, val, test
            viewpoint (int): viewpoint id ([1, 2, 3, 4])
            transform (object): transform object
        """
        # set root directory path
        self.data = Path(data)

        # load annotation
        if split in ["train", "val", "test"]:
            file_name = "{}{}.json".format(split, viewpoint)
        else:
            raise ValueError("split should be chosen from [train, val, test].")

        # load annotation file
        with self.data.joinpath("annotations", file_name).open("r") as f:
            annotations = json.load(f)["images"]

        # set annotation
        self.old_paths = []
        self.new_paths = []
        old_obj = False
        new_obj = False
        for i, annotation in enumerate(annotations):
            if i % 2 == 0:
                pair_id = annotation["pair_id"]

                old_path = annotation["path"]

                # check old objects
                if split == "test":
                    for ann in annotation["annotations"]:
                        box = np.array(ann["bbox"])
                        if (box != 0).any() and ann["category_id"] in LABELS:
                            old_obj = True
                            break
            else:
                # check pair id is same between old and new
                assert (
                    annotation["pair_id"] == pair_id
                ), "pair id is wrong (old = {}, new = {})".format(pair_id, annotation["pair_id"])

                new_path = annotation["path"]

                # check new objects
                if split == "test":
                    for ann in annotation["annotations"]:
                        box = np.array(ann["bbox"])
                        if (box != 0).any() and ann["category_id"] in LABELS:
                            new_obj = True
                            break

                # hold paths
                if split == "test":
                    if old_obj or new_obj:
                        self.old_paths.append(old_path)
                        self.new_paths.append(new_path)
                    old_obj = False
                    new_obj = False
                else:
                    self.old_paths.append(old_path)
                    self.new_paths.append(new_path)
        assert len(self.old_paths) == len(self.new_paths)

        # set transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.old_paths)

    def __getitem__(self, idx: int):
        # load image
        old_img_path = self.data.joinpath(self.old_paths[idx])
        new_img_path = self.data.joinpath(self.new_paths[idx])
        old_img = Image.open(old_img_path).convert("RGB")
        new_img = Image.open(new_img_path).convert("RGB")

        # load change mask
        old_chmask_path = self.data.joinpath(
            "chmasks", old_img_path.parent.stem, old_img_path.stem.replace("_rgb", "") + ".png"
        )
        new_chmask_path = self.data.joinpath(
            "chmasks", new_img_path.parent.stem, new_img_path.stem.replace("_rgb", "") + ".png"
        )
        old_chmask = (np.array(Image.open(old_chmask_path)) == 255).astype(np.int32)
        new_chmask = (np.array(Image.open(new_chmask_path)) == 255).astype(np.int32)

        # semantic change mask
        old_label_path = self.data.joinpath(
            "semmasks", old_img_path.parent.stem, old_img_path.stem.replace("_rgb", "") + ".png"
        )
        new_label_path = self.data.joinpath(
            "semmasks", new_img_path.parent.stem, new_img_path.stem.replace("_rgb", "") + ".png"
        )
        old_label = np.array(Image.open(old_label_path), np.int32)
        new_label = np.array(Image.open(new_label_path), np.int32)
        old_semantic_chmask = np.zeros_like(old_label)
        new_semantic_chmask = np.zeros_like(new_label)
        for original_label, label in LABELS.items():
            old_semantic_chmask += (old_chmask * (old_label == original_label) * label).astype(
                np.int32
            )
            new_semantic_chmask += (new_chmask * (new_label == original_label) * label).astype(
                np.int32
            )
        old_semantic_chmask = Image.fromarray(old_semantic_chmask)
        new_semantic_chmask = Image.fromarray(new_semantic_chmask)

        # transform
        if self.transform is not None:
            old_img = self.transform(old_img)
            new_img = self.transform(new_img)

        if self.target_transform is not None:
            old_semantic_chmask = self.target_transform(old_semantic_chmask)
            new_semantic_chmask = self.target_transform(new_semantic_chmask)
            old_semantic_chmask = old_semantic_chmask.squeeze().to(torch.long)
            new_semantic_chmask = new_semantic_chmask.squeeze().to(torch.long)

        return old_img, new_img, old_semantic_chmask, new_semantic_chmask
