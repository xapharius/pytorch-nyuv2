"""
author: Mihai Suteu
date: 15/05/19
"""


import os
import sys
import h5py
import torch
import shutil
import random
import logging
import tarfile
import zipfile
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class NYUv2(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.

    ### Output
    All images are of size: 480 x 640

    1. RGB: 3 channels, without custom rgb_normalize the values will be in [0, 1]
    due to ToTensor().

    2. Semantic Segmentation: 1 channel with integers representing one of the 13
    classes.

    3. Surface Normals: 3 channels, with values in [0, 1] due to ToTensor()

    4. Depth Images: 1 channel with floats representing the distance in meters.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        rgb_transform=None,
        seg_transform=None,
        sn_transform=None,
        depth_transform=None,
        rgb: bool = True,
        segmentation: bool = True,
        surface_normal: bool = True,
        depth: bool = True,
    ):
        """
        Images will be automatically returned as tensors.
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images,
        needs to end in a ToTensor transformation as more operations are internally
        performed afterwards
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images,
        needs to end in a ToTensor transformation as more operations are internally
        performed afterwards
        :param rgb: load RGB images
        :param segmentation: load semantic segmentation images
        :param surface_normal: load surface_normal images
        :param depth: load depth images
        """
        super().__init__()
        self.root = root

        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.sn_transform = sn_transform
        self.depth_transform = depth_transform

        pre = post = None
        if depth_transform is not None:
            if isinstance(depth_transform, transforms.Compose):
                pre, post = _split_transforms(depth_transform)
            else:
                pre, post = depth_transform, None
        self._depth_pre_transform, self._depth_post_transform = pre, post

        self.train = train
        self._split = "train" if train else "test"

        self.rgb = rgb
        self.segmentation = segmentation
        self.surface_normal = surface_normal
        self.depth = depth

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        # rgb folder as ground truth
        self._files = os.listdir(os.path.join(root, f"{self._split}_rgb"))

    def __getitem__(self, index: int):
        folder = lambda name: os.path.join(self.root, f"{self._split}_{name}")
        imgs = dict()

        if self.rgb:
            imgs["rgb"] = Image.open(os.path.join(folder("rgb"), self._files[index]))

        if self.segmentation:
            imgs["seg13"] = Image.open(
                os.path.join(folder("seg13"), self._files[index])
            )

        if self.surface_normal:
            imgs["sn"] = Image.open(os.path.join(folder("sn"), self._files[index]))

        if self.depth:
            imgs["depth"] = Image.open(
                os.path.join(folder("depth"), self._files[index])
            )

        seed = random.randrange(sys.maxsize)
        if self.rgb and self.rgb_transform:
            random.seed(seed)
            imgs["rgb"] = self.rgb_transform(imgs["rgb"])

        if self.segmentation:
            if self.seg_transform:
                random.seed(seed)
                imgs["seg13"] = self.seg_transform(imgs["seg13"])
            if isinstance(imgs["seg13"], torch.Tensor):
                # ToTensor scales to [0, 1] by default
                imgs["seg13"] = (imgs["seg13"] * 255).long()
            else:
                logging.warning(
                    "Segmentation images should produce integer values - transform "
                    "pipeline should output a tensor for conversion to trigger"
                )

        if self.surface_normal and self.sn_transform:
            random.seed(seed)
            imgs["sn"] = self.sn_transform(imgs["sn"])

        if self.depth:
            if self.depth_transform:
                random.seed(seed)
                imgs["depth"] = self._depth_pre_transform(imgs["depth"])
                if isinstance(imgs["depth"], torch.Tensor):
                    # Additional transformation - decoding images to float32
                    imgs["depth"] = _rgba_to_float32(imgs["depth"] * 255)
                else:
                    logging.warning(
                        "Depth images require decoding - transform pipeline "
                        "should output a tensor for decoder to trigger"
                    )
                if self._depth_post_transform is not None:
                    imgs["depth"] = self._depth_post_transform(imgs["depth"])

        return list(imgs.values())

    def __len__(self):
        return len(self._files)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Seg Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.seg_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    SN Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.sn_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                for type_ in ["rgb", "seg13", "sn", "depth"]:
                    path = os.path.join(self.root, f"{split}_{type_}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError as e:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        download_rgb(self.root)
        download_seg(self.root)
        download_sn(self.root)
        download_depth(self.root)
        print("Done!")


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_sn(root: str):
    url = "https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip"
    train_dst = os.path.join(root, "train_sn")
    test_dst = os.path.join(root, "test_sn")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            _unpack(tar)
            if not os.path.exists(train_dst):
                _replace_folder(
                    os.path.join(root, "nyu_normals_gt", "train"), train_dst
                )
                _rename_files(train_dst, lambda x: x[1:])
            if not os.path.exists(test_dst):
                _replace_folder(os.path.join(root, "nyu_normals_gt", "test"), test_dst)
                _rename_files(test_dst, lambda x: x[1:])
            shutil.rmtree(os.path.join(root, "nyu_normals_gt"))


def download_depth(root: str):
    url = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = _float32_to_rgba(depths[i].T)
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img, mode="RGBA").save(save_path)


def _float32_to_rgba(arr: np.ndarray):
    """
    Encode depth image from float32 into rgba that can be saved to disk as png
    Shape: [H * W] -> [H * W * 4]
    Value: abcdefgh -> [ab, cd, ef, gh]
    """
    arr = (arr * 1e7).astype(np.uint32)
    res = np.stack(
        [
            (arr % 100).astype(np.uint8),
            ((arr // 1e2) % 100).astype(np.uint8),
            ((arr // 1e4) % 100).astype(np.uint8),
            ((arr // 1e6) % 100).astype(np.uint8),
        ]
    )
    res = res.transpose(1, 2, 0)
    return res


def _rgba_to_float32(arr: torch.Tensor):
    """
    Decode a depth image from rbga (png) to the original float values.
    Expects rbga value ranges: 0 to 255
    Shape: [4 * H * W] -> [H * W]
    Value: [ab, cd, ef, gh] -> abcdefgh
    """
    res = (arr[0] + arr[1] * 1e2 + arr[2] * 1e4 + arr[3] * 1e6) / 1e7
    res = res.unsqueeze(dim=0)
    return res


def __test_depth_conversion(mat_file: str):
    """
    Test whether depth encoding and decoding returns the original value
    :param mat_file: path to the official labelled dataset .mat file
    :return: None if passes, assert if fails
    """
    depths = h5py.File(mat_file, "r")["depths"]
    raw = depths[0].T  # image at index 0
    encoded = _float32_to_rgba(raw)
    pil = Image.fromarray(encoded, mode="RGBA")
    transformed = transforms.ToTensor()(pil) * 255
    decoded = _rgba_to_float32(transformed)
    assert torch.allclose(torch.tensor(raw), decoded)


def _split_transforms(compose) -> (transforms.Compose, transforms.Compose):
    """
    Split transforms pipeline into pre and post ToTensor()
    :param compose: a transforms pipeline
    :return: two composes, first one includes ToTensor if present
    """
    pre_tensor = True
    pre_compose = []
    post_compose = []
    for t in compose.transforms:
        if pre_tensor:
            pre_compose.append(t)
        else:
            post_compose.append(t)
        if isinstance(t, transforms.ToTensor):
            pre_tensor = False
    pre_compose = transforms.Compose(pre_compose)
    post_compose = transforms.Compose(post_compose)
    return pre_compose, post_compose
