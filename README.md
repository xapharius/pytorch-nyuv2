# PyTorch NYUv2 Dataset Class
PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.  
Data sources available: RGB, Semantic Segmentation(13), Surface Normals, Depth Images.

Downloads data from:
- [RGB Train](http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz)
- [RGB Test](http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz)
- [Segmentation Train](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz)
- [Segmentation Test](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz)
- [Surface Normals](https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip)
- [Depth Images](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)

## Example
```
from nyuv2 import NYUv2
from torchvision import transforms

t = transforms.Compose([transforms.RandomCrop(400), transforms.RandomHorizontalFlip()])
NYUv2(root="/somepath/NYUv2", download=True, transform=t)
```
```
Dataset NYUv2
    Number of datapoints: 795
    Split: train
    Root Location: /somepath/NYUv2
    Transforms: Compose(
                    RandomCrop(size=(400, 400), padding=None)
                    RandomHorizontalFlip(p=0.5)
                    ToTensor()
                )
```

![NYUv2](https://user-images.githubusercontent.com/1637188/57874116-c6632000-7807-11e9-9d7c-8d3060fa48d7.png)

## Notes
- Applies the same transformation on all sources
- Always returns tensors

## Requirements
```
pytorch: 1.1.0
torchvision: 0.2.2
h5py: 2.9.0
```

