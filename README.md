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

t = transforms.Compose([transforms.RandomCrop(400), transforms.ToTensor()])
NYUv2(root="/somepath/NYUv2", download=True, 
      rgb_transform=t, seg_transform=t, sn_transform=t, depth_transform=t)
```
```
Dataset NYUv2
    Number of datapoints: 795
    Split: train
    Root Location: /somepath/NYUv2
    RGB Transforms: Compose(
                        RandomCrop(size=(400, 400), padding=None)
                        ToTensor()
                    )
    Seg Transforms: Compose(
                        RandomCrop(size=(400, 400), padding=None)
                        ToTensor()
                    )
    SN Transforms: Compose(
                       RandomCrop(size=(400, 400), padding=None)
                       ToTensor()
                   )
    Depth Transforms: Compose(
                          RandomCrop(size=(400, 400), padding=None)
                          ToTensor()
                      )
```

![NYUv2](https://user-images.githubusercontent.com/1637188/57874116-c6632000-7807-11e9-9d7c-8d3060fa48d7.png)

## Notes
- Each source has its own transformation pipeline
- Downloads datasets only for tasks where the passed transform is not None.
- Do not flip surface normals, as the output would be incorrect without further
 processing
- Semantic Segmentation Classes: (0) background, (1) bed, (2) books, (3) ceiling, (4) chair, (5) floor, (6) furniture, (7) objects, (8) painting, (9) sofa, (10) table, (11) tv, (12) wall, (13) window

## Requirements
```
h5py: 2.9.0
pillow: 6.2.0
pytorch: 0.4.0
torchvision: 0.4.0
```

