# Crack Identification and Damage Evolution Analysis in Fabric Composites: A FRPs-DSQE Based Statistical Approach

# Dependencies
## Neural network framework
This implementation is built on Mask DINO [![MaskDINO](https://img.shields.io/badge/Built_with-MaskDINO-FF6F00?logo=github)](https://github.com/IDEA-Research/MaskDINO). Clone the official repository and install dependencies.

## Dataset Setup
The fiber composite XCT images segmentation requires our proprietary fiber composite dataset. You can freely download the dataset from **[here](https://figshare.com/projects/Crack_Identification_and_Damage_Evolution_Analysis_in_Fabric_Composites_A_FRPs-DSQE_Based_Statistical_Approach/245780)** under a CC BY-NC 4.0 license.

# Training
Following the network architecture implementation and dataset acquisition, co-locate the obtained dataset with the network architecture implementation within a unified project directory. The standardized hierarchical directory structure should be organized as follows:

```bash
your_project_root/
├── MaskDINO/
│   ├── configs/
│   ├── datasets/
│   │   ├── coco_fiber/
│   │   │   ├── annotations/
│   │   │   ├── train2017/
│   │   │   └── test2017/
│   │   └── cityscapes/
│   │       ├── gtFine/
│   │       └── leftImg8bit/
│   ├── train_net.py
└── ...  # 其他文件
```
              
