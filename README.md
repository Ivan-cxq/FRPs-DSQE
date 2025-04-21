# Crack Identification and Damage Evolution Analysis in Fabric Composites: A FRPs-DSQE Based Statistical Approach

# Dependencies
## Neural network framework
This implementation is built on Mask DINO [![MaskDINO](https://img.shields.io/badge/Built_with-MaskDINO-FF6F00?logo=github)](https://github.com/IDEA-Research/MaskDINO). Clone the official repository and install dependencies.

## Dataset Preparation
The fiber composite XCT images segmentation requires our proprietary fiber composite dataset. You can freely download the dataset from **[here](https://figshare.com/projects/Crack_Identification_and_Damage_Evolution_Analysis_in_Fabric_Composites_A_FRPs-DSQE_Based_Statistical_Approach/245780)** under a CC BY-NC 4.0 license.

# Training
## Dataset Setup
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
└── ... 
```
## Dataset Adaptation and Library Modification
### Cityscapes(semantic segmentation)
To accommodate our private composite materials dataset, we adapt the Cityscapes semantic segmentation annotation protocol with domain-specific modifications. After establishing the network dependencies, perform the following critical file replacements:

```bash
Override Cityscapes Scripts:
Replace cityscapesscripts in:
Anaconda/envs/your_maskdino_name/Lib/site-packages/
with the modified version from:
\Deep-UniSeg(Based on MaskDINO)\#modified library\cityscapesscripts

Update Detectron2 Core:
Overwrite detectron2-0.6-py3.9-win-amd64.egg using the patched version in:
\Deep-UniSeg(Based on MaskDINO)\#modified library\detectron2_patch
```
The annotation rules are as follows
```bash
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'air'                  ,  1 ,        0 , 'background'      , 1       , False        , False        , (128, 64,128) ),
    Label(  'matrix'               ,  2 ,        1 , 'component'       , 2       , False        , False        , (244, 35,232) ),
    Label(  'yarn_crack'           ,  3 ,        2 , 'fracture'        , 3       , False        , False        , ( 70, 70, 70) ),
    Label(  'matrix_crack'         ,  4 ,        3 , 'fracture'        , 3       , False        , False        , (102,102,156) ),
    Label(  'pore'                 ,  5 ,        4 , 'fracture'        , 3       , False        , False        , (190,153,153) ),
    Label(  'weft'                 ,  6 ,        5 , 'component'       , 2       , True         , False        , (153,153,153) ),
    Label(  'warp'                 ,  7 ,        6 , 'component'       , 2       , True         , False        , (210,210,210) ),
]
```

### COCO2017(Instance segmentation)
For instance segmentation tasks compliant with the COCO2017 standard, we implement custom dataset registration through official protocols by modifying maskdino/train_net.py. The registration methodology comprises the following steps:

```bash
from detectron2.data.datasets import register_coco_instances

register_coco_instances("fiber_train", {},
                        json_file=r"root\you_folder\MaskDINO_main\datasets\coco_fiber\annotations\instances_train2017.json",
                        image_root=r"root\you_folder\MaskDINO_main\datasets\coco_fiber\train2017")
register_coco_instances("fiber_val", {},
                        r"root\you_folder\MaskDINO_main\datasets\coco_fiber\annotations\instances_val2017.json",
                        r"root\you_folder\MaskDINO_main\datasets\coco_fiber\val2017")

MetadataCatalog.get("fiber_train").thing_classes = ['weft', 'warp']
MetadataCatalog.get("fiber_val").thing_classes = ['weft', 'warp']
```








