# CAFS

## Getting Started

PASCAL VOC 2012 download link : UniMatch [github](https://github.com/LiheYoung/UniMatch)

```angular2html
[Your Pascal Path]
  ├── JPEGImages
  └── SegmentationClass
```

Pretrained Backbone : [ResNet101](https://drive.google.com/file/d/126ZzFt8PQ0KX7dvKCn-ZSeKb468mZOyj/view?usp=share_link), [MiT-B4](https://drive.google.com/file/d/1Gn0QT7-SgT3k20JtSX7nyIOQJaEPcbQT/view?usp=share_link)

Eval Weight : [link](https://drive.google.com/drive/folders/11_MzauUu0de4NWCb0D4IpGtVcXLO1hSc?usp=share_link)

```angular2html
CAFS
├── pretrained
│   ├── resnet101.pth
│   └── mit_b4.pth
└── cafs_pretrained
    ├── 1_CAFS_1_4_resnet101_78.04.pth
    ├── 2_CAFS_1_4_resnet101_78.32.pth
    └── 3_CAFS_1_4_resnet101_78.05.pth
```
## Config
You can control our methods in the [config file](https://github.com/anonymous1253/CAFS/blob/main/configs/pascal.yaml#L24-L29).

## Train
```bash
sh tool/train.sh <num_gpu> <port>

# ex : sh tool/train.sh 4 23500
```

## Eval
```bash
sh tool/eval.sh <num_gpu> <port>

# ex : sh tool/eval.sh 4 23500
```

## Paper Result

| Method           |  1/4 (2646) |
| -----------------| ----------- |
| ST++             | 76.6        |
| UniMatch         | 77.2        |
| -----------------| ----------- |
| CAFS-V3(ours)    | 78.0        |
| CAFS-B4(ours)    | 79.9        |


## Re implementation result

| Method              |  1/4 (2646) |
| --------------------| ----------- |
| CAFS-V3(re-imp-1)   | 78.04       |
| CAFS-V3(re-imp-2)   | 78.32       |
| CAFS-V3(re-imp-3)   | 78.05       |
| --------------------| ----------- |
| CAFS-B4(re-imp-1)   | TODO        |
| CAFS-B4(re-imp-2)   | TODO        |
| CAFS-B4(re-imp-3)   | TODO        |

Re implementation log files [link](https://drive.google.com/drive/folders/1VyvRo3unSsrIxthJ0eq1TaJIxvpothQn?usp=share_link)
