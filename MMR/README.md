## Preprocessing of the WSI files

All data for training and testing have to be tiled from a magnification of 20x and stain-normalized with Macenko's method.

## Predict dMMR

1. Create TUMOR mask (mask_TUM20x.py)
2. Tile WSIs from magnifications of 5x and 20x using the same center point (tile5x20x.py)
3. Predict dMMR (class: 0) with pred_MMR.py   

## Dependencies:

- Torch 2.1.0
- Torchvision 0.16.0
- OpenSlide 1.1.2
- OpenCV 4.5.2
- CUDA 12.3
