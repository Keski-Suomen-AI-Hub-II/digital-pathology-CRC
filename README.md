# Deep learning models for dMMR/MSI prediction

## Sisältö / In this repository:

- ***models***: ladattavat konvoluutioneuroverkkomallit / CNN-models for downloading
- ***readme.md***: ohjeet ja taustatiedot / instructions and background, english version follows the finnish one
- ***.py***: ...

---

## dMMR / MSI
Taustaa suomeksi.

## Konvoluutioneuroverkkomallit

--Tähän TUM5x, TUM20, OTHER5x, multi-scale -malleista jotain ---

- **TUM5x.pt**: 
    - MobileNetV3Large-arkkitehtuuri
    - esiopetus: ImageNet
    - WSI-kuvat (H%E-värjäys) suolistosyövästä ("Suolisyöpä Keski-Suomessa 2000-2015", sekä Lynchin oireyhtymätapaukset)
    - 5x suurennos kasvainsolukosta
        - kasvainalueiden tunnistamiseen käytetty AI Hub I -hankkeessa kehitettyä mallia --> linkki
    - WSI-kohtainen tarkkuus: AUC = 93,4 %
    
- **TUM20x.pt**:
    
- **OTHER5x.pt**: 

- **Multi-scale.pt**:

Luokat:

- **0**: dMMR
- **1**: pMMR

### Syötekuvat

- syötekoko kaikille malleille 224 x 224 px<sup>2</sup>
- kasvainkuvatiilet on pilkottu siten, että 3/4 pilkottavasta kohdasta on kasvainsolukoksi tunnistetulla alueella 5x suurennoksessa
    - 5x-kuvien ja 20x-kuvien keskipiste on sama (ks. kuva x)
    - tile5x20x.py
- värinormalisointiin on käytetty Macenko-normalisointia (viite)
- kuvien normalisointiin käytettävät keskiarvot = [0.485, 0.456, 0.406] ja keskihajonnat = [0.229, 0.224, 0.225]

![image](https://github.com/Keski-Suomen-AI-Hub-II/digital-pathology-CRC/assets/64031196/cae1901e-e4ac-41a3-b9b5-ed439bc7faa9)

---

## dMMR/MSI

Background

## Convolutional neural network models

--- Three CNNs trained with histopathological images from colorectal cancer can be downloaded from the **models** -folder.

- **TUM5x.pt**:
    - .... 
- **TUM20x.pt**:
    
- **OTHER5x.pt**: 

- **Multi-scale.pt**:
    
Classes:

- **0**: dMMR
- **1**: pMMR
- 

### Input images

- ...
- 
