# Deep learning models for dMMR/MSI prediction

## Sisältö / In this repository:

- ***MMR/***: ladattavat konvoluutioneuroverkkomallit ja python-tiedostot WSI-kuvien esikäsittelyä ja dMMR:n ennustamista varten / CNN-models for downloading and python-files for preprocessing of the WSI-files and dMMR prediction.
- ***readme.md***: ohjeet ja taustatiedot / instructions and background, ***english version follows the finnish one***

---

## dMMR / MSI

Molekyylitason profilointi on keskeinen osa useiden syöpätyyppien, kuten myös suolistosyövän (CRC) diagnostiikkaa.  tapauksissa. Yksi tärkeä geneettinen ominaisuus, joka CRC potilaiden kohdalla selvitetään, on DNA:n korjausmekanismin toimivuus (DNA mismatch repair, MMR). MMR-mekanismin viallisuus (MMR-deficient, dMMR) johtaa lyhyiden ei-koodaavien DNA-jaksojen eli mikrosatelliittijaksojen epästabiiliuteen (microsatellite unstable, MSI).  Syöpien, joilla todetaan MSI, on havaittu olevan muita immunogeenisempia ja omaavan hyvän vasteen immunologisille syöpälääkkeille, ja MSI-syövät eivät puolestaan välttämättä hyödy muille syöville kohdennetuista fluorourasiiliin pohjautuvista sytostaattihoidoista. CRC potilaiden kohdalla MSI on tärkeä selvittää myös sen vuoksi, että se voi olla merkki Lynchin oireyhtymästä (LS), joka on yleisin perinnöllinen CRC-tyyppi.

Vaikka MSI:n selvittäminen on potilaan hoidon kannalta kiistattoman tärkeää, kustannussyistä se voi jäädä selvittämättä, sillä perinteisillä menetelmillä se vaatii niin henkilö- kuin materiaaliresursseja. Tästä kirjastosta löytyy neljä erilaista mallia dMMR:n ennustamiseen suolistosyövän histopatologisista kuvista. Mallit on opetettu, validoitu ja testattu käyttäen "Suolistyöpä Keski-Suomessa 2000-2015" H&E värjättyjä histopatologisia WSI-kuvia sekä Lynch-oireyhtymä potilailta kerätyistä näytteistä skannattuja H&E värjättyjä WSI-kuvia.

## Konvoluutioneuroverkkomallit

Kaikkien mallit pohjautuvat MobileNetV3-arkkitehtuuriin, joka on esiopetettu ImageNet-kuvakirjastolla. Mallit ovat yksi- (TUM5x, TUM20x, OTHER5x) tai kaksihaaraisia (TUM5x-TUM20x). Kasvainalueiden tunnistamiseen on käytetty Ai Hub I-hankkeessa kehitettyä kasvain-strooma-mallia (MMR/models/TSR_model.pt). Kasvainsolukon maski tehdään 20x suurennoksesta.

- **TUM5x.pt**: 
    - tiilikohtainen luokittelutarkkuus 82,2 %
    - WSI-kohtainen luokittelutarkkuus: **AUC = 93,4 %**
    
- **TUM20x.pt**:
    - tiilikohtainen luokittelutarkkuus 78,6 %
    - WSI-kohtainen luokittelutarkkuus **AUC = 92,0 %**
    
- **OTHER5x.pt**: 
    - tiilikohtainen luokittelutarkkuus -
    - WSI-kohtainen luokittelutarkkuus **AUC = 83,0 %**

- **Multi-scale.pt**:
    - WSI-kohtainen luokittelutarkkuus **AUC = 93.0 %**

![image](https://github.com/Keski-Suomen-AI-Hub-II/digital-pathology-CRC/assets/64031196/eb8313b8-8c77-48c1-96be-d9171567ca01)


Luokat:

- **0**: dMMR
- **1**: pMMR

### Syötekuvat

- syötekoko kaikille malleille 224 x 224 px<sup>2</sup>
- kasvainkuvatiilet on pilkottu siten, että 3/4 pilkottavasta kohdasta on kasvainsolukoksi tunnistetulla alueella 5x suurennoksessa
    - 5x- ja 20x-kuvatiilien keskipiste on sama (ks. kuva x)
    - tile5x20x.py
- värinormalisointiin on käytetty Macenko-normalisointia (viite)
- kuvien normalisointiin käytettävät keskiarvot = [0.485, 0.456, 0.406] ja keskihajonnat = [0.229, 0.224, 0.225]

![image](https://github.com/Keski-Suomen-AI-Hub-II/digital-pathology-CRC/assets/64031196/cae1901e-e4ac-41a3-b9b5-ed439bc7faa9)

---

## dMMR/MSI

Molecular profiling is a central part of cancer diagnostics. Important molecular factors analysed depend on the disease type; for colorectal cancer (CRC) they include mutations in genes coding DNA repairing enzymes (DNA mismatch-repairing, MMR). If the tumor is found to be MMR-deficient (dMMR), it is considered to have microsatellite instability (MSI). Tumors with MSI are found to be more immunogenic and have a good response to immunologic cancer drugs compared to non-MSI tumors. MSI could also be one sign of a heritable type of CRC, the Lynch syndrome.

It is clinically really important to know the MSI status of a tumor, but the MSI screening takes material and labour costs, which is the reason why it is sometimes left unexamined. This library has four different models for predicting the dMMR of a digitized H&E-stained CRC specimen. Models are trained with Finnish CRC data and validated both internally and externally.

## Convolutional neural network models

All models are based on MobileNetV3-architecture, which is pre-trained with the ImageNet-image library. Models have one branch (TUM5x, TUM20x, OTHER5x) and two branches (TUM5x-TUM20x). The tumor areas are detected by a tumor-stroma-model developed in AI Hub I-project. The tumor mask is applied in magnification of 20x.

- **TUM5x.pt**:
    - tile-level classification accuracy 82.2 %
    - WSI-level accuracy **AUC = 93.4 %**
      
- **TUM20x.pt**:
    - tile-level classification accuracy 78.6 %
    - WSI-level accuracy **AUC = 92.0 %**
      
- **OTHER5x.pt**:
    - tile-level classification accuracy -
    - WSI-level accuracy **AUC = 83.0 %**
       

- **Multi-scale.pt**:
    - WSI-level accuracy **AUC = 93.0 %**

![image](https://github.com/Keski-Suomen-AI-Hub-II/digital-pathology-CRC/assets/64031196/eb8313b8-8c77-48c1-96be-d9171567ca01)

Classes:

- **0**: dMMR
- **1**: pMMR


### Input images

- input size of all models is 224 x 224 px<sup>2</sup>
- patching is accomplished with a sliding frame: if 3/4 of the frame in magnification of 5x is within the tumor mask, the patch is tiled
  - 5x and 20x tiles share the same centre point
  - tile5x20x.py
- Macenko's algorithm is applied in the color normalization of the image tiles
- as the models are pre-trained with ImageNet, the image tiles are normalized before training using the following means and standard deviations:
    - means = [0.485, 0.456, 0.406]
    - standard deviations = [0.229, 0.224, 0.225]
 

![image](https://github.com/Keski-Suomen-AI-Hub-II/digital-pathology-CRC/assets/64031196/cae1901e-e4ac-41a3-b9b5-ed439bc7faa9)
