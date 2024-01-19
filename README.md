# Deep learning models for dMMR/MSI prediction

## Sisältö / In this repository:

- ***MMR/***: ladattavat konvoluutioneuroverkkomallit ja python-tiedostot WSI-kuvien esikäsittelyä ja dMMR:n ennustamista varten / CNN-models for downloading and python-files for preprocessing of the WSI-files and dMMR prediction.
- ***readme.md***: ohjeet ja taustatiedot / instructions and background, english version follows the finnish one

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
