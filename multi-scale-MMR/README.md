# Deep learning models for dMMR/MSI prediction

## Sisältö / In this repository:

- ***/models/***: ladattavat konvoluutioneuroverkko- ja XGBoost-mallit dMMR:n ennustamista varten / CNN and XGBoost models for dMMR prediction.
- ***prepFEATURES.py***: Python-skripti piirteiden laskemiseen multi-scale-mallia varten / Python-script to prepare the probability features for the multi-scale model
- ***predMMR.py***: Python-skripti dMMR:n ennustamiseen multi-scale -mallilla / Python-script to make the dMMR prediction with multi-scale model
- ***readme.md***: ohjeet ja taustatiedot / instructions and background, ***english version follows the finnish one***

---

## dMMR / MSI

Molekyylitason profilointi on keskeinen osa useiden syöpätyyppien, kuten myös suolistosyövän (CRC) diagnostiikkaa.  tapauksissa. Yksi tärkeä geneettinen ominaisuus, joka CRC potilaiden kohdalla selvitetään, on DNA:n korjausmekanismin toimivuus (DNA mismatch repair, MMR). MMR-mekanismin viallisuus (MMR-deficient, dMMR) johtaa lyhyiden ei-koodaavien DNA-jaksojen eli mikrosatelliittijaksojen epästabiiliuteen (microsatellite unstable, MSI).  Syöpien, joilla todetaan MSI, on havaittu olevan muita immunogeenisempia ja omaavan hyvän vasteen immunologisille syöpälääkkeille, ja MSI-syövät eivät puolestaan välttämättä hyödy muille syöville kohdennetuista fluorourasiiliin pohjautuvista sytostaattihoidoista. CRC potilaiden kohdalla MSI on tärkeä selvittää myös sen vuoksi, että se voi olla merkki Lynchin oireyhtymästä (LS), joka on yleisin perinnöllinen CRC-tyyppi.

Vaikka MSI:n selvittäminen on potilaan hoidon kannalta kiistattoman tärkeää, kustannussyistä se voi jäädä selvittämättä, sillä perinteisillä menetelmillä se vaatii niin henkilö- kuin materiaaliresursseja. Tästä kirjastosta löytyy neljä erilaista mallia dMMR:n ennustamiseen suolistosyövän histopatologisista kuvista. Mallit on opetettu, validoitu ja testattu käyttäen "Suolistyöpä Keski-Suomessa 2000-2015" H&E värjättyjä histopatologisia WSI-kuvia sekä Lynch-oireyhtymä potilailta kerätyistä näytteistä skannattuja H&E värjättyjä WSI-kuvia.

## Malli

Malli koostuu sekä CNN- että XGBoost-kerroksista, arkkitehtuuri on esitetty alla olevasssa kuvassa. CNN-haarat perustuvat MobileNetV3-arkkitehtuuriin, joka on esiopetettu ImageNet-kuvakirjastolla ja mallit on opetettu käyttäen kuvien alueita, joissa esiintyy kasvainsolukkoa. Kasvainalueiden tunnistamiseen on käytetty AI Hub I-hankkeessa kehitettyä kasvain-strooma-mallia (MMR/models/TSR_model.pt). Kasvainsolukon maski tehdään 20x suurennoksesta.

<img width="1500" alt="graphicalabstract" src="https://github.com/user-attachments/assets/44e45f67-f87f-460e-bc63-e0affd74fd40" />

- **TUM5x.pt**
- **TUM20x.pt**

- **Multi-scale**:
    - WSI-kohtainen luokittelutarkkuus **AUCPR = 93.7 %**

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

## Model

The model consists of CNN and XGBoost layers, with the architecture illustrated in the figure below. The CNN branches are based on the MobileNetV3 architecture, which has been pre-trained on the ImageNet. The Models are trained using image regions containing tumor cells. The identification of tumor regions is carried out using the tumor-stroma model (MMR/models/TSR_model.pt) developed during the AI Hub I project. The tumor cell mask is generated from a 20x magnification image.

<img width="1500" alt="graphicalabstract" src="https://github.com/user-attachments/assets/44e45f67-f87f-460e-bc63-e0affd74fd40" />

- **TUM5x.pt**
- **TUM20x.pt**

- **Multi-scale**:
    - WSI-level accuracy **AUCPR = 93.7 %**

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
