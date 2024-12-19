#!/usr/bin/env python
# coding: utf-8

"""
IN:
msis = python list of WSI names with MSI/dMMR
msss = python list of WSI names with MSS/pMMR
wsiroot = dir where all WSIs are
tileroot = dir where 20x patches are saved for tumor masking
maskroot = dir where tumor masks will be saved
rroot = main root where to save masks, 5x and 20x probs (as numpy-files)
(rroot should include folders: "masks", "5x" and "20x)
modelTUM = CNN model (.pt) which predicts tumor areas from 20x tiles
froot = dir where to save feature-arrays

def MMRfeatures(msis, msss, wsiroot, tileroot,rroot, modelTUM, froot):
    
    imgs = msis+msss

    for i in range(len(imgs)):
        
        start_time = time.monotonic()
        class_name = "dMMR_" if imgs[i] in msis else "pMMR_" if imgs[i] in msss else "unknown"
        wsi = wsirootroot+imgs[i]
        slide = openslide.OpenSlide(wsi)
        n, m = slide.level_dimensions[0][1], slide.level_dimensions[0][0]
        
        mag_keys = {
            'hamamatsu': 'openslide.objective-power',
            'aperio': 'aperio.AppMag'
        }
        
        vendor = slide.properties.get('openslide.vendor')
        key = mag_keys.get(vendor)
        mag = int(slide.properties.get(key, 20)) if key else 20
        
#         if (slide.properties['openslide.vendor']=='hamamatsu'):
#             if ('openslide.objective-power' not in slide.properties): mag = 20
#             else: mag = int(slide.properties['openslide.objective-power'])

#         if (slide.properties['openslide.vendor']=='aperio'):
#             if ('aperio.AppMag' not in slide.properties): mag = 20
#             else: mag = int(slide.properties['aperio.AppMag'])
    
        size = {20: 224, 40: 448}.get(mag)
        
        tum5xroot = rroot+"5x/"
        tum20xroot = rroot+"20x/"
        
        done_imgs = set(os.listdir(rroot))
        
        # 1: TILE the entire slide 20x for tumor mask
        print(imgs[i])
        os.mkdir(rroot+imgs[i])
        tileNEW(rroot, slide, n, m, size, imgs[i], normalizer1, normalizer2, mag)
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Tiling time: {epoch_mins}m {epoch_secs}s')

        # 2: MASK tumor
        maskTUM(rroot+"masks/", rroot+imgs[i]+"/", n, m, size, modelTUM, imgs[i])
        print(imgs[i]+ "mask done.")
        
        # 3: TILE from two magnifications with same centre point
        print(imgs[i])
        TUM5xTUM20x(rroot, slide, rroot+"masks/"+imgs[i]+"MASK.npy", size, imgs[i])

        # 4: PREDICT mmr
        if (len(os.listdir(rroot+"5x/"+imgs[i])) > 30):
            
            MMR(rroot, rroot+"5x/"+imgs[i], rroot+"20x/"+imgs[i], model5x, model20x, imgs[i])
            f5x = [[] for _ in range(13)]
            f20x = [[] for _ in range(13)]
            percs = [99.75, 99.5, 99, 90, 80, 10]
            probs = [0.999, 0.99, 0.9]
            probs2 = [0.001, 0.01, 0.1]

            preds5x = np.load(root+"probs5x/"+imgs[i]+".npy")
            preds20x = np.load(root+"probs20x/"+imgs[i]+".npy")
            preds5x = list(preds5x[:,0])
            preds20x = list(preds20x[:,0])

            count = len(preds5x)

            preds5x = np.array(preds5x)
            preds20x = np.array(preds20x)
            
            f5[0] = np.median(preds5x)
            f20[0] = np.median(preds20x)

            for j in range(len(percs)):

                f5[j+1] = np.percentile(preds5x, percs[j])
                f20[j+1] = np.percentile(preds20x, percs[j])

            for j in range(len(probs)):

                f5[j+7] = sum(1 for value in preds5x if value > probs[j])
                f20[j+7] = sum(1 for value in preds20x if value > probs[j])

                f5[j+7] = f5[j+7] / count
                f20[j+7] = f20[j+7] / count

            for j in range(len(probs2)):

                f5[j+10] = sum(1 for value in preds5x if value < probs2[j])
                f20[j+10] = sum(1 for value in preds20x if value < probs2[j])

                f5[j+10] = f5[j+10] / count
                f20[j+10] = f20[j+10] / count

            row = f5+f20
            print(row)
            
            row = np.array(row)
            
            np.save(froot+imgs[i]+"FEATURES.npy", row)
            print("Features saved.")

        else: 
            print("Less than 30 tumor tiles, MMR not predicted from " +imgs[i]+ ".")
            return
         
