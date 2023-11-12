import numpy as np
from glob import glob
import cv2
import os
from math import floor, ceil

os.makedirs('cropped', exist_ok=True)

imfs = glob('*jpg')
imfs.sort()
for imf in imfs:

    bimf = os.path.basename(imf)
    dstim_path = f'cropped/{bimf}'

    if os.path.exists(dstim_path):
        continue

    im = cv2.imread(imf)
    lf = imf.replace('.jpg', '.txt')
    lmks = np.loadtxt(lf)
    xs = lmks[::2]
    ys = lmks[1::2]
    xmin = floor(xs.min())
    xmax = ceil(xs.max())
    ymin = floor(ys.min())
    ymax = ceil(ys.max())

    w = xmax-xmin
    h = ymax-ymin
    s = max(w, h)

    xmin -= int(s/2)
    xmax += int(s/2)
    ymin -= int(s/2)
    ymax += int(s/2)
    w = xmax-xmin
    h = ymax-ymin
    s = max(w, h)

    xs = xs.reshape(-1,1)
    ys = ys.reshape(-1,1)


    # r = cv2.Rect(xmin, ymin, w, h)
    im = im[ymin:ymin+s, xmin:xmin+s, :]
    cs = im.shape[0]
    im = cv2.resize(im, (224, 224))
    lmks = np.concatenate((xs-xmin, ys-ymin), axis=1)
    lmks *= 224.0/cs

    cv2.imwrite(dstim_path, im)
    
    np.savetxt(f'cropped/{bimf.replace(".jpg", ".txt")}', lmks)

