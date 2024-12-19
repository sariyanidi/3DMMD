
# Installation

## 1) Download morphable models

You need to obtain the Basel Face Model (BFM'09) and the Expression Model through the links below:
**Models**
* Basel Face Model (BFM'09): [click here](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads) to obtain the Basel Face Model from the University of Basel
* Expression Model: Download the expression model (the Exp_Pca.bin) file from [this link](https://github.com/Juyong/3DFace)

Once you download both Basel Face Model (`01_MorphableModel.mat`) and the Expression Model (`Exp_Pca.bin`), copy theminto the `data/raw` directory. Specifically, these files should be in the following locations:

```
data/raw/01_MorphableModel.mat
data/raw/Exp_Pca.bin
```

## 2) Installation

```
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install torch==2.1.0

git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
python setup.py install
cd ..
pip install kornia
pip install opencv-python
pip install scikit-image
pip install pandas
pip install torchvision==0.16.0
pip install matplotlib
pip install ninja

git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/

python prepare_BFM.py

mkdir checkpoints
mkdir models/checkpoints
wget https://sariyanidi.com/dbox/3DIlite/backbone.pth -P ./checkpoints/
wget https://sariyanidi.com/dbox/3DIlite/medium_model15.00combined_celeb_ytfacesresnet50139979True1e-05-2-BFMmm-23660UNL_STORED.pth -P ./checkpoints/
wget https://sariyanidi.com/dbox/3DIlite/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-052True139979_V2.pth -P ./checkpoints/
wget https://sariyanidi.com/dbox/3DIlite/resnet50-0676ba61.pth -P ./models/checkpoints/
```

## 3) Test
```
mkdir testoutput
python process_video.py
```
