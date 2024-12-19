

# Download morphable models

```
data/raw/01_MorphableModel.mat
data/raw/Exp_Pca.bin
```


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

mkdir checkpoints
mkdir models/checkpoints
wget https://sariyanidi.com/dbox/3DIlite/backbone.pth -P ./checkpoints/
wget https://sariyanidi.com/dbox/3DIlite/medium_model15.00combined_celeb_ytfacesresnet50139979True1e-05-2-BFMmm-23660UNL_STORED.pth -P ./checkpoints/
wget https://sariyanidi.com/dbox/3DIlite/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-052True139979_V2.pth -P ./checkpoints/
wget https://sariyanidi.com/dbox/3DIlite/resnet50-0676ba61.pth -P ./models/checkpoints/
```

