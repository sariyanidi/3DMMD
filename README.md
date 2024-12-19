torch

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





#########


python create_imagelists.py /online_data/face/yt_faces2/yt_cropped2 7 170 0
python create_imagelists.py /online_data/face/yt_faces2/yt_cropped2 7 341 170


./fit_to_multiframe /home/sariyanide/car-vision/cuda/3DI/scripts/imlists/faceyt_cropped2.list7_0-170.txt ./configs/BFMmm-19830.cfg1.global4.txt 30 /offline_data/face/yt_faces2/3DI
./fit_to_multiframe /home/sariyanide/car-vision/cuda/3DI/scripts/imlists/faceyt_cropped2.list7_170-341.txt ./configs/BFMmm-19830.cfg1.global4.txt 30 /offline_data/face/yt_faces2/3DI



python scripts/cropper.py
python scripts/renamer.py

