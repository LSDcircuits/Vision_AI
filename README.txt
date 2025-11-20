introduction

This repo is for the Machine vision AI course assingment, where we train a AI to detect a papi light and make a script
which responses to the papi light status. 
a aditional note this worksapce is different than given in the usb by uni, it uses the same modules some updated for verions 
for drivers compatable with arm64, but the functionality is the same. abel the images with labelImg, -> run the trainng script -> use script to detect box -> use open cv to flag (H,L,G)

repository contents:

1. Download scripts for arm64 operating systems ( apple & raspberry pi ) // not reliable yet only tested on 1 pc
2. Pictures used for training where 6 of the most clear are used for val and the rest for training.
3. Training models trained using YOLO // to be expanded to more light settings
4. python scripts papi.py & papi_simple.py // the simple one is the more approachable but is too robust for ratios in different light settings

The python scripts only use the models trained to detect and to get openCV working within a frame, where afterwards OpenCV analyzzes the bitmap to give a results based on ratios measured, 
specifically red to white ratio. the video feed is fed through a link which can be changed to USB, video or anything which streams just 1 line.


APT (system packages):

python3-pyqt5
pyqt5-dev-tools
qt5-image-formats-plugins
python3-lxml



python modules:

numpy<2
opencv-python==4.7.0.72
ffmpeg-python
tqdm
matplotlib
pandas
requests
pyyaml
rich
ultralytics
torch
torchvision
torchaudio
lxml
labelImg (optional if using source code)


file structure for YOLO:

~/path/papi_dataset/images/train/
~/path/papi_dataset/imagesval/

~/path/papi_dataset/label/train/
~/path/papi_dataset/label/val/

Yaml file for YOLO:

path: path/papi_dataset
train: images/train
val: images/val
names:
0: papi

