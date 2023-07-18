# RankBasedReduce
Systematic design to reduce webpages by assigning weights to images.

## Installation guide
Install imagemagick and webp with the following commands:
```
sudo apt install imagemagick
sudo apt install webp
```
Then install and run RBR with the following commands
```
git clone https://github.com/RumaisaHabib/RankBasedReduce
cd RankBasedReduce
pip3 install -r requirements.txt
python3 server.py
```
You should also change the ```config.json``` file to suit your muzeel MySQL setup.

On another terminal (in the same directory):
```
python main.py -w <URL> -r <NEW PAGE RATIO>
```
```-p```: Enable PREPROCESSING (include this in the command the first time running RBR to download the image data).

```-o```: Find the optimal QSS (Grid Search) instead of running the RBR algorithm (may take a long time to run)

```-j```: To enable JS reduction using Muzeel

```-t```: SSIM Threshold for images
```-m```: To open mobile version of webpage
```-c```: To disable headless chrome
```-g```: Resolution Granularity
```-a```: Weight of Area Heuristic
```-b```: Weight of Bytes Effeciency (Bytes SSIM) Heuristic

Examples of command line arguments include ``` -w https://www.daraz.pk -r 0.80 -p``` or ```-w https://www.dawn.com -r 0.90 -p -o```

Note: Follow this exact format: ```https://www.<URL>``` to ensure the correct reduced html is generated.

```
To view new webpage, make sure server is on using command (in the same directory):

python3 server.py

PEM: boom
```

## How RBR works
### Step 1 
Rate images based on:
- SSIM & Bytes reduction relationship (Byte efficiency)
- Area

### Step 2
Reduce images in order 0...n where 0 is the image with the most _reduction potential_ as defined by step 1.

```
for image i in set of images
  while True
    Reduce image resolution
    If target webpage size is achieved
      break
    If similarity of reduced image is < good
      continue
```
