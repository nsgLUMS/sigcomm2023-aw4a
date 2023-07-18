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

```-o```: Find the optimal QSS instead of running the RBR algorithm (may take a long time to run)

Examples of command line arguments include ``` -w https://www.daraz.pk -r 0.80 -p``` or ```-w https://www.dawn.com -r 0.90 -p -o```

Note: Follow this exact format: ```https://www.<URL>``` to ensure the correct reduced html is generated.

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
