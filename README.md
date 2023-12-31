# AW4A
Systematic design to reduce webpages using Image Reduction.

## Installation guide
Install imagemagick, webp, sql client with the following commands:
```
sudo apt install imagemagick
sudo apt install webp
sudo apt install libmysqlclient-dev
```
Make sure chromedriver is installed. If it is not, use the following tutorial: https://skolo.online/documents/webscrapping/#step-2-install-chromedriver

Then install and run RBR with the following commands
```
git clone https://github.com/nsgLUMS/sigcomm2023-aw4a
cd sigcomm2023-aw4a
pip3 install -r requirements.txt
python3 server.py
```
It will prompt you for a password, which is ```boom```.

On another terminal (in the same directory):
```
python main.py -w <URL> -r <NEW PAGE RATIO> -p
```
By default, the code is set to run RBR unless the ```-o``` flag is used

```-p```: Enable PREPROCESSING (include this in the command the first time running RBR to download the image data).

```-o```: Find the optimal QSS (Grid Search) instead of running the RBR algorithm (may take a long time to run)

```-j```: To enable JS reduction using Muzeel

```-t```: SSIM Threshold for images

```-m```: To open mobile version of webpage

```-c```: Set False to disable headless chrome

```-g```: Resolution Granularity

```-a```: Weight of Area Heuristic

```-b```: Weight of Bytes Effeciency (Bytes SSIM) Heuristic

Examples of command line arguments include ``` -w https://www.daraz.pk -r 0.80 -p``` or ```-w https://www.dawn.com -r 0.90 -po```

Note: Follow this exact format: ```https://www.<URL>``` to ensure the correct reduced html is generated.

To easily run some websites use:
1. For RBR:
```
bash rbr_test.sh urls.txt
```
2. For Grid Search
WARNING: Grid Search has a large space and time complexity.
```
bash gridsearch_test.sh urls.txt
```

To view new webpage, make sure server is on using command (in the same directory):
```
python3 server.py

PEM: boom
```

Steps to run with -j:
1. Set up Muzeel using this link: https://github.com/comnetsAD/Muzeel
2. Change the ```config.json``` file to suit your muzeel MySQL setup. In particular, you should change it to reflect your database name and password.
```
"user": "root",
"host": "localhost",
"password": <YOUR PASSWORD>,
"database":  <YOUR DATABASE NAME>
```
3. Run Muzeel for the website you want to reduce
4. Put the resultant .m files (generated by Muzeel) in a folder called ```muzeel/{host}``` in the main directory (where <host> is the name of the site without ```www.``` e.g ```netflix.com```

## How RBR works
### Step 1 (Preprocessing)
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

## How Grid Search works
### Step 1 (Preprocessing)
Find all possible combinations of images according to SSIM and the resulting QSS.
All possible combinations are sorted by QSS

### Step 2 (Grid Search)
Search the list of combinations to find the first one that meets the size target


## Generate figures in paper
The file ```generate_figs.zip``` contains the data and code required to replicate the figures (9, 10 & 11) in [A Framework for Improving Web Affordability and Inclusiveness](https://dl.acm.org/doi/10.1145/3603269.3604872). 

To generate the figures, extract ```generate_figs.zip```. Open the jupyter notebook ```generate_graphs.ipynb``` using whichever environment you prefer. Run all cells. The resultant graphs should appear in the folder ```graphs```.
