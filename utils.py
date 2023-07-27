import warnings
warnings.filterwarnings("ignore")
import os, sys
from urllib.error import HTTPError
from skimage.metrics import structural_similarity as ssim
from selenium import webdriver
import mysql.connector
from PIL import Image
import argparse, hashlib
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import multiprocessing as mp
from itertools import product
import re
import pandas as pd
import cv2
import urllib
import json
import numpy as np
import time
import itertools
from subprocess import check_output
import more_itertools as mit

parser=argparse.ArgumentParser()
parser.add_argument('-o', default=False, help="Set true to find the optimal QSS (Grid Search)",action="store_true")
parser.add_argument('-p', default=False, help="Set true to preprocess (download webpage)",action="store_true")
parser.add_argument('-w', help="Webpage URL Using Format https://[url]")
parser.add_argument('-r', help="New page ratio (between 0 and 1)")
parser.add_argument('-j', default=False, help="Set true to do JS reduction",action="store_true")
parser.add_argument('-t', default=0.9, help="SSIM threshold")
parser.add_argument('-m', default=False, help="Set true to set mobile version reduction",action="store_true")
parser.add_argument('-c', default=True, help="Set false to disable headless chrome option (default true)",action="store_false")
parser.add_argument('-g', default=0.5, help="Resolution Granualarity: the steps between each image resolution")
parser.add_argument('-a', default=1, help="Weight of Area Heuristic")
parser.add_argument('-b', default=1, help="Weight of Bytes Effeciency/Bytes SSIM Heuristic")

args = parser.parse_args()
GET_OPTIMAL = args.o
PREPROCESSING = args.p # if true then do all preprocessing
DO_JS = args.j
SSIM_THRESH = float(args.t)
MOBILE = args.m
HEADLESS_MODE = args.c

BYTE_SIZE = 1024
IMG_SRC = "Image Source"
JS_SRC = "JS Source"
JS_FINAL_NAME = "JS NAME"
IMG_NAME = "Image Name"
FINAL_NAME = "Final Name"
OG_SIZE = "Original Size (KB)"
CURR_SIZE = "Final Size (KB)"
WORST_KB = "Worst Size (KB)"
WORST_SSIM = "Worst SSIM" # At the threshold
WORST_RES = "Worst Resolution"
LOC = "Location"
AREA = "Area"
BSSIM = "ByteSSIM"
SSIM = "SSIM"
WEIGHTED_SUM = "Weighted Sum"
PROPOSED_KB = "Proposed KBs"
PROPOSED_SSIM = "Proposed SSIM"
RES_PERCENT = "Resolution Percentage"
QSS = "QSS"
LOCALHOST = 0
PARALLEL = 1
AREA_WEIGHT = float(args.a)
BSSIM_WEIGHT = float(args.b)


# Measurements
    
def get_SSIM(imageA, imageB):
    '''
    Given two image names, returns the structural simlarity between them
    '''
    # Import images
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB, 1)

    # Convert the images to grayscale
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return ssim(gray1, gray2)
    
def get_QSS(df):
    '''
    Given a df with a column "Area" and a column "SSIM", calculates the QSS of the webpage
    '''
    return np.sum(df[AREA] * df[SSIM])/np.sum(df[AREA])

def SSIM_bytes_relationship(image, temp):
    '''
    Given an image, returns a value that represent its SSIM-bytes relationship
    byteSSIM = bytes/SSIM
    '''
    try:
        os.makedirs(temp) # Folder to store reduced images for testing purposes
    except Exception as e:
        print(e)
        pass
    byteSSIM = []
    og_size = image_size(image)
    chunk_size = 5
    current_res = 100
    prev_ssim = 1
    prev_size = og_size
    ssim = 1 
    new_size = og_size
    while (current_res > 0 ):
        resolution_reduction(image, f"{temp}/{current_res}", current_res) # Reduce resolution to given percentage
        new_size = image_size(f"{temp}/{current_res}")    # Get the size of the new image
        ssim = get_SSIM(image, f"{temp}/{current_res}")  # Get the SSIM between new image and the old
        if (ssim < SSIM_THRESH or new_size>og_size):
            ssim = prev_ssim 
            new_size = prev_size
            current_res+=chunk_size
            break
        
        prev_ssim = ssim
        prev_size = new_size
        current_res -= chunk_size
    
    current_res = max(current_res, 0) # Dont let negative resolution stay
    os.system(f"rm -r {temp}") 
    
    delta_ssim = 0.000000001 if ssim == 1 else 1-ssim   # To avoid division by 0 error
    actual_size = new_size
    new_size = 0 if new_size>og_size else new_size
    return (og_size-new_size)/delta_ssim, actual_size, ssim, current_res

def try_best_SSIM_bytes(image, temp, only_name, host):
    '''
    Given an image, if it is a PNG, finds the best image format by calculating Bytes-SSIM relationship
    for original and webp formats. Returns the Bytes-Effeciency metric for the best format
    '''
    img_name = only_name
    format = check_format(image)
    if "PNG" in format:
        png, kb_png, sm_png, res_png = SSIM_bytes_relationship(image, temp)
        dir_only_name = '.'.join(image.split('.')[:-1])
        webp_name = f"{dir_only_name}.webp"
        convert_to_webp(image, webp_name)
        webp,kb_webp,sm_webp, res_webp = SSIM_bytes_relationship(webp_name, temp)

        best = max(png, webp)
        # if the webp causes ssim to be below 0.9 do not use it
        webp_ssim = get_SSIM(webp_name, image)
        if webp_ssim < 0.9:
            best = png
            
        if best == png:
            return png, f"{only_name}.png", image_size(image), kb_png, sm_png, res_png, 1
        else:
            return webp, f"{only_name}.webp", image_size(webp_name), kb_webp, sm_webp, res_webp, webp_ssim
    ef, kb, sm, res = SSIM_bytes_relationship(image, temp)
    return ef, image.split('/')[-1], image_size(image) ,kb, sm, res, 1

def image_size(image):
    '''
    Returns size (in kilobytes) of image
    '''
    return os.path.getsize(image)/BYTE_SIZE

def save_html(url, driver):
    '''
    Save the source html of a given url in {host}/info/source.html 
    '''
    host = url_2_host(url)
    # Get the page HTML 
    html = driver.execute_script("return document.body.innerHTML")

    # Replace local references with absolute paths
    html = re.sub("src=\"//", "src=\"https://",html)
    html = re.sub("src=\"/", "src=\""+ url + "/",html)
    html = re.sub("src=\"portal/", "src=\"" + url + "/portal/",html)

    # Remove source set  (easier for debugging)
    html = re.sub("srcset=\"portal/", "srcset=\"" + url + "/portal/",html)
    html = re.sub(", portal/", ", " + url + "/portal/",html)
    f = open(f"{host}/info/source.html", "w")
    f.write(html)
    f.close()

def page_dims(driver):
    '''
    Returns the dimensions of the page (height, width)
    '''
    return driver.execute_script("return document.documentElement.scrollHeight"), driver.execute_script("return document.documentElement.scrollWidth")

def collect_page(url, options):
    '''
    Collects the page data and downloads the images and JS sources from the given url
    The data includes:
    - Page html
    - Page size
    - Number of images
    - Size of each image
    - Area of each image
    All sizes are in KBs
    This can be stored as json + csv on disk or returned
    '''
    print("Collecting page data")
    # Performance metrics setup for websize 
    logging_prefs = {'performance' : 'INFO'}    
    options.set_capability('goog:loggingPrefs', logging_prefs)

    # Overall page data
    page_data = {}
    host = url_2_host(url)

    # Define driver options 
    driver = webdriver.Chrome(service=Service(), options=options) # Start web driver
    driver.get(url)

    height, width = page_dims(driver)
    driver.set_window_size(width+100,1000)

    # Scroll to the bottom of the page to actviate lazy loading. Get new height
    height = scroll_to_bottom(driver)
    height, width = page_dims(driver)

    # Store scrollable page height
    page_data["scrollHeight"] = height
    # Set window size for full screenshot
    driver.set_window_size(width, height)
    driver.save_screenshot(f"{host}/info/original.png")

    # Save html
    save_html(url, driver)
    
    # Get total page size
    total_bytes = []
    for entry in driver.get_log('performance'):
        if "Network.dataReceived" in str(entry):
            # Get the page contents sizes if found
            r = re.search(r'encodedDataLength\":(.*?),', str(entry))
            total_bytes.append(int(r.group(1)))
    kb = round((float(sum(total_bytes) / BYTE_SIZE)), 2)
    page_data["kiloBytesIn"] = kb   # Total page size

    # Image data
    results = pd.DataFrame(columns=(IMG_SRC, IMG_NAME, FINAL_NAME, OG_SIZE, LOC, AREA, BSSIM, CURR_SIZE, WORST_KB, WORST_SSIM, WORST_RES, SSIM)).set_index(IMG_NAME)
    img_tags = driver.find_elements(By.TAG_NAME, 'img') # Collect image objects
    page_data["imageCount"] = len(img_tags)
    img_srcs = [i.get_attribute('src') for i in img_tags] # Extract the sources of image objects
    img_locs = [i.location["y"]/height for i in img_tags] # Extract the relative locations of image objects
    img_areas = [i.size['width'] * i.size['height'] for i in img_tags]  # Extract the area of image objects
    
    # Javascript data 
    scripts = driver.find_elements(By.TAG_NAME, 'script')
    js_srcs = []
    try:
        os.makedirs(f"{host}/js_dump/original")
    except Exception as e:
        print("js_dump/orginal already exists") 
    for s in scripts:
        if s.get_attribute("src"):
            src = s.get_attribute("src")
            js_srcs.append(src)
            
    js_results = pd.DataFrame({JS_SRC: js_srcs, JS_FINAL_NAME: None})
    # Outputs
    if PREPROCESSING:
        results = download_images(img_srcs, img_locs, img_areas, host, results)
    else:
        results = pd.read_csv(f"{host}/info/results.csv").set_index(FINAL_NAME)
        print(list(results.index))
        
    if PREPROCESSING:
        f = open(f"{host}/info/page_data.json", "w")
        json.dump(page_data, f)
        f.close()
    else:
        with open(f"{host}/info/page_data.json", 'r') as j:
            page_data = json.loads(j.read())

    driver.close()
    return results, js_results, page_data["kiloBytesIn"]

def reduce_to_ssim(target_ssim, image_dir, image_name, dp, og_name, image_prefix="new_", other_dir="test_ssim"):
    '''
    Given a target SSIM and an image name, finds the best SSIM within the SSIM Threshold. Returns image meta data
    '''
    new_image_name = image_prefix+image_name
    new_dir = f"{image_dir}/{other_dir}"
    curr_ssim = 1
    min_val = 0
    max_val = 100
    factor = 50
    target_reached = False
    og_size = image_size(image_dir + "/" + image_name)
    if target_ssim == 1.0 and og_name == image_name: # If image is not converted to webp and the target is 1
        return True, new_dir+new_image_name, 100, og_size, get_SSIM(f"{image_dir}/{og_name}", f"{image_dir}/{image_name}")
    
    # While loop uses binary search to find the closest possible ssim to target
    while True:
        os.system(f"convert {image_dir}/{image_name} -quality {str(factor)}% {new_dir}/{new_image_name}")
        
        curr_ssim = get_SSIM((image_dir + "/" + og_name), (new_dir + "/" + new_image_name))
        rounded_ssim = round(curr_ssim, dp)
        
        new_size = image_size(new_dir + "/" + new_image_name)

        if rounded_ssim == target_ssim:
            os.system(f"cp {new_dir }/{new_image_name} {new_dir}/final_{image_name}")
            target_reached = True
            break
        
        if(rounded_ssim > target_ssim):
            max_val = factor
        else:
            min_val = factor
        new_factor = round(((min_val+max_val)/2),4)
        if factor == new_factor:
            break
        else:
            factor = new_factor 
            og_size = new_size
    try:
        status = os.system(f"cp {new_dir}/final_{image_name} {new_dir}/{new_image_name} 2>/dev/null")
    except Exception as e:
        print(e)
        pass
    if status and target_ssim == 1.0:
        os.system(f"cp {image_dir}/{image_name} {new_dir}/final_{image_name}")
        os.system(f"cp {new_dir }/final_{image_name} {new_dir}/{new_image_name}")
        target_reached = True
    new_size = image_size(new_dir + "/" + new_image_name)
    if new_size <= 0:
        target_reached = False
    else:
        curr_ssim = get_SSIM((image_dir + "/" + og_name), (new_dir + "/" +new_image_name))
    try:
        os.system(f"rm -rf {new_dir}/{new_image_name}")
        os.system(f"rm -rf {new_dir}/final_{new_image_name}")
    except:
        pass
        
    return target_reached, new_dir+new_image_name, factor, new_size, curr_ssim

def get_qss_from_string(str, df):
    '''
    Calculates QSS from string and df
    '''
    new_str = str.replace("(", "")
    new_str = new_str.replace(")", "")
    new_str = new_str.replace(",", "")
    new_str = new_str.replace("\n", "")
    ssim_list = new_str.split(" ")
    areas = list(df[AREA])
    qss_list = [(area*float(ssim)) for area,ssim in zip(areas,ssim_list)]
    qss = sum(qss_list)/sum(areas)
    return qss, new_str

def write_to_file_task(df, chunk, pid, list_of_files, dir, old_dir, all_meta_data):
    '''
    For each chunk of valid combinations, calculates and appends the QSS of each combo at the start of each line then writes to disk
    '''
    filename = f'{dir}/combos_{pid}.txt'
    with open(filename, 'w+') as f:
        for j in chunk:
            one_qss, new_str = get_qss_from_string(str(j)+"\n", df)
            f.write(f"{str(one_qss)} {new_str}\n")
    list_of_files.append(filename)

def get_optimal_ssim_task(queue, dir, group_num, filename, df, target_img_kbs, ERROR, image_dict, dp, old_dir, filestore_dictionary):
    '''
    Given a filename, returns the first combination that meets the bytes target
    '''
    new_dir = "all_ssim"
    try:
        os.makedirs(f"{dir}/{new_dir}")
    except Exception as e:
        pass
    
    f = open(filename,"r")
    line = f.readline()
    linecount=0
    while line:
        linecount+=1
        combination = remove_endline(line.split(" ")[1:])
        df[PROPOSED_SSIM] = combination
        optimal_qss = None
        size = 0
        qss_sum = 0
        skip=False
        for img_name,str_ssim in zip((list(df.index)),combination):
            ssim = float(str_ssim)
            image_meta_data = read_from_filestore(old_dir+"/all_ssim", img_name, ssim, filestore_dictionary, dp, group_num)
            if image_meta_data == None: # Another process tried this and found it impossible
                skip = True
                break
            size+=image_meta_data[1] # Second value is the size
            df.at[img_name, CURR_SIZE] = image_meta_data[1] 
            qss_sum+=image_meta_data[2] # third value is SSIM with all dp 
        if skip:
            skip = False
            line = f.readline()
            continue        
        optimal_qss= qss_sum/len(combination)
        if size <= (target_img_kbs + ERROR):
            # First QSS to meet the target is the optimal QSS as we had previously sorted the array in descending order
            df.to_csv(f"{dir}/experiment_ssim.csv")
            queue.put((group_num, optimal_qss, combination, size))
            print("Put in queue",group_num)
            f.close()
            return group_num , optimal_qss
        else:
            # Did not meet the target bytes requirement
            # print("Invalid combination, size:", size)
            pass
        line = f.readline()
    f.close()
    # print("LINECOUNT:", linecount)
    
    queue.put((group_num , None, None))
    return None

def get_optimal_qss_ssim(results, target_img_kbs, granularity, dir):
    '''
    Given a list of image names (with their areas) and the directory they are in, returns the "optimal" qss through brute force
    '''
    df = results.copy()
    GROUPS = 100 # Number of processes you are willing to allocate
    ERROR_MARGIN = 0.05
    ERROR = target_img_kbs*ERROR_MARGIN
    optimal_qss = 0
    lower = 0.90 # Lower bound for SSIM value
    upper = 1.0
    limit = 300 # Limit the number of possibilities according to your memory constrains
    dp = str(granularity).count('0') + 1 # Number of decimal places for rounding 

    possibilities = [round(lower + x*(upper-lower)/granularity, dp) for x in range(granularity)] + [upper] # All the possible SSIMs given a granularity
    print(len(possibilities), "possibilities")
    if len(possibilities) > limit:  # Memory constraint: reduce combinations
        possibilities.reverse()
        possibilities = possibilities[:limit]
        possibilities.reverse()
    print(possibilities)
    
    if PREPROCESSING:
        # Creates folder with all possible image SSIMs along with meta data
        make_all_possible_images(dir, possibilities, results[[IMG_NAME]], dp)
        l = [possibilities] * len(results)
        combinations2 = product(*l)
        chunks = mit.chunked(combinations2, 10**6)
        print("Chunks Completed")
        # Read the meta data stored on disk
        filestore_dictionary = filestore_dict(dir, list(df.index))
        all_meta_data = get_filestore_data(dir+"/all_ssim", list(df.index), filestore_dictionary)
        list_of_files = []
        processes = []
        try: 
            os.makedirs(f"{dir}/combos")
        except Exception as e:
            print(e)
            pass
        c = 0
        # write valid image combinations to disk
        for i in (chunks):
            sys.stdout.flush()
            p = mp.Process(target=write_to_file_task, args=(df, i, c, list_of_files, f"{dir}/combos", dir, all_meta_data))
            processes.append(p)
            p.start()
            c+=1
        print("All process started")

        for process in processes:
            process.join()
        print("All processes finished")

        # concat all txts to get a complete list of combincations sorted by QSS
        os.system(f"cat ./{dir}/combos/* > ./{dir}/combos/merged_combos.txt")
        os.system(f"sort --parallel=8 -r -k 1 ./{dir}/combos/merged_combos.txt > ./{dir}/sorted_merged_combos.txt")
        os.system(f"rm -rf ./{dir}/combos")

    if not PREPROCESSING:
        filestore_dictionary = filestore_dict(dir, list(df.index))
        all_meta_data = get_filestore_data(dir+"/all_ssim", list(df.index),filestore_dictionary)

    valid_image_ssims = {}    
    for i in list(df.index):
        valid_image_ssims[i] = {}

    # finding optimal from the list of sorted combos
    if not PARALLEL:
        for i, combination in enumerate(new_combos): 
            df[PROPOSED_SSIM] = combination
            optimal_qss, exp, size, im, failed_ssim = reduce_to_proposed_ssim(df, dir, dp)   # Try to reduce the images to the proposed ssim
            # im is the image name that fails, with failed_ssim being the target it could not reach. Used for optimization purposes
            if not optimal_qss:
                # print("Did not reach SSIM goal") 
                # Remove all combinations where this image is allocated the failed SSIM 
                im_index = list(df.index).index(im)

                # Remove all unwanted combinations (where the image ssim is the failed ssim)
                new_combos = remove_unwanted_combinations(new_combos, i+1, im_index, failed_ssim)
                continue
            if size <= (target_img_kbs + ERROR):
                # First QSS to meet the target is the optimal QSS as we had previously sorted the array in descending order
                exp.to_csv(f"{dir}/experiment_ssim.csv")
                return optimal_qss
            else:
                pass
                # Did not meet the target bytes requirement
                # print("Invalid combination, size:", size)
        return None
    else:
        # Parallel task here
        if PREPROCESSING:
            # splitting merged combos into groups for parallel processing
            merged_combos = f"./{dir}/sorted_merged_combos.txt"
            command = f"wc -l {merged_combos}".split()
            lines = int(check_output(command).split()[0])
            print(lines)
            try:
                os.makedirs(f"{dir}/combos/")
            except Exception as e:
                print(e)
                pass

            os.system(f"split -d --additional-suffix=.txt -l {max(lines//GROUPS,1)} {merged_combos} ./{dir}/combos/combo_")
        
        # start checking groups in parallel
        files = [f"./{dir}/combos/{x}" for x in os.listdir(f"./{dir}/combos")]
        files.sort()
        q = mp.Queue()
        rets = []
        processes = []
        for group_num, filename in enumerate(files):
            try:
                os.makedirs(f"{dir}/{group_num}/test_ssim")
            except Exception as e:
                # print(e)
                pass
            copy_files_to_dir(dir, f"{dir}/{group_num}")
            p = mp.Process(target=get_optimal_ssim_task, args=(q,f"{dir}/{group_num}", group_num, filename, df.copy(), target_img_kbs, ERROR, valid_image_ssims, dp, dir, filestore_dictionary))
            processes.append(p)
            p.start()
        terminate = False
        global_optimal = None
        best_combo = None
        red_img_size = None
        for process in processes:
            if terminate:
                process.terminate()
                continue
            process.join()
            # print("P done", process)
            while not q.empty():
                ret = q.get()
                rets.append(ret)
            # If there is an optimal that can be found with the processes that have completed so far, they will be returned
            best_pair = find_optimal_in_queue(rets) 
            if best_pair:
                print(best_pair)
                # Should kill all the processes at this point
                terminate = True
                global_optimal = best_pair[1]
                best_combo = best_pair[2]
                red_img_size = best_pair[3]
                
        return global_optimal, best_combo, red_img_size

def find_optimal_in_queue(arr):
    '''
    Finds the optimal QSS from an array of group best QSS using the following rules:
    - If the list is empty, return None
    - Iterate through the array:
        - If the arr[i][0] != arr[i-1][0]+1, return None
        - If the QSS (arr[i][1]) is None, move on to the next value
        - If the QSS is not None, return arr[i]
    '''
    arr = sorted(arr)
    if not len(arr): return None
    if arr[0][1]: return arr[0]
    for i,pair in enumerate(arr[1:]):
        if pair[0] != arr[i][0] + 1: return None
        if pair[1]: return pair

def make_combinations_from_dict(nested_dict):
    '''
    given a nested dict with all valid ssim vals for each image, returns all possible combinations of SSIMs
    '''
    valid_combos = []
    for key, values in nested_dict.items():
        valid_combos.append(values.keys()) # get all valid SSIMs from dict of each image and store in list
    
    products = itertools.product(*valid_combos) # get all combinations from this list
    return list(products)

def filestore_dict(dir, images):
    '''
    Given a list of images, create a nested dictionary with all images and the respective dictionaries
    of their possible versions.
    '''
    filestore_dict = {}
    for one_image in images: 
        filestore_dict_one_image = {}
        list_of_files = os.listdir(f"{dir}/all_ssim/{one_image}")
        list_of_keys = [(int(i.split('.')[0]), i) for i in list_of_files]
        for i in list_of_keys:
            filestore_dict_one_image[i[0]] = i[1]
        filestore_dict[one_image] = filestore_dict_one_image
    return filestore_dict

def get_filestore_data(dir, images, filestore_dictionary):
    '''
    Read all meta data for all images from disk
    '''
    all_meta_data = {}
    for one_image in images:
        all_meta_data[one_image] = {}
        filestore = filestore_dictionary[one_image]
        for val in filestore.values():
            f = open(f"{dir}/{one_image}/{val}")
            data = json.load(f)
            all_meta_data[one_image].update(data)
            f.close()
    return all_meta_data

def check_valid_combo(combo, all_meta_data, df):
    ssim_list = combo.split(" ")
    image_list = list(df.index)
    for one_image, one_ssim in zip(image_list, ssim_list):
        if all_meta_data[one_image][one_ssim] != None:
            continue
        else:
            return False
    return True
    
def read_from_filestore(old_dir, one_image, ssim_val, filestore_dict, dp, group_num):
    '''
    Given image name and SSIM val, find the meta data from the dictionary
    '''
    filestore = filestore_dict[one_image]
    keys = list(filestore.keys())
    keys.sort()
    new_ssim_val = ssim_val
    new_ssim_val = ssim_val * 10**dp
    key_store = None
    for key in keys:
        if new_ssim_val <= key:
            key_store = key
            break
    file_to_search = filestore[key_store]
    
    f = open(f"{old_dir}/{one_image}/{file_to_search}")
  
    # returns JSON object as a dictionary
    data = json.load(f)
    f.close()
    image_meta_data = None
    try:
        image_meta_data = data[str(ssim_val)]
    except KeyError:
        print(f"Could not find {one_image}, val {ssim_val} {new_ssim_val} in file {f}")
        print(data)
    return image_meta_data

def all_possible_versions_of_image_task(q, dir, possibilities, one_image, dp, new_dir,og_name):
    '''
    Takes one image as input and tries to reach all possible ssim values in a subset. writes a dictionary of all valid ssims along with metadata 
    for a particular image and a particular set of possible ssim values to file
    '''
    this_dir = f"{dir}/all_ssim/{one_image}"
    one_img_ssim_data = {}
    # iterate through all SSIM values and try to reduce image to that value (using only quality reduction)
    for ssim_val in possibilities:
        target_reached, _, factor, new_size, curr_ssim = reduce_to_ssim(ssim_val, dir, one_image, dp, og_name, f"{(str(ssim_val)).split('.')[-1]}_", new_dir)
        if ssim_val==1 and not target_reached:
            print("ERROR: SSIM 1 target not reached")
        if target_reached: # if valid ssim value
            one_img_ssim_data[ssim_val] = (factor, new_size, curr_ssim)
        else:
            one_img_ssim_data[ssim_val] = None
            
    if possibilities[-1] == 1.0:
        new_file = str(possibilities[-1]).split(".")[0] + "00"
    else:
        new_file = str(possibilities[-1]).split(".")[-1]
    with open(f"{this_dir}/{new_file}.json", "w") as outfile:
        json.dump(one_img_ssim_data, outfile)
    
    one_img_ssim_data.clear()
    
def all_possible_versions_of_image(q, dir, possibilities, one_image, dp, new_dir, og_name):
    '''
    Takes one image name as input and tries to reach all possible ssim values. returns dictionary of all valid ssims along with metadata 
    '''
    try:
        os.mkdir(f"{dir}/all_ssim/{one_image}")
    except Exception as e:
        print(e)
        return

    GROUPS = 5
    tasks = list(np.array_split(possibilities, GROUPS))
    groups_q = mp.Queue()
    rets = {}
    processes = []
    for task in tasks:
        if len(task) != 0:
            p = mp.Process(target=all_possible_versions_of_image_task, args=(groups_q, dir, task, one_image, dp, new_dir,og_name))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()
    while not groups_q.empty():
        ret = groups_q.get()

    q.put((rets, one_image))

def make_all_possible_images(dir, possibilities, all_images, dp):
    '''
    Given a list of all images and possible ssim combinations, returns a dictionary of dictionaries with valid ssim values along with meta data
    for all images
    '''
    new_dir = "all_ssim"
    q = mp.Queue()
    processes = []
    try:
        os.mkdir(f"{dir}/{new_dir}")
    except Exception as e:
        print(e)
    all_image_possibilities = {}
    for img,r in all_images.iterrows():
        p = mp.Process(target=all_possible_versions_of_image, args=(q, dir, possibilities, img, dp, new_dir, r[IMG_NAME]))
        processes.append((img, p))
        p.start()
    for process in processes:
        process[1].join()
    while not q.empty():
        ret = q.get()
    
def copy_files_to_dir(source, target):
    '''
    Given a source folder, copies the files to target folder
    '''
    os.system(f"rsync -z -q --exclude=sorted_merged_combos.txt {source}/* {target}/")

def remove_endline(arr):
    '''
    Given an array, removes \n from its elements and returns fresh array
    '''
    return [x.split("\n")[0] for x in arr]

#Ranking Helper Functions:
def beneficial_linear_normalisation(df, header): 
    # for area metric
    metric_list = df[header].to_list()
    max_val = max(metric_list)
    df[header] = df[header]/max_val
    return df
    
def non_beneficial_linear_normalisation(df, header):
    # for ssimbytes
    metric_list = df[header].to_list()
    min_val = min(metric_list)
    if min_val == 0:
        second = df[header].nlargest(2).iloc[-1]
        for i, r in df.iterrows():
            if r[header] == 0:
                df[header][i] = 1
            else:
                df[header][i] = second/(df[header][i])
    else:
        df[header] = min_val/df[header]
    return df

# Ranking algorithms
def individual_rank(df): 
    '''
    Ranking columns are named: Area, Location, ByteSSIM (choosing best format) in the csv 
    Given a csv file name, makes rankings based each of the columns
    '''
    # Area is beneficial since images with a smaller area should be reduced first
    df = beneficial_linear_normalisation(df, AREA)
    # BSSIM is non beneficial since images with a high BSSIM should be reduced first
    df = non_beneficial_linear_normalisation(df, BSSIM)
    df = non_beneficial_linear_normalisation(df, LOC)

    return df

def cumulative_rank(df, dir, weights): 
    '''
    Given a ranked list of image rankings, combine the rankings to get one prioritization ranking
    Using Weighted Sum Method (WSM): https://www.analyticsvidhya.com/blog/2020/09/how-to-rank-entities-with-multi-criteria-decision-making-methodsmcdm/
    '''
    factors = [AREA, BSSIM, LOC]
    df[WEIGHTED_SUM] = df[factors[0]]*weights[0] + df[factors[1]]*weights[1] + df[factors[2]]*weights[2]
    # sorting is in ascending order. So images with a lower weighted sum are reduced first 
    df.sort_values(WEIGHTED_SUM, inplace=True)
    df.to_csv(f"{dir}/info/heuristics_results.csv")
    return df

# Image reducers
def resolution_reduction(image, new_image, resolution):
    '''
    Given an image, reduces the resolution to the given percentage and returns the bytes reduction achieved 
    '''
    if resolution >= 100: # If resolution is unchanged, return the same image
        os.system(f"rsync {image} {new_image}")#
    else:    
        os.system(f"convert {image} -quality {resolution}% {new_image}")
    return image_size(new_image), image_size(image)

def convert_to_jpg(image, new_image):
    '''
    Converts the given image to jpg and returns the bytes reduction achieved
    '''
    os.system(f"convert {image} {new_image}")
    return (image_size(image) - image_size(new_image))

def convert_to_webp(image, new_image):  # TO DO
    '''
    Converts the given image to webp and returns the bytes reduction achieved
    '''
    os.system(f"convert {image} -define webp:lossless=true {new_image}")
    return (image_size(image) - image_size(new_image))

# Misc
def image_namer(original_name):
    '''
    Hashes the image name and appends the format
    '''
    m = hashlib.md5()
    m.update(original_name.encode('UTF-8'))
    return m.hexdigest()

def url_2_host(url):
    '''
    Remove http(s):// from the url
    '''
    return url.split("/")[-1]

def download_images(srcset, locset, areaset, dir, results):
    '''
    Given a set of image sources (srcset), downloads them in dir. Returns the image data in a df
    Image data includes:
    - Original image name (from source)
    - New image name (saved in directory)
    - Original Image size in KBs
    - Location of the image 
    - Area of the image
    - New Size of the Image in KBs (Will change if format converted to Webp)
    - Bytes Effeciency (Bytes-SSIM)
    - Current SSIM of the Image (1 if original image)
    '''
    print("Downloading images")
    done = []
    for i, loc, area in zip(srcset, locset, areaset):
        if not i or i in done:
            # If the source is 'None' type or already attempted to be downloaded, do not download
            continue
        done.append(i)
        try:
            path,name=os.path.split(i)
            if len(name) > 2000:
                print("File name too big. Aborting...")
                sys.exit(1)
            name = name.split("&")[0]
            image_name = name.split("?")[0]
            if image_name[-3:] == "gif" or os.path.exists(f"{dir}/{image_name}") or image_name[-3:] == "svg" or "http" not in i:
                # Current implementation does not handle gifs or different images of the same name
                continue
            image_name = image_namer(name)

            # Getting the metadata of the image for its size
            req = urllib.request.Request(url=i, headers={'User-Agent': 'Mozilla/5.0'})
            path = urllib.request.urlopen(req)  
            meta = path.info()
            
            # Image downloading
            try:
                i = re.sub(r'^.*?http', 'http', i)
                i = i.split("?")[0]
                os.system(f"cd {dir} && wget -q --show-progress {i} -O {image_name}")
                while LOCALHOST and not os.path.exists(f"{dir}/{image_name}"):
                    pass
                while LOCALHOST and image_size(f"{dir}/{image_name}") < img_size:
                    pass
                try:
                    only_name = image_name
                    format = check_format(f"{dir}/{image_name}").lower()
                    os.system(f"mv {dir}/{image_name} {dir}/{image_name}.{format}")
                    image_name = f"{image_name}.{format}"
                except Exception as e:
                    print("IMAGE RENAME ERR:", e)
            except Exception as e:
                print("Could not download: ", i)

            # Write image meta data to df    
            img_size = image_size(f"{dir}/{image_name}")
            bssim, chosen_final, new_size, worst_kb, worst_ssim, worst_res, curr_ssim = try_best_SSIM_bytes(f"{dir}/{image_name}", f"{dir}/temp/{image_name}", only_name, dir)
            results.loc[image_name] = [i, chosen_final, img_size, loc, area, bssim, new_size, worst_kb, worst_ssim, worst_res, curr_ssim]
        except HTTPError as e:
            if e.code == 403:
                print("HTTPError",e)
        except Exception as e:
            print("ERROR:", e)
    
    # Write all meta data to file
    results = results.reset_index().set_index(FINAL_NAME)
    results.to_csv(f"{dir}/info/results.csv")
    results.to_csv(f"{dir}/info/heuristics_results.csv")
    
    return results

def scroll_to_bottom(driver):
    '''
    Given an active webdriver, scroll to the bottom of the page manually to load lazy loaded images
    Source code was acqurired from: https://stackoverflow.com/a/27760083
    '''
    SCROLL_PAUSE_TIME = 0.05

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Slow scroll through page to activate images
    curr = 0
    next = 20
    while curr + next <= last_height:
        driver.execute_script(f"window.scrollTo({curr}, {curr + next});")
        time.sleep(0.01)
        curr = curr + next
        
    driver.implicitly_wait(10)
    return last_height

def check_format(image):
    '''
    Checks given format 
    '''
    return Image.open(image).format

def get_page_info(host, total_kbs, results):
    '''
    Print the original page size, image kbs. Return total image kbs
    '''
    df = results.copy()
    img_kbs = sum(df[OG_SIZE])
    print("Page KBs:", total_kbs)
    print("Image KBs:", img_kbs)
    print(f"Image bytes make up {img_kbs/total_kbs*100}% of the total webpage bytes")
    df = individual_rank(df)
    # Weights for metrics
    weights = [AREA_WEIGHT, BSSIM_WEIGHT, 0] # index 0 = area, index 1 = Bytes Effeciency (Bytes-SSIM), index 2 = Location (not used)
    df = cumulative_rank(df, host, weights)
    df.to_csv(f"{host}/info/heuristics_results.csv")
    return df, img_kbs, list(df.index.values)

def get_current_img_kbs(results):
    '''
    Gets the current total image kbs 
    '''
    return sum(results[CURR_SIZE])

def reduce(dir, image, resolution, first_time, results):
    '''
    Reduces an image with the given granularity and updates the results df
    '''
    size, old_size = resolution_reduction(f"{dir}/{image}", f"{dir}/reduced_{image}", resolution)
    if first_time:
        results.rename(index={image: f"reduced_{image}"}, inplace=True)

    results.at[f"reduced_{image}", CURR_SIZE] = image_size(f"{dir}/reduced_{image}")

    results.to_csv(f"{dir}/info/heuristics_results.csv")
    return results, size, old_size

def reduce_javascript(host, js_results):
    '''
    Given a list of javascript sources, compare with SQL to find the corresponding .js file
    Then replace the source html with the new one
    '''
    print("Reducing javascript")
    config=json.loads(open("config.json").read())
    mydb = mysql.connector.connect(
    host=config["host"],
    user=config["user"],
    password=config["password"],
    database=config["database"]
    )

    # SQL fetching for mapping
    mycursor = mydb.cursor()
    initiatingUrl = f"https://{host}/"
    sqlString = "SELECT requestURL, updateFilePath FROM cachedPages where updateFilePath is not NULL and initiatingUrl = %s"
    mycursor.execute(sqlString, (initiatingUrl,))

    files_mapping = mycursor.fetchall()
    new_dir = f"{host}/js_dump/updated"
    try:
        os.makedirs(new_dir)
    except:
        pass
    og_dir = f"{host}/js_dump/original"
    
    for src, updated in files_mapping:
        # Only keep those in our html
        if src in list(js_results[JS_SRC]):
            updated = updated.split(".")[0] + ".js" # we need the .js extension
            print(src, updated)
            # Download the original
            os.system(f"cd {og_dir} && wget --connect-timeout=5 -q --show-progress {src}")
            os.system(f"cp muzeel/{host}/{updated} {new_dir}/")
            # Get the source index
            index = js_results.index[js_results[JS_SRC] == src].tolist()[0]
            # Put updated name at the final name position
            js_results.at[index, JS_FINAL_NAME] = updated
            
    total_og = 0
    # Get full size of original js
    for f in os.listdir(og_dir):
        total_og += image_size(f"{og_dir}/{f}")
    
    total_updated = 0
     # Get full size of updated js
    for f in os.listdir(new_dir):
        total_updated += image_size(f"{new_dir}/{f}")
        
    reduction = total_og - total_updated
    
    return js_results, total_og, total_updated

def reduce_to_proposed_ssim(df, dir, dp, make_folder=True):
    '''
    Reduce images to the proposed ssim and return qss and new img bytes
    '''
    exp = df.copy()
    total_size = 0
    if make_folder:
        if os.path.exists(f"{dir}/test_ssim"):
            os.system(f"rm -rf {dir}/test_ssim")
        os.makedirs(f"{dir}/test_ssim")
    for i, r in exp.iterrows():
        target_reached, _ , factor, new_size, curr_ssim = reduce_to_ssim(r[PROPOSED_SSIM], dir, i, dp)
        if not target_reached:
            return None, None, None, i, r[PROPOSED_SSIM]
        exp.at[i, SSIM] = curr_ssim
        total_size+=new_size
    
    return get_QSS(exp), exp, total_size, None, None

def reduced_html(results, js_results, host, options):
    '''
    After reductions, generates new HTML with old sources replaced with new sources
    '''
    with open(f"{host}/info/source.html", "r", encoding='utf-8') as f:
        html= f.read()
        
    # First images
    for i, r in results.iterrows():
        to_replace = r[IMG_SRC] 
        replace_with = f"https://localhost:4696/{host}/{i}"
        html = html.replace(to_replace, replace_with)
    # Then JS if any
    for i, r in js_results.iterrows():
        if not r[JS_FINAL_NAME]:
            continue
        to_replace = r[JS_SRC] 
        replace_with = f"https://localhost:4696/{host}/js_dump/updated/{r[JS_FINAL_NAME]}"
        html = html.replace(to_replace, replace_with)

    options.set_capability('acceptInsecureCerts', True)
    
    # Chrome web driver
    driver = webdriver.Chrome(service=Service(), options=options)
    driver.get(f"https://{host}")

    # Manually remove srcset, forcing it use the image we give
    images = driver.find_elements(By.TAG_NAME, 'img')
    for element in images:
        driver.execute_script("arguments[0].removeAttribute('srcset')", element)
    
    f = open(f"{host}/info/source.html", "w")
    f.write(driver.page_source)
    f.close()
    
    # Replace with our new html
    driver.execute_script("document.body.innerHTML = arguments[0]", html)
        
    f = open(f"{host}/info/reduced.html", "w")
    f.write(driver.page_source)
    f.close()

    height, width = page_dims(driver)
    driver.set_window_size(width+100,1000)

    # Scroll to the bottom of the page to actviate lazy loading. Get new height
    height = scroll_to_bottom(driver)
    height, width = page_dims(driver)

    # Set window size for full screenshot
    driver.set_window_size(width, height)
    time.sleep(10)
    driver.save_screenshot(f"{host}/info/reduced.png")
    
    # REDO BUT WITH JS ONLY
    with open(f"{host}/info/source.html", "r", encoding='utf-8') as f:
        html= f.read()
        
    for i, r in js_results.iterrows():
        if not r[JS_FINAL_NAME]:
            continue
        to_replace = r[JS_SRC] 
        replace_with = f"https://localhost:4696/{host}/js_dump/updated/{r[JS_FINAL_NAME]}"
        html = html.replace(to_replace, replace_with)

    options.set_capability('acceptInsecureCerts', True)
    
    # Chrome web driver
    driver = webdriver.Chrome(service=Service(), options=options)
    driver.get(f"https://{host}")

    # Manually remove srcset, forcing it use the image we give
    images = driver.find_elements(By.TAG_NAME, 'img')
    for element in images:
        driver.execute_script("arguments[0].removeAttribute('srcset')", element)
    
    # Replace with our new html
    driver.execute_script("document.body.innerHTML = arguments[0]", html)
        
    f = open(f"{host}/info/reduced_js.html", "w")
    f.write(driver.page_source)
    f.close()

    height, width = page_dims(driver)
    driver.set_window_size(width+100,1000)

    # Scroll to the bottom of the page to actviate lazy loading. Get new height
    height = scroll_to_bottom(driver)
    height, width = page_dims(driver)

    # Set window size for full screenshot
    driver.set_window_size(width, height)
    time.sleep(10)
    driver.save_screenshot(f"{host}/info/reduced_js.png")
    
def closer_to(A, B, val):
    '''
    Return the number closer to val. If equal, return A
    '''
    if (abs(A-val)) <= (abs(B-val)):
        return A, "A"
    return B, "B"

def print_reductions(results, total_kbs, og_img_kbs, goal, og_js, new_js, SSIM_THRESH, host):
    '''
    Prints the overall results of the reduction
    '''
    if og_js:
        js_string = f"""
        Reduced js bytes from {og_js}KBs to {new_js}KBs
        {(og_js-new_js)/og_js*100}% decrease in js KBs
        """
    else:
        js_string = "Did not reduce JS"
        og_js = 0
        new_js = 0
    new_img_kbs = get_current_img_kbs(results)
    new_page_size = total_kbs-og_img_kbs+new_img_kbs-og_js+new_js
    goal_page_size = goal*total_kbs
    
    msg = f"""
    Given the target: {goal} * {total_kbs}KBs = {goal_page_size}KBs
    Reduced image bytes from {og_img_kbs}KBs to {new_img_kbs}KBs
    {(og_img_kbs-new_img_kbs)/og_img_kbs*100}% decrease in images KBs
    {js_string}
    Page size reduced to {new_page_size}KBs from {total_kbs}KBs
    Goal - The difference between achieved page size and new page size is {goal_page_size - new_page_size}KBs
    SSIM Thresh is at {SSIM_THRESH}
    """
    print(msg)
    f = open(f"{host}/info/reduction_result.txt", "w")
    f.write(msg)
    f.close()

def remove_unwanted_combinations(arr, start, index, val):
    '''
    Starting from index start of the provided array, remove all elements in which the index-th column == val
    '''
    copy = arr[start:]
    copy = list(filter(lambda x: x[index] != val, copy))
    arr = np.concatenate((arr[:start],copy))
    return arr
