'''
A systematic framework implementation to reduce a webpage
'''
from utils import *
GRANULARITY = float(args.g) # Granularity for resolution reduction
BYTES_ERROR_MARGIN = 0.05 # Error margin for target bytes

# configuration for Nexus 5
PHONE_WIDTH = 360
PHONE_HEIGHT = 640
PIXEL_RATIO = 3.0

if GET_OPTIMAL:
    print("WARNING: GET_OPTIMAL mode may add severe performance overheads if there are many images")

# Web driver options
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument("--test-type")
options.add_argument('--allow-insecure-localhost')
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

if MOBILE:
    # Mobile emulation, currently configured for Nexus 5
    mobile_emulation = {
        "deviceMetrics": { "width": PHONE_WIDTH, "height": PHONE_HEIGHT, "pixelRatio": PIXEL_RATIO},
        "userAgent": "Mozilla/5.0 (Linux; Android 8.1.0; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/114.0.5735.196 Mobile Safari/535.19" }
    options.add_experimental_option("mobileEmulation", mobile_emulation)
    options.add_experimental_option("excludeSwitches", ["disable-popup-blocking"])

if HEADLESS_MODE:
    options.add_argument('headless')
    options.add_argument('fullscreen')
options.set_capability("acceptInsecureCerts", True)

# Collect page data and store on disk
url = args.w
goal = float(args.r)
host = url_2_host(url)

# if preprocessing enabled, scrape URL data again, otherwise use previously downloaded data
# the data must be stored in the folder under host name
if PREPROCESSING:
    if os.path.exists(host):
        os.system(f"rm -r {host}")
    os.makedirs(f"{host}/info")
else:
    if not os.path.exists(host):
        print("ERROR: Enable preprocessing since preprocessed data for this URL does not exist.")
        quit()
try:
    results, js_results, total_kbs = collect_page(url, options)
    # Get the original image bytes and the current image names
    results, og_img_kbs, img_names = get_page_info(host, total_kbs, results)
    target_img_kbs = og_img_kbs- ((1-goal)*total_kbs)
    target_total_kbs = (goal)*total_kbs
    to_be_removed = total_kbs - target_total_kbs
    print(f"Target image KBs: {target_img_kbs}")
    print(f"Target  KBs: {(goal)*total_kbs}")
    print(f"To  remove : {to_be_removed}")
except Exception as e:
    quit(e)

if target_img_kbs <= 0:
    print("Note: can't reduce page to this size with just image reduction.")
    if not DO_JS:
        sys.exit(1)

if GET_OPTIMAL:
    print("Finding optimal qss")
    experiment_filename = "optimal_experiment.csv"
    if os.path.exists(experiment_filename):
        experiment = pd.read_csv(experiment_filename).set_index(["URL", "Reduction"])
    else:
        experiment = pd.DataFrame(columns=["QSS", "Combo","Original Bytes","Reduced Bytes","Time(s)", "URL", "Reduction"]).set_index(["URL", "Reduction"])
    start = time.time()
    optimal_qss_ssim, best_combo, red_img_bytes = get_optimal_qss_ssim(results, target_img_kbs, 10, host)
    time_taken = time.time() - start
    print(f"Time taken (Optimal): {time_taken}")
    # Update the experiment
    temp = pd.DataFrame([{"QSS":optimal_qss_ssim, "Combo":best_combo, "Original Bytes":og_img_kbs, "Reduced Bytes":red_img_bytes,"Time(s)":time_taken, "URL":url, "Reduction":(1-goal)*10}], columns=["QSS","Combo","Original Bytes","Reduced Bytes","Time(s)","URL","Reduction"]).set_index(["URL","Reduction"])
    experiment = pd.concat([experiment, temp])
    experiment.to_csv(experiment_filename)
    print(f"Optimal QSS (SSIM method): {optimal_qss_ssim}")

if not GET_OPTIMAL:
    # Start reducing images one by one
    experiment_filename = "rbr_experiment.csv"
    if os.path.exists(experiment_filename):
        experiment = pd.read_csv(experiment_filename).set_index(["URL", "Reduction"])
    else:
        experiment = pd.DataFrame(columns=["QSS","QFS","Quality","Original Bytes","Reduced Bytes","Time(s)", "URL", "Reduction"]).set_index(["URL", "Reduction"])
    print("Reducing images")
    if not PREPROCESSING:
        results = pd.read_csv(f"{host}/info/heuristics_results.csv").set_index(FINAL_NAME)
        img_names = list(results.index.values)
    
    # Try worst ones first
    total_to_remove = og_img_kbs - (target_img_kbs + (BYTES_ERROR_MARGIN*target_img_kbs))
    total_removed = og_img_kbs - get_current_img_kbs(results)
    skip_list = []
    temp_results = results.copy()
    start = time.time()
    
    og_js = None
    new_js = None
    
    if DO_JS:
    	# Try replacing Javascript
    	# NOTE: You need to run muzeel beforehand and store the resultant .m files in the muzeel/host folder.
        js_results, og_js, new_js = reduce_javascript(host, js_results)
        js_results.to_csv(f"{host}/info/js_results.csv")
        to_be_removed = to_be_removed - (og_js-new_js)
        target_img_kbs =  og_img_kbs - to_be_removed
    
    if to_be_removed > og_img_kbs:
        print("Cannot reach target")
        sys.exit(1)
        
    
    print("Threshold:", SSIM_THRESH)
    for img in img_names:
        if img in skip_list:
            continue
        counter = 0
        res = 100 # Start with 100% resolution 
        res-=GRANULARITY # Start decreasing resolution according to specified granularity
        old_size = image_size(f"{host}/{img}")
        first_time = True # This image is being reduced for the first time
        og_name = results.loc[img][IMG_NAME] # Original name of image
        curr_SSIM = get_SSIM(f"{host}/{og_name}", f"{host}/{img}") # Measure the SSIM currently (might be <1 due to format changes)
        results.at[f"{img}", SSIM] = curr_SSIM
        while (get_current_img_kbs(results) > target_img_kbs + (BYTES_ERROR_MARGIN*target_img_kbs) and curr_SSIM > SSIM_THRESH and res>=1):
            results, new_size, _ = reduce(host, img, res, first_time, results)
            first_time = False
            curr_SSIM = get_SSIM(f"{host}/{og_name}", f"{host}/reduced_{img}")
            if (curr_SSIM < SSIM_THRESH):
                results, _, _ = reduce(host, img, res+GRANULARITY, first_time, results)
                results.at[f"reduced_{img}", RES_PERCENT] = res+GRANULARITY
                break
            results.at[f"reduced_{img}", RES_PERCENT] = res
            results.at[f"reduced_{img}", SSIM] = curr_SSIM
            counter+=1
            res-=GRANULARITY
            old_size = new_size

    time_taken = time.time() - start
    # Save results
    results.to_csv(f"{host}/info/heuristics_results.csv")
    if not LOCALHOST:
        print("Generating new webpage")
        reduced_html(results, js_results, host, options)
        
    qss = get_QSS(results)
    qfs = None
    quality = None
    if not DO_JS:
        qfs = 1
        quality = qfs+qss
    temp = pd.DataFrame([{"QSS":qss, "QFS":qfs ,"Quality": quality,"Original Bytes":og_img_kbs, "Reduced Bytes":get_current_img_kbs(results), "Time(s)":time_taken, "URL":url, "Reduction":(1-goal)*100}], columns=["QSS","QFS","Quality","Original Bytes","Reduced Bytes","Time(s)","URL","Reduction"]).set_index(["URL","Reduction"])
    experiment = pd.concat([experiment, temp])
    if (get_current_img_kbs(results) <= target_img_kbs + BYTES_ERROR_MARGIN*target_img_kbs):
        experiment.to_csv(experiment_filename)
    print("QSS:",qss)
    f = open(f"qss.txt", "a")
    f.write(f"{host}, {goal}: {qss}\n")
    f.close()

    print_reductions(results, total_kbs, og_img_kbs, goal, og_js, new_js, SSIM_THRESH, host)

