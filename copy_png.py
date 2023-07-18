import os
import cv2

directory = '.'
new_directory = 'imgs'

# If dir does not exist otherwise delete next line
os.mkdir(new_directory)

def copy_images():
    for file_name in os.listdir(directory):
        sub_dir_path = directory + '/' + file_name
        if (os.path.isdir(sub_dir_path)):
            for image_name in os.listdir(sub_dir_path):
                if image_name[-4:] == '.png' or image_name[-4:] == '.jpg' or image_name[-5:] == '.webp' or image_name[-5:] == '.jpeg':
                    print(image_name)
                    img = cv2.imread(sub_dir_path+"/"+image_name)
                    copied_image_path = new_directory + '/' + image_name
                    try:
                    	cv2.imwrite(copied_image_path, img)
                    except:
                    	pass

copy_images()
