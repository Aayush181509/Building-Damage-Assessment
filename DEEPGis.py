import os
from re import sub
import shutil
import subprocess
import platform
import adaptive_sieve_functions as asf
import os
from time import time
import shutil
from utils import segment_print, time_dif

def DEEPOutput(DATA_DIR, OUTPUT, cfg, new_polygonize_path):
    if platform.system() == "Linux":
        QGIS_DIR = os.path.abspath("/bin")
    if platform.system() == "Windows":
        QGIS_DIR = os.path.abspath("C:\\Program Files\\QGIS 3.16\\bin")
        
    GDAL_SIEVE = "..\..\gdal_sieve.py"
    GDAL_POLYGONIZE = new_polygonize_path

    for file_vrt in os.listdir(DATA_DIR):
        if file_vrt.endswith(".vrt"):
            VRT_FILE = os.path.join(DATA_DIR, file_vrt)
            VRT_FILE_UNET_OUTPUT_RAW = os.path.join(os.getcwd(),'submission_format', file_vrt) 
            VRT_UNET_OUTPUT_RAW = os.path.join(os.getcwd(), OUTPUT, "SEGMENTATION.tif")
            VRT_SIEVE_OUTPUT = os.path.join(os.getcwd(), OUTPUT, "SEGMENTATION_SIEVED.tif")
            shutil.copy(VRT_FILE,'submission_format')

            segment_print("START COMBINE TILE_SEG")
            devnull = open(os.devnull, 'w')
            start = time()
            try:
                call_string = ["gdal_translate", "-b", "1" , "-q", VRT_FILE_UNET_OUTPUT_RAW, VRT_UNET_OUTPUT_RAW] 
                subprocess.call(call_string, stdout = devnull , stderr = devnull)
            except:
                GDAL_TRANSLATE = os.path.join(QGIS_DIR,'gdal_translate')
                call_string = [GDAL_TRANSLATE, "-b", "1" , "-q", VRT_FILE_UNET_OUTPUT_RAW, VRT_UNET_OUTPUT_RAW]
                subprocess.call(call_string, stdout = devnull , stderr = devnull)

            print("Combined tif image file saved at:\n{}".format(VRT_FILE_UNET_OUTPUT_RAW))
            print("Combining segmented tiles finished in {}".format(time_dif(start)))
            
            segment_print("START SIEVE")
            asf.adaptive_sieve(gdal_sieve_path = GDAL_SIEVE,
                               input_Gtiff_path = VRT_UNET_OUTPUT_RAW,
                               output_Gtiff_path = VRT_SIEVE_OUTPUT,
                               sieve_size_m2 = cfg.sieve_size_m2,
                               connectedness = cfg.sieve_connectedness,
                               show_intermediate = cfg.show_intermediate,
                               verbose = 2)
            
            segment_print("START POLYGONIZE")
            start = time()
            subprocess.call(["python", GDAL_POLYGONIZE, VRT_SIEVE_OUTPUT])
            print("Building segmentation polygons saved at: \n{}".format(os.path.join(os.getcwd(),"segmentation", "buildings.geojson")))
            print("Polygonizing building segmentation masks finished in {}".format(time_dif(start)))

    if not cfg.show_intermediate:
        shutil.rmtree(os.path.join(os.getcwd(), "submission_format"))