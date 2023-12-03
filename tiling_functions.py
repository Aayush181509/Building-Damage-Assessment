from genericpath import isdir
from re import S
from osgeo import gdal
import itertools
import subprocess
import os
from time import time,strftime
import multiprocessing
from utils import time_dif
import fnmatch
import cv2
import geopandas as gpd
import rasterio
from tqdm import tqdm
from pyproj import Proj, transform
import numpy as np
# from rasterio.windows import Window


def gen_params_tiling(tifpath, outpath, tile_size_x, tile_size_y):
    ds = gdal.Open(tifpath)
    band = ds.GetRasterBand(1)
    # print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    min = band.GetMinimum()
    max = band.GetMaximum()
    if not min or not max:
        (min,max) = band.ComputeRasterMinMax(True)
        # print("Min={:.3f}, Max={:.3f}".format(min,max))

    if band.GetOverviewCount() > 0:
        print("Band has {} overviews".format(band.GetOverviewCount()))

    if band.GetRasterColorTable():
        print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

    # Remove tif after load to free RAM
    del ds

    # Determine size of image and create tile_size_x * tile_size_y tile coordinate ranges
    xsize = band.XSize
    ysize = band.YSize
    x_range = range(0, xsize, tile_size_x)
    y_range = range(0, ysize, tile_size_y)

    # Create lists for itertools to create combinations. Since these values are unique we create 1 item long lists (works with itertools)
    size_x = [tile_size_x]
    size_y = [tile_size_y]
    in_path = [tifpath]
    out_path = [outpath]

    # Generate a list of tuple combinations for multiprocessing
    paramlist = list(itertools.product(x_range, y_range, size_x, size_y, in_path, out_path))

    return paramlist
    
def single_tile(params):
    start_x = str(params[0])
    start_y = str(params[1])
    size_x = str(params[2])
    size_y = str(params[3])
    inpath = str(params[4])
    outpath = str(params[5])
        
    call_string = ["gdal_translate", "-of", "GTIFF", "-srcwin", start_x, start_y,
                    size_x, size_y, inpath, "{}{}{}_{}.tif".format(outpath, "\\tile_", start_x, start_y), "-q"]
        
    subprocess.call(call_string)

# Multiprocessing parameters resulting from gen_params_tilling
def run_tiling(multiprocessing_params):
    pool = multiprocessing.Pool()
    pool.map(single_tile, multiprocessing_params)
    pool.close()
    pool.join()

def create_tif_txt(path):
    file_list = os.listdir(path)
    if file_list:
        tiff_path_list = []
        for file in file_list:
            if '.tif' in file or '.tiff' in file:
                tiff_path_list.append(os.path.join(path,file))

        with open(os.path.join(path,"tif_list.txt"), 'w+') as file:
            for item in tiff_path_list:
                file.write(item + "\n")
            file.close()

def build_vrt(path):
    call_string = ["gdalbuildvrt",
                   "-input_file_list",
                   "{}".format(os.path.join(path,"tif_list.txt")),
                   "{}".format(os.path.join(path, "TEST.vrt"))] 
                   
    subprocess.call(call_string)


def create_vrt_tiles(tifpath, outpath, tile_size_x, tile_size_y):
    start = time()
    def check_int(x, name):
        if not isinstance(x, int):
            try:
                x = int(x)
                print("'{}' was converted to int {}".format(name, x))
            except TypeError:
                print("The value given for '{}' is not an int and cannot be converted to an int.\
                       Please input an int value".format(name))

    if not os.path.isfile(tifpath):
            raise ValueError ("Please input the full file path.")

    if ".tif" not in str(tifpath):
            raise ValueError ("Please define a path to a tif file.")

    if not os.path.isdir(outpath):
        os.mkdir(outpath)
        
    check_int(tile_size_x, "tile_size_x")
    check_int(tile_size_y, "tile_size_y")
    
    tiling_params = gen_params_tiling(tifpath, outpath, tile_size_x, tile_size_y)

    if len(tiling_params) != len(fnmatch.filter(os.listdir(outpath), "*.tif")):
        run_tiling(tiling_params)
        create_tif_txt(outpath)
        build_vrt(outpath)
        print("Tiles saved at {}".format(outpath))
        print("Creating tiles finished in {}".format(time_dif(start)))
    else:
        print("The tiles for file {} already exist".format(os.path.basename(tifpath)))

def single_patch_idx_rasterio_read(patch_params):
    
    df_index        = patch_params[0]
    geojson_poly_df = patch_params[1]
    raster_path     = patch_params[2]
    save_path       = patch_params[3]
    buffer_size     = patch_params[4]

    # padding around detection to crop
    raster_dataset = rasterio.open(raster_path)
    poly = geojson_poly_df.iloc[df_index, 2].buffer(buffer_size) 
    inProj = Proj(init=raster_dataset.meta['crs']['init']) #Proj(init='epsg:32737') 
    outProj = Proj(init=raster_dataset.meta['crs']['init']) # convert to cog crs
    
    # convert from geocoords to display window
    minx, miny = transform(inProj,outProj,*poly.bounds[:2])
    maxx, maxy = transform(inProj,outProj,*poly.bounds[2:])
    ul = raster_dataset.index(minx, miny)
    lr = raster_dataset.index(maxx, maxy)
    disp_minx, disp_maxx, disp_miny, disp_maxy = lr[0], (max(ul[0],0)+1), max(ul[1],0), (lr[1]+1)

    if disp_maxx-disp_minx <= 150: disp_maxx += 25; disp_minx-=25; 
    if disp_maxy-disp_miny <= 150: disp_maxy += 25; disp_miny-=25;

    window = (max(disp_minx,0), disp_maxx), (max(disp_miny,0), disp_maxy)
    data = raster_dataset.read(window=window)
    
    tile_bgr = cv2.cvtColor(np.rollaxis(data,0,3), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path,
                            "{}.{}".format(str(df_index),
                            "jpg")),
                tile_bgr)

def generate_building_patches(raster_path: str,
                              geojson_path: str,
                              patch_path: str,
                              **kwargs):

    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    geojson_dataframe = gpd.read_file(geojson_path)
    patch_ext_params   = list(itertools.product(list(range(len(geojson_dataframe))),
                                                [geojson_dataframe[(geojson_dataframe['geometry'].type=='Polygon')]],
                                                [raster_path],
                                                [patch_path],
                                                [kwargs.get("buffer_size", 0.00001)]))

    start = time()  
    pool = multiprocessing.Pool()
    pool.map(single_patch_idx_rasterio_read, patch_ext_params)
    pool.close()
    pool.join()
    print("Building patches created in {}".format(time_dif(start)))
            
