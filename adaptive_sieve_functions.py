from osgeo import gdal, osr
from pyproj.crs import CRS
import subprocess
import os
from time import time
import sys
from utils import time_dif 
gdal.UseExceptions()

# def pixel_area(coordinate_params, projection_name, verbose):
def pixel_area(gdal_dataset, verbose):
    
    gt = gdal_dataset.GetGeoTransform()
    #print("qqq1")
    #print(gt)
    origin_lat, pixel_size_x, origin_long, pixel_size_y = [gt[index] for index in [0,1,3,-1]]

    projection = gdal_dataset.GetProjection()
    #print("qqq projection")
    #print(projection)
    srs=osr.SpatialReference(wkt=projection)
    #print(srs)
    print("qqq srs.GetAttrValue('unit')")
    print(srs.GetAttrValue('unit'))
    
    if srs.IsProjected:
        print("Projected")
        print("qqq srs.GetAttrValue('projcs')")
        print(srs.GetAttrValue('projcs'))
        print("qqq srs.GetAttrValue('geogcs')")
        print(srs.GetAttrValue('geogcs'))
    else:
        print("qqq Not projected")
    proj_crs = CRS.from_wkt(srs.ExportToWkt())
    geod = proj_crs.get_geod() 

    lats = [origin_lat,
            origin_lat + pixel_size_x,
            origin_lat + pixel_size_x,
            origin_lat]

    longs = [origin_long,
             origin_long,
             origin_long + pixel_size_y,
             origin_long + pixel_size_y]

    #qqq start
    
    if "degree" in str(srs.GetAttrValue('unit')).lower():
        pixel_area, _ = geod.polygon_area_perimeter(longs, lats)
        print("It is in degrees")
    else:
        pixel_area = abs(pixel_size_x*pixel_size_y)
        print("It is not in degrees")
    #qqq end

    if verbose > 1:
        print("The area of one pixel is {}m{}".format(pixel_area, '\u00B2'))

    return pixel_area

def sieve_pixel_calc(sieve_area, pixel_area, verbose):
    sieve_pixel_count = round(sieve_area/pixel_area)

    if verbose > 0:
        print("The number of pixels needed to cover an area of {}m{} is {}".format(sieve_area, '\u00B2', sieve_pixel_count))

    return sieve_pixel_count

def adaptive_sieve(gdal_sieve_path,
                   input_Gtiff_path,
                   output_Gtiff_path,
                   show_intermediate,
                   sieve_size_m2 = 9,
                   connectedness = 4,
                   verbose = 0):

    start = time()

    temp_sieve_input = os.path.join(os.path.dirname(output_Gtiff_path), "tmp_in.tif")
    temp_sieve_output = os.path.join(os.path.dirname(output_Gtiff_path), "tmp_out.tif")

    source_dataset = gdal.Open(input_Gtiff_path)
    projection = source_dataset.GetProjection()
    source_srs=osr.SpatialReference(wkt=projection)
    source_GEOGCS = source_srs.GetAttrValue("GEOGCS")
    source_dataset = None

    if "wgs" in source_GEOGCS.lower() and "84" in source_GEOGCS.lower():
        print("Dataset in WGS84 crs")
        try:
            calc_dataset = gdal.Open(input_Gtiff_path)

        except RuntimeError:
            print('Unable to open tif at {}'.format(input_Gtiff_path))
            sys.exit(1)

        sieve_pixel_count = sieve_pixel_calc(sieve_area = sieve_size_m2,
                                            pixel_area = pixel_area(gdal_dataset=calc_dataset, 
                                                                    verbose = verbose),
                                            verbose = verbose)
        
        calc_dataset = None

        call_string = ["python", gdal_sieve_path, "-st", str(sieve_pixel_count),
                    "-{}".format(str(connectedness)), "-of", "GTiff",
                    input_Gtiff_path, output_Gtiff_path]

        if verbose > 2:    
            print(call_string)

        subprocess.call(call_string)

    else:
        print("Dataset in some other crs")
        gdal.Warp(temp_sieve_input, input_Gtiff_path, dstSRS = 'WGS84')

        try:
            calc_dataset = gdal.Open(temp_sieve_input)

        except RuntimeError:
            print('Unable to open tif at {}'.format(input_Gtiff_path))
            sys.exit(1)

        sieve_pixel_count = sieve_pixel_calc(sieve_area = sieve_size_m2,
                                            pixel_area = pixel_area(gdal_dataset=calc_dataset, 
                                                                    verbose = verbose),
                                            verbose = verbose)
        
        calc_dataset = None

        call_string = ["python", gdal_sieve_path, "-st", str(sieve_pixel_count),
                    "-{}".format(str(connectedness)), "-of", "GTiff",
                    temp_sieve_input, temp_sieve_output]

        if verbose > 2:    
               print(call_string)
        
        subprocess.call(call_string)
        gdal.Warp(output_Gtiff_path, temp_sieve_output, dstSRS = source_srs)
        
        try:
            os.remove(temp_sieve_output)
            os.remove(temp_sieve_input)
    
        except OSError:
            print("Cannot remove temporary files at {} and {}".format(temp_sieve_input, temp_sieve_output))

    if not show_intermediate:
        os.remove(input_Gtiff_path)

    if verbose > 0: 
        print("Sieved tiff has been saved at:\n{}".format(output_Gtiff_path))
        print("Sieving finished in {}".format(time_dif(start)))
    