import sys
import os
import subprocess
import gdal
from time import time
from multiprocessing import Pool
import json
import geopandas as gpd
import glob
from utils import time_dif, Logger
gdal.UseExceptions()


def polygonize(path):

    subprocess.call(["python", "..\..\gdal_polygonize.py", path,
                    "-q", "-f", "ESRI Shapefile", "{0}.shp".format(os.path.splitext(path)[0]), "polygons", "band_value"])

if __name__ == "__main__":
    inpath = sys.argv[1]
    assert os.path.exists(inpath)

    src = gdal.Open(inpath)
    xsize = src.RasterXSize
    ysize = src.RasterYSize

    xdif = xsize / 16
    ydif = ysize / 16

    vrt_list = []

    for x in range(16):
        xstart = xdif * x
        xwinsize = min(xdif, xsize-xstart)
        for y in range(16):
            ystart = ydif * y
            ywinsize = min(ydif, ysize-ystart)
            subprocess.call(["gdal_translate", "-of", "VRT", "-q", "-srcwin", str(xstart), str(ystart), str(xwinsize), str(ywinsize), inpath, "tmp_{0}_{1}.vrt".format(x,y)])
            vrt_list.append("tmp_{0}_{1}.vrt".format(x,y))

    pool = Pool()
    pool.map(polygonize, vrt_list)
    pool.close()
    pool.join()

    first = True
    for vrt in vrt_list:
        pth = "{0}.shp".format(os.path.splitext(vrt)[0])
        if first:
            subprocess.call(["ogr2ogr", "-f", "ESRI Shapefile", "-nlt", "MULTIPOLYGON", "tmp_merged.shp", pth])
            first = False
        else:
            subprocess.call(["ogr2ogr", "-f", "ESRI Shapefile", "-nlt", "MULTIPOLYGON", "-update", "-append", "tmp_merged.shp", pth])
            
    subprocess.call(["ogr2ogr", "tmp_output.shp", "tmp_merged.shp", "-dialect", "sqlite", "-sql", "SELECT ST_Union(geometry), band_value FROM tmp_merged where band_value = 255 GROUP BY band_value"])
    
    original_df = gpd.read_file("tmp_output.shp")
    exploded = original_df.explode()
    exploded.to_file("tmp_output2.geojson", driver="GeoJSON")

    with open("tmp_output2.geojson", 'r') as data_file:
        input_json = json.load(data_file)

    i=0
    for x in input_json["features"]:
        x["properties"]["id_row"]=i
        i=i+1

    with open(os.path.join("segmentation", "buildings.geojson"), 'w') as data_file:
        output_dict = json.dump(input_json, data_file)

    for filename in glob.glob("tmp_*"):
        os.remove(filename) 
