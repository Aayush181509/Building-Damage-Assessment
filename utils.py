import math
from time import strftime, gmtime, time
import os
from os.path import join, isfile, isdir, islink
import shutil
import geopandas as gpd
import pandas as pd
import sys

def segment_print(text="test", width=82, symbol = "-"):
    if (width - (len(text) + 2)) % 2 == 0:
        x = int((width - (len(text) + 2)) / 2)
        print("\n\n{} {} {}\n\n".format("".join([symbol]*x), text, "".join([symbol]*x)))
        
    else:
        x = math.floor((width - (len(text) + 2)) / 2)
        y = math.ceil((width - (len(text) + 2)) / 2)
        print("\n\n{} {} {}\n\n".format("".join([symbol]*x), text, "".join([symbol]*y)))

def time_dif(start_time):
    return strftime("%H:%M:%S", gmtime(time() - start_time))

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.
    If the error is for another reason it re-raises the error.
    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = join(folder_path, filename)
        try:
            if islink(file_path):
                os.unlink(file_path)
                os.remove(file_path)
            elif isfile(file_path):
                os.remove(file_path)
            elif isdir(file_path):
                shutil.rmtree(file_path, onerror=onerror)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def save_geojson(geojson_path, pred_df_path, save_path, show_intermediate):
    geojson_dataframe = gpd.read_file(geojson_path)
    geojson_dataframe = geojson_dataframe.set_index("id_row")
    prediction_dataframe = pd.read_csv(pred_df_path)
    prediction_dataframe = prediction_dataframe.set_index("id_row")
    merged_dataframe = pd.merge(geojson_dataframe, prediction_dataframe,
                                left_index=True, right_index=True)
    newgpd = gpd.GeoDataFrame()
    newgpd.geometry = merged_dataframe['geometry']#.apply(wkt.loads)
    newgpd['CONDITION'] = merged_dataframe['CONDITION']
    newgpd['ID'] = merged_dataframe.index
    newgpd.to_file(save_path, driver='GeoJSON')
    
    if not show_intermediate:
        os.remove(geojson_path)

class Logger(object):
    def __init__(self,log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log = open(self.log_path, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    