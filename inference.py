from time import time
import hydra
import os
from os.path import join
import warnings
warnings.filterwarnings("ignore")
import json
import sys
import subprocess
import upsampling_functions as ups
import tiling_functions as tf
from DEEPGis import DEEPOutput
from utils import segment_print, time_dif, clear_folder, save_geojson, Logger
import building_segmentation as bs

ROOT_DIR = os.getcwd()

@hydra.main(config_path='config/config_eval.yaml')
def main(cfg):
    sys.stdout = Logger(join(os.getcwd(), "inference.log"))
    print("qqq clear folder start")
    if cfg.clear_temp:
        clear_folder(join(ROOT_DIR, cfg.temp_dir))
        print("qqq folder clear")

    if cfg.upsample:
        segment_print("START IMAGE UPSAMPLING")
        image_path = ups.upsample_geoimage(path = join(ROOT_DIR, cfg.input_dir, cfg.input_name),
                                        cv2_super_path = join(ROOT_DIR, cfg.cv2_super_path),
                                        method = cfg.upsampling_method,
                                        upsampling_factor = cfg.upsampling_factor,
                                        combination_len = cfg.combination_len,
                                        patch_ratio = cfg.patch_ratio)

        segment_print("START RASTER TILLING")
        tf.create_vrt_tiles(tifpath = image_path,
                            outpath = join(ROOT_DIR, cfg.temp_dir),
                            tile_size_x = cfg.tile_size_x,
                            tile_size_y= cfg.tile_size_y)
    
    else:
        segment_print("START RASTER TILLING")
        tf.create_vrt_tiles(tifpath = join(ROOT_DIR, cfg.input_dir, cfg.input_name),
                            outpath = join(ROOT_DIR, cfg.temp_dir),
                            tile_size_x = cfg.tile_size_x,
                            tile_size_y= cfg.tile_size_y)
    
    segment_print("START BUILDING SEGMENTATION")
    bs.segment_buildings(ROOT_DIR, cfg)
   
    DEEPOutput(DATA_DIR = join(ROOT_DIR,cfg.data_dir),
               OUTPUT = 'segmentation',
               cfg = cfg,
               new_polygonize_path = os.path.join(ROOT_DIR, "new_polygonize.py"))

    if cfg.run_classification:
        segment_print("GENERATING BUILDING PATCHES")   
        tf.generate_building_patches(raster_path  = join(ROOT_DIR, cfg.input_dir, cfg.input_name),
                                     geojson_path = join(os.getcwd(), "segmentation", "buildings.geojson"),
                                     patch_path   = join(ROOT_DIR, cfg.temp_dir,
                                                         os.path.basename(os.path.normpath(os.getcwd()))))
        
        conf_json_path = join(ROOT_DIR,
                              cfg.temp_dir,
                              os.path.basename(os.path.normpath(os.getcwd())),
                              "config.json")
        
        class_config = {"damage_precision" :cfg.damage_precision,
                        "model_path"       :join(ROOT_DIR, cfg.model_dir, cfg.class_model_name),
                        "geojson_path"     :join(os.getcwd(), "segmentation", "buildings.geojson"),
                        "patch_path"       :join(ROOT_DIR,
                                                 cfg.temp_dir,
                                                 os.path.basename(os.path.normpath(os.getcwd())))}

        with open(conf_json_path, 'w') as config_json:
            json.dump(dict(class_config), config_json)

        segment_print("START BUILDING CLASSIFICATION")
        start = time()
        subprocess.run('conda.bat activate DEEP_class && python {} {} && conda.bat deactivate'
                       .format(join(ROOT_DIR, "damage_classification.py"),
                               str(conf_json_path)),
                       shell=False)

        save_geojson(geojson_path = class_config.get("geojson_path"),
                     pred_df_path = join(ROOT_DIR,
                                         cfg.temp_dir,
                                         os.path.basename(os.path.normpath(os.getcwd())),
                                         "pred_df.csv"),
                     show_intermediate=cfg.show_intermediate,
                     save_path    = join(os.getcwd(), "segmentation", "buildings_damage_class.geojson"))

        print("Building segmentation polygons saved at:\n{}".format(join(os.getcwd(),
                                                                     "segmentation",
                                                                     "buildings_damage_class.geojson")))
        print("Building classification done in {}".format(time_dif(start)))

    if cfg.clear_temp:
        clear_folder(join(ROOT_DIR, cfg.temp_dir))
        

if __name__ == '__main__':
    start = time()
    main()
    segment_print("PIPELINE FINISHED IN {}".format(time_dif(start)))
    