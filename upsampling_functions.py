from pyexpat import model
import rasterio
import cv2
import itertools
import glob
import numpy as np
import os
from os.path import join, dirname, basename
from time import time
from utils import time_dif 
from tqdm import tqdm
import pprint as pp

path = join(os.getcwd(), "input", "part1.tif")

def get_upscale_seq(method, combination_len, 
                    upsampling_factor, verbose, **kwargs):

    if "resize" not in method.lower():
        cv2_dss_factors = {"edsr":   [2, 3, 4],
                           "espcn":  [2, 3, 4],
                           "fsrcnn": [2, 3, 4],
                           "lapsrn": [2, 4, 8]}

        combinations = [p for p in itertools.product(cv2_dss_factors.get(method.lower()),
                                                     repeat=combination_len)]
        
        correct = []
        for combination in combinations:
            temp = []
            for i, factor in enumerate(combination):
                temp.append(factor)
                if upsampling_factor%np.prod(temp) == 0:
                    correct.append(temp)
                    break                

        try:
            minimal_sequence = min(correct, key=len)

        except Exception as err:
            print("Upsampling_factor not possible with given method of upsampling, defaulting to cv2.INTER_CUBIC")
            override = True
            minimal_sequence = [upsampling_factor]
            
        else:
            override=False
        
    else:
        override = False
        minimal_sequence = [upsampling_factor]

    if verbose > 0:
       print("Method:{} Upsampling_factor:{} Minimal_sequence: {}".format(method, np.prod(minimal_sequence), minimal_sequence))
    
    return minimal_sequence, override

def upscale_patches(in_image, patch_ratio, model, factor, verbose, **kwargs):
    from skimage.util.shape import view_as_windows
    
    #Cut image to patchable size
    image = in_image[0:in_image.shape[0] - in_image.shape[0]%patch_ratio, 0:in_image.shape[1] - in_image.shape[1]%patch_ratio]

    #Defining patch size
    window_size = (image.shape[0]//patch_ratio, image.shape[1]//patch_ratio, image.shape[2])

    '''Converts image into patches and predicts on the patches then reconstructes the image'''
    patch_image = view_as_windows(image, window_size, step = window_size)
    result = np.zeros((image.shape[0] * factor, image.shape[1] * factor, image.shape[2]), dtype = np.uint8)

    if verbose > 2:
        print(window_size)
        print(patch_image.shape)
        print(result.shape)

    for x in tqdm(range(patch_image.shape[0]), position=0):
        for y in tqdm(range(patch_image.shape[1]), position=1, leave=False):
            x_pos, y_pos = x * window_size[0] * factor, y * window_size[1] * factor
            result[x_pos:x_pos + (window_size[0] * factor), y_pos:y_pos + (window_size[1] * factor)] = model.upsample(patch_image[x,y,0])
                
    return result

def upsample_image(path, cv2_super_path, method,
                   minimal_sequence, override = False, **kwargs):

    upsample_start = time()
    dnn_model_path = glob.glob(cv2_super_path, recursive=True)[0]

    model_dict = {}    

    for (root, dirnames, filenames) in os.walk(os.path.join(dnn_model_path, method.upper())):
        for file in filenames:
            upscale = [c for c in file if c.isdigit()][0]
            model_dict.update({f"{upscale}":os.path.join(root, file)})
    
    if kwargs.get("verbose", None) > 1:
        print("OpenCV supersampling path:", cv2_super_path)
        print("Minimal_sequence:", minimal_sequence)
        print(model_dict)

    in_image = cv2.imread(path)

    if "resize" in method or override:
        if len(method.split("_")) < 1:
            result = cv2.resize(in_image, dsize = (in_image.shape[1] * np.prod(minimal_sequence), in_image.shape[0] * np.prod(minimal_sequence)))

        else:
            cv2_interpolation = {"nearest":  cv2.INTER_NEAREST,
                                 "linear":   cv2.INTER_LINEAR,
                                 "area":     cv2.INTER_AREA,
                                 "cubic":    cv2.INTER_CUBIC,
                                 "lanczos4": cv2.INTER_LANCZOS4}
            
            result = cv2.resize(in_image,
                                dsize = (in_image.shape[1] * np.prod(minimal_sequence), in_image.shape[0] * np.prod(minimal_sequence)),
                                interpolation = cv2_interpolation.get(method.split("_")[-1]))
        return result                            
    else:
        for i, factor in enumerate(minimal_sequence):
            upscaling_pass_start = time()
            
            if kwargs.get("verbose", None) > 1:
                print("STEP", i, "Upscale_factor", factor)
                print("DSS")
                
            if i == 0:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_dict.get(str(factor)))
                sr.setModel(method.lower(), factor)

                #CUDA implementation test, not straightforward. Probably requires separate compilation of OpenCV
                # try:
                #     sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                #     sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                # except Exception:
                #     continue

                if kwargs.get("patch_ratio", None):
                    result = upscale_patches(in_image, model = sr, factor=factor, **kwargs)
                else:
                    result = sr.upsample(in_image)

            else:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_dict.get(str(factor)))
                sr.setModel(method.lower(), factor)
                
                #CUDA implementation test, not straightforward. Probably requires separate compilation of OpenCV
                # try:
                #     sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                #     sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                # except Exception:
                #     continue

                if kwargs.get("patch_ratio", None):
                    result = upscale_patches(in_image, model = sr, factor = factor **kwargs)
                else:
                    result = sr.upsample(result)
            
            print("Single upsampling pass, upscaling by {} done in {}".format(factor, time_dif(upscaling_pass_start)))        
        print("Complete upsampling done in {}".format(time_dif(upsample_start)))

        return result

def upsample_geoimage(path, method = 'LAPSRN', cv2_super_path = "F:\Code\CV2_Supersample", 
                      upsampling_factor = 2, combination_len = 4, patch_ratio = 4, verbose = 0):
    """
    Methods: 
    CV2_supersample - edsr, espcn, fsrcnn, lapsrn  
    CV2_resize - nearest, linear, area, cubic, lanczos4
    """
    
    call_params = locals()
    minimal_sequence, override = get_upscale_seq(**call_params)

    with rasterio.Env():
        with rasterio.open(path) as dataset:
            profile = dataset.profile
            upsample_data = upsample_image(minimal_sequence = minimal_sequence,
                                           override=override, **call_params) 
            
            upscaled_transform = dataset.transform * dataset.transform.scale(
                                 (dataset.height / upsample_data.shape[0]),
                                 (dataset.width / upsample_data.shape[1]))

            profile.update(height = upsample_data.shape[0],
                           width = upsample_data.shape[1],
                           transform = upscaled_transform)
        
        file_name = join(dirname(path), "{}_{}_{}".format(basename(path),
                                                          method.upper() if override else "CUBIC",
                                                          upsampling_factor))

        with rasterio.open(file_name, 'w', **profile) as upscaled_tif:
            upscaled_tif.write(np.rollaxis(cv2.cvtColor(upsample_data, cv2.COLOR_BGR2RGB), axis = 2))
    
    return file_name