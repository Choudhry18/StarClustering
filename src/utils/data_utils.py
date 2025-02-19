import os
import sys
import pickle
import numpy as np
from astropy.io import fits


def create_target_db(tab=None, target=None, sz=22, human_inspected_only = True):
    """
    Function for loading of CAT annotations, FITS file, and creation of numpy array
    with all star cluster candidates. Size of output 'slices' is (N x 1 x sz xsz),
    were N is the number of valid star cluster candidates of the current galaxy.

    input:
      - cat:       CAT complete filename
      - target:    target galaxy (e.g.: eso486-g021, ngc4449, etc.)
      - sz:        window size for visualization (window size: sz x sz)

    output:
      - slices:    numpy array of size (N x 1 x sz x sz) with star cluster candidates
      - coords:    numpy array of size (N,) with star cluster candidate coordinates

    [GPS - 01/22/2019]
    """

    # targets with ACS filters
    acs435814 = ['ngc628-e']
    acs555814 = ['ngc4395-s','ngc7793-w','ugc4305','ugc4459','ugc5139']
    acs606814 = ['ic4247','ngc3738','ngc5238','ngc5474','ngc5477','ugc1249','ugc685','ugc7408','ugca281']
    acs435555814 =['ngc1313-e','ngc1313-w','ngc4449','ngc5194-ngc5195-mosaic','ngc5253','ngc628-c']

    if 'ngc1313' in target:
        col_num = 35
    elif any(ext in target for ext in ('ngc3351', 'ngc4242', 'ngc45')):
        col_num = 34
    else:
        col_num = 33

    sid, x, y, classCol = np.loadtxt(f'legus/tab_files/{tab}', usecols=(0, 1, 2, col_num), unpack=True)

    if(human_inspected_only):
        print(tab)
        # Filter out rows where classCol is 0
        valid_indices = np.where(np.isin(classCol, [1, 2, 3, 4]))[0]
        sid = sid[valid_indices]
        x = x[valid_indices]
        y = y[valid_indices]
        classCol = classCol[valid_indices]
        classCol_int = classCol.astype(int)
        class_counts = np.bincount(classCol_int)
        for value, count in enumerate(class_counts):
            print(f'Class {value}: {count} instances')

    # Load FITS data
    file_names = [file for file in sorted(os.listdir('legus/frc_fits_files/')) if target in file]

    fits_image_filename1 = [file for file in file_names if '275' in file]
    fits_image_filename2 = [file for file in file_names if '336' in file]
    fits_image_filename3 = [file for file in file_names if '435' in file or '438' in file]
    fits_image_filename4 = [file for file in file_names if '555' in file or '606' in file]
    fits_image_filename5 = [file for file in file_names if '814' in file]

    hdul1 = fits.open('legus/frc_fits_files/'+fits_image_filename1[0])
    hdul2 = fits.open('legus/frc_fits_files/'+fits_image_filename2[0])
    hdul3 = fits.open('legus/frc_fits_files/'+fits_image_filename3[0])
    hdul4 = fits.open('legus/frc_fits_files/'+fits_image_filename4[0])
    hdul5 = fits.open('legus/frc_fits_files/'+fits_image_filename5[0])

    # Working with Image Data
    data1 = hdul1[0].data
    data2 = hdul2[0].data
    data3 = hdul3[0].data
    data4 = hdul4[0].data
    data5 = hdul5[0].data

    w = int(np.floor(sz/2)) # used for slicing
    ty, tx = np.shape(data1)
    good = np.where(np.logical_and(np.logical_and(x > w+1 , (tx - x) > w+1) ,
                     np.logical_and(y > w+1 , (ty - y) > w+1))) \
    
    # to discard candidates close to border ([x,y] < w)
    sid, x, y, classCol = sid[good], x[good], y[good], classCol[good]
    slices = []
    if sz%2 == 1:
        ww = 1
    else:
        ww = 0
    for i in range(len(x)): # create array with all slices of current galaxy
        obj_slices = np.zeros((5,sz,sz))
        obj_slices[0,:,:] = data1[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[1,:,:] = data2[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[2,:,:] = data3[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[3,:,:] = data4[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[4,:,:] = data5[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        slices.extend([obj_slices])
    slices = np.asarray(slices) # Array of size (N x 5 x sz x sz)
    
    # save coordinates of each candidate and ids in .tab file
    coords = np.concatenate((np.asarray(x, dtype=np.int64)[:,np.newaxis], \
                        np.asarray(y, dtype=np.int64)[:,np.newaxis]),axis=1)
    ids = np.asarray(sid, dtype=np.int64)
    labels = np.asarray(classCol, dtype=np.int64)
    return slices, coords, ids, labels


def load_db(file_name):
    """
    Function for loading dataset from .dat dictionary.

    input:
      - file_name: file name and directory of current galaxy dataset file

    output:
      - dset['data']: numpy array with image data of size (N x 5 x sz x sz)
      - dset['label']: numpy array with labels of size (N,)
      - dset['coordinates']: numpy array with coordinates of each object of size (N, 2)

    where N is the number of candidates of current galaxy and sz is the slice size.

    [GPS - 01/22/2019]
    """
    with open(file_name, 'rb') as infile:
        dset = pickle.load(infile)
    return dset['data'], dset['coordinates'], dset['ids'], dset['labels']


def get_name(name):
    """
    Remove \n character at the en of line (if exists)
    """
    if name[-1] == '\n':
        return name[0:-1]
    else:
        return name


def merge_tab_files(tab_files):
    """
    Merge content of multiple tab files and delete the old files
    """
    merged_content = ""
    for tab_file in tab_files:
        tab_file_path = os.path.join('legus/tab_files', tab_file)
        with open(tab_file_path, 'r') as file:
            merged_content += file.read()
        os.remove(tab_file_path)
    return merged_content

def get_tab_filenames(targets):
    """
    Retrieve and merge tab filenames from a target list
    """
    tab_filenames = []
    tabs = os.listdir('legus/tab_files')
    for target in targets:
        print(targets)
        target_tabs = []
        for tab in tabs:
            if ('_' + get_name(target) + '_') in tab or ("_" + get_name(target) + "-") in tab or ("_" + get_name(target)) in tab:
                target_tabs.append(tab)
        if target_tabs:
            merged_content = merge_tab_files(target_tabs)
            merged_tab_filename = f"merged_{get_name(target)}.tab"
            with open(os.path.join('legus/tab_files', merged_tab_filename), 'w') as merged_file:
                merged_file.write(merged_content)
            tab_filenames.append(merged_tab_filename)
        else:
            sys.exit(f'NOT FOUND: tab file not found for galaxy {target}')
    
    return tab_filenames