import pytest
from pathlib import Path
import os
import dask.array as da
from random import randint
import numpy as np
import zarr
import json

import n5_to_zarr as n5toz 

def test_version():
    assert n5toz.__version__ == "0.1.0"

@pytest.fixture
def filepaths():
    p = os.path.join('..', 'test_data')
    os.makedirs(p, exist_ok = True)
    input = os.path.join(p, 'input', 'test_file.n5')
    output = os.path.join(p, 'output', 'test_file_new.zarr')

    os.makedirs(input, exist_ok = True )
    os.makedirs(output, exist_ok = True)
    return (input, output)

@pytest.fixture
def n5_data(filepaths):
    store_n5 = zarr.N5Store(filepaths[0])
    n5_root = zarr.open_group(store_n5, mode = 'r')
    zarr_arrays = sorted(n5_root.arrays(recurse=True))
    return (filepaths[0], n5_root, zarr_arrays)

def test_populate_zattrs(n5_data):
    n5_root = n5_data[1]
    n5_path = n5_data[0]
    zattrs = n5toz.populate_zattrs(n5_path, n5_root)
    
    z_axes = [sub['name'] for sub in zattrs['multiscales'][0]['axes']]
    z_units = [sub['unit'] for sub in zattrs['multiscales'][0]['axes']]

    assert z_axes == n5_root.attrs['axes'] and z_units == n5_root.attrs['units']

def test_ome_dataset_metadata(n5_data):
    n5_src = n5_data[0]

    for item in n5_data[2]:
        n5arr = item[1]
        zarr_meta = n5toz.ome_dataset_metadata(n5_src, n5arr)
        f_arr_attrs_n5 = open(os.path.join(n5_src, n5arr.path, 'attributes.json' ))
        arr_attrs_n5 = json.load(f_arr_attrs_n5)['transform']
        f_arr_attrs_n5.close()

        assert (n5arr.path == zarr_meta['path'] 
                and zarr_meta['coordinateTransformations'][0]['scale'] ==  arr_attrs_n5['scale']
                and zarr_meta['coordinateTransformations'][0]['translation'] ==  arr_attrs_n5['translate'])

def test_import_datasets(n5_data, filepaths):
    n5_src = filepaths[0]
    zarr_dest = filepaths[1]
    n5_arrays = n5_data[2]
    n5toz.import_datasets(n5_src, zarr_dest)
    z_store = zarr.NestedDirectoryStore(zarr_dest)
    z_root = zarr.open_group(z_store, mode = 'r')

    for item in n5_arrays:
        n5arr = item[1]
        z_arr = zarr.open_array(os.path.join(zarr_dest, n5arr.path), mode='r')
        assert z_arr.shape == n5arr.shape and z_arr.dtype == n5arr.dtype 





