import pytest
import os
import dask.array as da
import numpy as np
import zarr
from numcodecs import GZip, Blosc

import n5_to_zarr as n5toz 

def test_version():
    assert n5toz.__version__ == "0.1.0"

@pytest.fixture(scope='session')
def filepaths(tmp_path_factory):
    path = tmp_path_factory.mktemp('test_data', numbered=False)
    input = path / 'input/test_file.n5'
    output = path / 'output/test_file_new.zarr'

    populate_n5file(input)
    return (input, output)

#test file
def populate_n5file(input):
    store = zarr.N5Store(input)
    root = zarr.group(store = store, overwrite = True) 
    paths = ['render/branch_0/data', 'render/branch_0/data1/data1_lvl1/data1_lvl2',
             'render/branch_1/data2', 'render/branch_2/data3/data3_lvl1/data3_lvl2']
    datasets = []
    for path in paths:
        n5_data = zarr.create(store=store, 
                                path=path, 
                                shape = (100,100, 100),
                                chunks=10,
                                dtype='float32', compressor=GZip(level=4))#, compressor = Blosc(cname="zstd", clevel=9, shuffle=0))
        n5_data[:] = 42 * np.random.rand(100,100, 100)
        datasets.append(n5_data)

    test_metadata_n5 = {"pixelResolution":{"dimensions":[4.0,4.0,4.0],
                        "unit":"nm"},
                        "ordering":"C",
                        "scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],
                                  [32,32,32],[64,64,64],[128,128,128],[256,256,256],
                                  [512,512,512],[1024,1024,1024]],
                        "axes":["z","y","x"],
                        "units":["nm","nm","nm"],
                        "translate":[-2519,-2510,1]}
    for i in range(3):
        root[f'render/branch_{i}'].attrs.update(test_metadata_n5)
        
    res_params = [(4.0, 2.0), (8.0, 0.0), (16.0, 4.0), (32.0, 8.0)]
     
    for (data, res_param) in zip(datasets, res_params):
            transform = {
            "axes": [
                "z",
                "y",
                "x"
            ],
            "ordering": "C",
            "scale": [
                res_param[0],
                res_param[0],
                res_param[0]
            ],
            "translate": [
                res_param[1],
                res_param[1],
                res_param[1]
            ],
            "units": [
                "nm",
                "nm",
                "nm"
            ]}
            data.attrs['transform'] = transform
    
@pytest.fixture
def n5_data(filepaths):
    populate_n5file(filepaths[0])
    store_n5 = zarr.N5Store(filepaths[0])
    n5_root = zarr.open_group(store_n5, mode = 'r')
    zarr_arrays = sorted(n5_root.arrays(recurse=True))
    return (filepaths[0], n5_root, zarr_arrays)
    
def test_apply_ome_template(n5_data):
    n5_group = n5_data[1]
    if 'scales' in n5_group.attrs.asdict():
        zattrs = n5toz.apply_ome_template(n5_group)
        
        z_axes = [sub['name'] for sub in zattrs['multiscales'][0]['axes']]
        z_units = [sub['unit'] for sub in zattrs['multiscales'][0]['axes']]

        assert z_axes == n5_group.attrs['axes'] and z_units == n5_group.attrs['units']

def test_ome_dataset_metadata(n5_data):
    z_group = zarr.group()

    for item in n5_data[2]:
        n5arr = item[1]
        zarr_meta = n5toz.ome_dataset_metadata(n5arr, z_group)
        arr_attrs_n5 = n5arr.attrs['transform']

        assert (n5arr.path == zarr_meta['path'] 
                and zarr_meta['coordinateTransformations'][0]['scale'] ==  arr_attrs_n5['scale']
                and zarr_meta['coordinateTransformations'][1]['translation'] ==  arr_attrs_n5['translate'])

def test_import_datasets(n5_data, filepaths):    
    n5_src = filepaths[0]
    zarr_dest = filepaths[1]
    n5_arrays = n5_data[2]
    n5toz.import_datasets(n5_src, zarr_dest, Blosc(cname="zstd", clevel=9, shuffle=0))

    for item in n5_arrays:
        n5arr = item[1]
        z_arr = zarr.open_array(os.path.join(zarr_dest, n5arr.path), mode='r')
        assert z_arr.shape == n5arr.shape and z_arr.dtype == n5arr.dtype

def test_reconstruct_json(n5_data):

    root_dir = n5_data[0]
    n5_arrays = get_arrays(root_dir)

    #break n5 file - remove attributes.json files from subdirectories of a root directory
    dir_list = os.listdir(root_dir)
    for item in dir_list:
        dir = os.path.join(root_dir, item)
        if os.path.isdir(dir):
            os.remove(os.path.join(dir, "attributes.json"))
    
    n5_arrs_broken = get_arrays(root_dir)
    
    assert n5_arrays != n5_arrs_broken

    # add attributes.json file, if missing
    n5toz.reconstruct_json(root_dir)

    # get all arrays that are contained within / group.
    n5_arrays_fixed = get_arrays(root_dir)
    
    assert n5_arrays == n5_arrays_fixed

#retrieve zarr arrays from zarr file
def get_arrays(src_dir):
    store = zarr.N5Store(src_dir)
    group = zarr.open_group(store, mode= 'r')

    #acquire unzipped list of tupples [(names), (arrays)]
    arrays_info = list(zip(*group.arrays(recurse=True)))

    if arrays_info:
        arrays = arrays_info[1]
    else:
        arrays = []
    return arrays
    










