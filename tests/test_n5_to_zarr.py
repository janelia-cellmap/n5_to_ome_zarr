import pytest
import os
import dask.array as da
import numpy as np
import zarr

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
    paths = ['data', 'data1/data1_lvl1/data1_lvl2']
    
    n5_data = zarr.create(store=store, 
                            path=paths[0], 
                            shape = (100,100, 100),
                            chunks=10,
                            dtype='float32')
    
    n5_data1 = zarr.create(store=store, 
                            path=paths[1], 
                            shape = (100,100, 100),
                            chunks=10,
                            dtype='float32')

    n5_data[:] = 42 * np.random.rand(100,100, 100)
    n5_data1[:] = 42 * np.random.rand(100,100, 100)
    datasets = [n5_data, n5_data1]

    test_metadata_n5 = {"pixelResolution":{"dimensions":[4.0,4.0,4.0],
                        "unit":"nm"},
                        "ordering":"C",
                        "scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],
                                  [32,32,32],[64,64,64],[128,128,128],[256,256,256],
                                  [512,512,512],[1024,1024,1024]],
                        "axes":["z","y","x"],
                        "units":["nm","nm","nm"],
                        "translate":[-2519,-2510,1]}
    root.attrs.update(test_metadata_n5)

    res_params = [(13.0, 0.0), (15.0, 2.0)]
    
   
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
        arr_attrs_n5 = n5arr.attrs['transform']

        assert (n5arr.path == zarr_meta['path'] 
                and zarr_meta['coordinateTransformations'][0]['scale'] ==  arr_attrs_n5['scale']
                and zarr_meta['coordinateTransformations'][0]['translation'] ==  arr_attrs_n5['translate'])

def test_import_datasets(n5_data, filepaths):
    n5_src = filepaths[0]
    zarr_dest = filepaths[1]
    n5_arrays = n5_data[2]
    n5toz.import_datasets(n5_src, zarr_dest)

    for item in n5_arrays:
        n5arr = item[1]
        z_arr = zarr.open_array(os.path.join(zarr_dest, n5arr.path), mode='r')
        assert z_arr.shape == n5arr.shape and z_arr.dtype == n5arr.dtype 





