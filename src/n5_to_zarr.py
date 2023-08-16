import zarr 
import numpy as np
import os
import shutil
import json
import dask.array as da
from pathlib import Path
import click

__version__ = "0.1.0"

def populate_zattrs(n5_path, n5_root):
    
    f_zattrs_template = open('src/zarr_attrs_template.json')
    z_attrs = json.load(f_zattrs_template)
    f_zattrs_template.close()

    #populate .zattrs
    z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
                                          "type": "space",
                                           "unit": unit} for (axis, unit) in zip(n5_root.attrs['axes'], 
                                                                                 n5_root.attrs['units'])]
    z_attrs['multiscales'][0]['version'] = '1.0'
    z_attrs['multiscales'][0]['name'] = str(n5_path).split('/')[-1].split('.')[0]
    z_attrs['multiscales'][0]['coordinateTransformations'] = [{"type": "scale",
                    "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}]
    
    return z_attrs

def ome_dataset_metadata(n5_src, n5arr):
   
    json_file_path = os.path.join(n5_src, n5arr.path, 'attributes.json' )

    with open(json_file_path, 'r') as j:
        arr_attrs_n5_init = json.loads(j.read())
    arr_attrs_n5 =  arr_attrs_n5_init['transform']
 
    dataset_meta =  {
                    "path": n5arr.path,
                    "coordinateTransformations": [{
                        'type': 'scale',
                        'scale': arr_attrs_n5['scale'],
                        'translation' : arr_attrs_n5['translate']
                    }]}
    
    return dataset_meta

@click.command()
@click.option('--n5src', help='Source path to the .n5 file and its name. Example: "../folder1/folder2/file.n5"')
@click.option('--zarrdest', help='Destination path to the .zarr file and its name. Example: "../folder1/folder2/file.zarr"')
def import_datasets(n5src, zarrdest):
    store_n5 = zarr.N5Store(n5src)
    n5_root = zarr.open_group(store_n5, mode = 'r')
    zarr_arrays = sorted(n5_root.arrays(recurse=True))

    z_store = zarr.NestedDirectoryStore(zarrdest)
    zg = zarr.open_group(z_store, mode='a')

    #provide n5 metadata according to the ome-ngff multiscale specifications
    z_attrs = populate_zattrs(n5src, n5_root)
    

    for item in zarr_arrays:
        n5arr = item[1]
        darray = da.from_array(n5arr, chunks = n5arr.chunks)

        if not (zarr.storage.contains_array(z_store, n5arr.path)):
            dataset = zg.create_dataset(n5arr.path, 
                                #data=n5arr,
                                shape=n5arr.shape,
                                chunks=n5arr.chunks,
                                dtype=n5arr.dtype
                                )
        else: 
            dataset = zarr.open_array(os.path.join(zarrdest, n5arr.path), mode='a')
        
        da.store(darray, dataset, lock = False)
        #add dataset metadata to zarr attributes
        z_attrs['multiscales'][0]['datasets'].append(ome_dataset_metadata(n5src, n5arr))

    
    # add metadata to .zattrs 
    zg.attrs['multiscales'] = z_attrs['multiscales']

if __name__ == '__main__':
    import_datasets()