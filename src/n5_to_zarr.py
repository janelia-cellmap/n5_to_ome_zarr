import zarr 
import os
import json
import dask.array as da
import click
import pydantic_zarr as pz
from numcodecs import Blosc
import os
from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
import time

__version__ = "0.1.0"

def apply_ome_template(zgroup):
    
    f_zattrs_template = open('src/zarr_attrs_template.json')
    z_attrs = json.load(f_zattrs_template)
    f_zattrs_template.close()

    #populate .zattrs
    z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
                                          "type": "space",
                                           "unit": unit} for (axis, unit) in zip(zgroup.attrs['axes'], 
                                                                                 zgroup.attrs['units'])]
    z_attrs['multiscales'][0]['version'] = '1.0'
    z_attrs['multiscales'][0]['name'] = zgroup.name
    z_attrs['multiscales'][0]['coordinateTransformations'] = [{"type": "scale",
                    "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}]
    
    return z_attrs

def normalize_to_omengff(zgroup):
    group_keys = zgroup.keys()

    for key in group_keys:
        if type(zgroup[key]) == zarr.hierarchy.Group:

            normalize_to_omengff(zgroup[key])
            if 'scales' in zgroup[key].attrs.asdict():
                zattrs = apply_ome_template(zgroup[key])
                zarrays = sorted(zgroup[key].arrays(recurse=True))

                #add datasets metadata to the omengff template
                for arr in zarrays:
                    zattrs['multiscales'][0]['datasets'].append(ome_dataset_metadata(arr[1]))

                zgroup[key].attrs['multiscales'] = zattrs['multiscales']


def ome_dataset_metadata(n5arr):
   
    arr_attrs_n5 = n5arr.attrs['transform']
    dataset_meta =  {
                    "path": n5arr.path,
                    "coordinateTransformations": [{
                        'type': 'scale',
                        'scale': arr_attrs_n5['scale'],
                        'translation' : arr_attrs_n5['translate']
                    }]}
    
    return dataset_meta

# d=groupspec.to_dict(),  
def normalize_groupspec(d, comp):
    for k,v in d.items():
        if k == "compressor":
            d[k] = comp.get_config()

        elif k == 'dimension_separator':
            d[k] = '/'
        elif type(v) is dict:
            normalize_groupspec(v, comp)

def copy_n5_store(n5_root, z_store, comp):
    spec_n5 = pz.GroupSpec.from_zarr(n5_root)
    spec_n5_dict = spec_n5.dict()
    normalize_groupspec(spec_n5_dict, comp)
    spec_n5 = pz.GroupSpec(**spec_n5_dict)
    return spec_n5.to_zarr(z_store, path= '')



def import_datasets(n5src, zarrdest, comp):
    
    store_n5 = zarr.N5Store(n5src)
    n5_root = zarr.open_group(store_n5, mode = 'r')
    zarr_arrays = sorted(n5_root.arrays(recurse=True))

    z_store = zarr.NestedDirectoryStore(zarrdest)
    zg = copy_n5_store(n5_root, z_store, comp)

    normalize_to_omengff(zg)

    for item in zarr_arrays:
        n5arr = item[1]
        darray = da.from_array(n5arr, chunks = n5arr.chunks)
        dataset = zarr.open_array(os.path.join(zarrdest, n5arr.path), mode='a')
        
        da.store(darray, dataset, lock = False)


@click.command()
@click.argument('n5src', type=click.STRING)
@click.argument('zarrdest', type=click.STRING)
@click.option('--cname', default = "zstd", type=click.STRING)
@click.option('--clevel', default = 9, type=click.INT)
@click.option('--shuffle', default = 0, type=click.INT)
def cli(n5src, zarrdest, cname, clevel, shuffle):
    start_time = time.time()
    compressor = Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
    import_datasets(n5src, zarrdest, compressor)
    total_time = time.time() - start_time
    print(f"Total conversion time: {total_time} s")

if __name__ ==  '__main__':

    num_cores = 10
    cluster = LSFCluster( cores=num_cores,
            processes=1,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="30:00"
            )
    cluster.scale(num_cores)

    with Client(cluster) as cl:        
        cl.compute(cli(), sync=True)

# if __name__ == '__main__':
#     cli()
