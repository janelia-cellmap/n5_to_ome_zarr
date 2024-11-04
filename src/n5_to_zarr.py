import zarr 
import os
import json
import click
import pydantic_zarr as pz
from numcodecs import Blosc
import os
from operator import itemgetter
from itertools import chain
import natsort

import dask.array as da
from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster


__version__ = "0.1.0"

#creates attributes.json, if missing 
def reconstruct_json(n5src):
    dir_list = os.listdir(n5src)
    if "attributes.json" not in dir_list:
        with open(os.path.join(n5src,"attributes.json"), "w") as jfile:
            dict = {"n5": "2.0.0"}
            jfile.write(json.dumps(dict, indent=4))
    for obj in dir_list:
        if os.path.isdir(os.path.join(n5src, obj)):
            reconstruct_json(os.path.join(n5src, obj))

def apply_ome_template(zgroup):
    
    f_zattrs_template = open('src/zarr_attrs_template.json')
    z_attrs = json.load(f_zattrs_template)
    f_zattrs_template.close()

    junits = open('src/unit_names.json')
    unit_names = json.load(junits)
    junits.close()

    units_list = []

    for unit in zgroup.attrs['units']:
        if unit in unit_names.keys():
            units_list.append(unit_names[unit])
        else:
            units_list.append(unit)

    #populate .zattrs
    z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
                                          "type": "space",
                                           "unit": unit} for (axis, unit) in zip(zgroup.attrs['axes'], 
                                                                                 units_list)]
    z_attrs['multiscales'][0]['version'] = '0.4'
    z_attrs['multiscales'][0]['name'] = zgroup.name
    z_attrs['multiscales'][0]['coordinateTransformations'] = [{"type": "scale",
                    "scale": [1.0, 1.0, 1.0]}, {"type" : "translation", "translation" : [1.0, 1.0, 1.0]}]
    
    return z_attrs

def normalize_to_omengff(zgroup):
    group_keys = zgroup.keys()
    
    for key in chain(group_keys, '/'):
        if isinstance(zgroup[key], zarr.hierarchy.Group):
            if key!='/':
                normalize_to_omengff(zgroup[key])
            if 'scales' in zgroup[key].attrs.asdict():
                zattrs = apply_ome_template(zgroup[key])
                zarrays = zgroup[key].arrays(recurse=True)

                unsorted_datasets = []
                for arr in zarrays:
                    unsorted_datasets.append(ome_dataset_metadata(arr[1], zgroup[key]))

                #1.apply natural sort to organize datasets metadata array for different resolution degrees (s0 -> s10)
                #2.add datasets metadata to the omengff template
                zattrs['multiscales'][0]['datasets'] = natsort.natsorted(unsorted_datasets, key=itemgetter(*['path']))
                zgroup[key].attrs['multiscales'] = zattrs['multiscales']


def ome_dataset_metadata(n5arr, group):
    
    arr_attrs_n5 = n5arr.attrs['transform']
    dataset_meta =  {
                    "path": os.path.relpath(n5arr.path, group.path),
                    "coordinateTransformations": [{
                        'type': 'scale',
                        'scale': arr_attrs_n5['scale']},{
                        'type': 'translation',
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
        elif isinstance(v,  dict):
            normalize_groupspec(v, comp)

def copy_n5_tree(n5_root, z_store, comp):
    spec_n5 = pz.GroupSpec.from_zarr(n5_root)
    spec_n5_dict = spec_n5.dict()
    normalize_groupspec(spec_n5_dict, comp)
    spec_n5 = pz.GroupSpec(**spec_n5_dict)
    return spec_n5.to_zarr(z_store, path= '')



def import_datasets(n5src, zarrdest, comp, repair_n5_attrs):

    if repair_n5_attrs:
            reconstruct_json(n5src)
    store_n5 = zarr.N5Store(n5src)
    n5_root = zarr.open_group(store_n5, mode = 'r')
    zarr_arrays = (n5_root.arrays(recurse=True))

    z_store = zarr.NestedDirectoryStore(zarrdest)
    zg = copy_n5_tree(n5_root, z_store, comp)

    normalize_to_omengff(zg)

    for item in zarr_arrays:
        n5arr = item[1]
        darray = da.from_array(n5arr, chunks = n5arr.chunks)
        dataset = zarr.open_array(os.path.join(zarrdest, n5arr.path), mode='a')
        
        da.store(darray, dataset, lock = False)


@click.command()
@click.argument('n5src', type=click.STRING)
@click.argument('zarrdest', type=click.STRING)
@click.option('--cluster', '-c', default = "lsf", type=click.STRING)
@click.option('--num_workers', '-w', default = 100, type=click.INT)
@click.option('--repair_n5_attrs', default= False, type=click.BOOL)
@click.option('--cname', default = "zstd", type=click.STRING)
@click.option('--clevel', default = 9, type=click.INT)
@click.option('--shuffle', default = 0, type=click.INT)
def cli(n5src, zarrdest, cname, clevel, shuffle, cluster, num_workers, repair_n5_attrs):
    compressor = Blosc(cname=cname, clevel=clevel, shuffle=shuffle)

    if cluster == "lsf":
        num_cores = 1
        cluster_dask = LSFCluster(
                cores=num_cores,
                processes=1,
                memory=f"{15 * num_cores}GB",
                ncpus=num_cores,
                mem=15 * num_cores,
                walltime="48:00",
                death_timeout = 240.0,
                local_directory = "/scratch/zubovy/"
                )
        cluster_dask.scale(num_workers)
    elif cluster == "local":
        cluster_dask = LocalCluster()

    with Client(cluster_dask) as cl:
        text_file = open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w")
        text_file.write(str(cl.dashboard_link))
        text_file.close()
        cl.compute(import_datasets(n5src, zarrdest, compressor, repair_n5_attrs), sync=True)

if __name__ == '__main__':
    cli()
