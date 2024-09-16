This script could be used for conversion of an .n5 file to .zarr format with OME-NGFF multiscales metadata structure.
#### How to run
1. open command line terminal
2. install poetry tool for dependency management and packaging: https://pypi.org/project/poetry/
3. switch to the n5_to_zarr/src directory:
    ``cd PATH_TO_DIRECTORY/n5_to_zarr/src``
4. install python dependencies:
    ``poetry install``
5. run script using cli (for lsf cluster):
   ``bsub -n 10 -J n5tozarr -o LOG_FILE_PATH 'umask 002; poetry run python src/n5_to_zarr.py "PATH_TO_SOURCE_DIRECTORY/input_file.n5" "PATH_TO_DEST_DIRECTORY/output_file.zarr" --cluster=lsf --num_workers=300';``
    num_workers and cluster parameters are optional.
