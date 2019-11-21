import os

# Path to light repo data  
_MNULFI_DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dat')

def dat_dir():
    return _MNULFI_DATA_DIR
