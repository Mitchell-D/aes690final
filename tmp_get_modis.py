#import h5py

## Must use pyhdf since it's an hdf4 :(
from pyhdf.SD import SD,SDC
from pathlib import Path
import numpy as np

def parse_modis_time(fpath:Path):
    """
    Use the VIIRS standard file naming scheme (for the MODAPS DAAC, at least)
    to parse the acquisition time of a viirs file.

    method taken from krttdkit.acquire.modis

    Typical files look like: VJ102IMG.A2022203.0000.002.2022203070756.nc
     - Field 1:     Satellite, data, and band type
     - Field 2,3:   Acquisition time like A%Y%j.%H%M
     - Field 4:     Collection (001 for Suomi NPP, 002 for JPSS-1)
     - Field 5:     File creation time like %Y%m%d%H%M
    """
    return dt.strptime("".join(fpath.name.split(".")[1:3]), "A%Y%j%H%M")

if __name__=="__main__":
    data_dir = Path("data")
    #modis_path = data_dir.joinpath(f"modis/MYD03.A2018001.0810.061.2018002030829.hdf")
    modis_path = data_dir.joinpath(f"modis/MYD021KM.A2018001.0810.061.2018002041512.hdf")

    #f = h5py.File(modis_path.as_posix(), mode="r")
    sd = SD(modis_path.as_posix(), SDC.READ)
    print(sd.datasets().keys())
    print(sd.select("Band_250M").get())
    print(sd.select("Band_500M").get())
    print(sd.select("Band_1KM_RefSB").get())
    print(sd.select("Band_1KM_Emissive").get())
