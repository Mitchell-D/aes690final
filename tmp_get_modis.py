import h5py

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
    modis_path = Path(f"MOD03.A2020006.0120.061.2020006072240.hdf")
