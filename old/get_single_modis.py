from pathlib import Path
from datetime import datetime
from datetime import timedelta
import pickle as pkl
from pprint import pprint as ppt
import numpy as np
import argparse
import subprocess
import shlex

import krttdkit.visualize.guitools as gt
import krttdkit.operate.enhance as enh
import krttdkit.visualize.geoplot as gp
from krttdkit.visualize import TextFormat as TFmt
from krttdkit.products import MOD021KM
from krttdkit.operate import Recipe


token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"

l1b_bands = (
        3,  # 459-479nm blue
        10, # 483-493nm teal/blue
        4,  # 545-565nm green
        1,  # 620-670nm near-red
        2,  # 841-876nm NDWI / land boundaries
        16, # 862-877nm NIR / aerosol distinction
        19, # 916-965nm H2O absorption
        5,  # 1230-1250nm optical depth
        26, # 1360-1390nm cirrus band
        6,  # 1628-1652nm snow/ice band
        7,  # 2106-2155nm cloud particle size
        20, # 3660-3840nm SWIR
        21, # 3929-3989 another SWIR
        27, # 6535-6895nm Upper H2O absorption
        28, # 7175-7475nm Lower H2O absorption
        29, # 8400-8700nm Infrared cloud phase, emissivity diff 11-8.5um
        31, # 10780-11280nm clean LWIR
        32, # 11770-12270nm less clean LWIR
        33, # 14085-14385nm dirty LWIR
        )

debug=True
target_latlon=(33.18,-88.75)
satellite="terra"
target_time=datetime(2017,1,1,18)
subgrid_size = (400,400)

'''
data_dir = Path("data/modis")
tmp_path = MOD021KM.download_granule(
        data_dir=data_dir,
        raw_token = token,
        target_latlon=target_latlon,
        satellite=satellite,
        target_time=target_time,
        day_only=True,
        debug=debug,
        )
'''
tmp_path = Path("data/tmp_modis.hdf")

M = MOD021KM.from_hdf(tmp_path, l1b_bands)
#M.get_subgrid(target_latlon, *subgrid_size, from_center=True)
gt.quick_render(M.get_rgb("TC"))
