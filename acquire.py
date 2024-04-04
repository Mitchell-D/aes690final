import shlex
from subprocess import Popen, PIPE

def download(target_url:str, dest_dir:Path, raw_token:str=None,
             token_file:Path=None, replace:bool=False, debug=False):
    """
    Download a file with a wget subprocess invoking an authorization token.

    Generate a token here:
    https://ladsweb.modaps.eosdis.nasa.gov/profiles/#generate-token-modal

    :@param target_url: File path, probably provided by query_product().
    :@param dest_dir: Directory to download the new file into.
    :@param token_file: ASCII text file containing only a LAADS DAAC API token,
            which can be generated using the link above.
    """
    if not raw_token and not token_file:
        raise ValueError(f"You must provide a raw_token string or token_file.")

    if token_file:
        token = token_file.open("r").read().strip()
    else:
        token = raw_token
    #result = requests.get(target_url, stream=True, headers={
    #    'Authorization': k'Bearer {token}'})
    dest_path = dest_dir.joinpath(Path(target_url).name)
    if dest_path.exists():
        if not replace:
            #raise ValueError(f"File exists: {dest_path.as_posix()}")
            print(f"WARNING: file exists: {dest_path.as_posix()}")
            return dest_path
        dest_path.unlink()
    command = f"wget -e robots=off -np - -nH --cut-dirs=4 {target_url}" + \
            f' --header "Authorization: Bearer {token}"' + \
            f" -P {dest_dir.as_posix()}"
    if debug:
        print(f"\033[33;7mDownloading\033[0m \033[34;1;4m{target_url}\033[0m")
    stdout, stderr = Popen(
            shlex.split(command), stdout=PIPE, stderr=PIPE
            ).communicate()
    if stderr:
        print(stderr)
    return dest_path

def query_modis_l1b(product_key:str, start_time:dt, end_time:dt,
                   latlon:tuple=None, archive=None, day_only:bool=False,
                   debug:bool=False):
    """
    Query the LAADS DAAC for MODIS L2 MOD021KM (Terra) and MYD021KM (Aqua)
    calibrated surface reflectance and select brightness temperatures.
    Instead of 1KM (for 1km resolution), products may use HKM or QKM
    substrings for half-kilometer and quarter-kilometer resolution bands,
    respectively.

    :@param product_key: One of the listed VIIRS l1b product keys
    :@param start_time: Inclusive start time of the desired range. Only files
            that were acquired at or after the provided time may be returned.
    :@param end_time: Inclusive end time of the desired range. Only files
            that were acquired at or before the provided time may be returned.
    :@param archive: Some products have multiple archives, which seem to be
            identical. If archive is provided and is a validarchive set,
            uses the provided value. Otherwise defualts to defaultArchiveSet
            as specified in the product API response.

    :@return: A list of dictionaries containing the aquisition time of the
            granule, the data granule download link, and optionally the
            download link of the geolocation file for the granule.
    """
    valid = {"MOD021KM", "MYD021KM", "MOD02QKM", "MYD02QKM",
             "MOD02HKM", "MYD02HKM"}
    if product_key not in valid:
        raise ValueError(f"Product key must be one of: {valid}")

    products = laads.query_product(product_key, start_time, end_time, latlon,
                                   archive=archive, debug=debug)
    for i in range(len(products)):
        products[i].update({"atime":parse_modis_time(Path(
                    products[i]['downloadsLink']))})
    products = [ p for p in products
            if p["illuminations"] == "D" or not day_only ]

    return list(sorted(products, key=lambda p:p["atime"]))

def query_product(product_key:str, start_time:dt, end_time:dt,
                  latlon:tuple=None, archive=None, debug=False, _pinfo=None):
    """
    Use the CGI component of the API to query data files for any product within
    an inclusive time range. Optionally specify a latitude/longitude geographic
    location that must be contained within the data.

    https://ladsweb.modaps.eosdis.nasa.gov/api/v2/measurements/products

    :@param product_key: string key of the desired product. For valid options,
            see get_all_products(), or the LAADS DAAC link above.
    :@param start_time: datetime of the first valid minute in the range
    :@param end_time: datetime of the last valid minute in the range
    :@param latlon: tuple (lat, lon) in degrees specifying a location that must
            be contained within the data swath.
    :@param archive: Sometimes apparently-identical products are stored at
            multiple endpoints. If no archive is provided, defaults to the
            value provided in the API response.
    :@param _pinfo: Hidden workaround to provide the output of get_product_info
            for this product instead of querying the API for it again.
    """
    # Query the product info dict
    pinfo = _pinfo if not _pinfo is None else \
            get_product_info(product_key, debug=debug)
    # Determine which archive to use
    if not archive is None and archive not in pinfo["archives"].keys():
        raise ValueError(f"Provided archive {archive} not a valid option " + \
            f"for product {product_key};\nvalid: {pinfo['archives'].keys()}")
    archive = str(archive) if not archive is None \
            else str(list(pinfo["archives"].keys())[0])
    url = api_root + "/content/details?products=" + product_key
    url += "&temporalRanges=" + start_time.strftime('%Y-%jT%H:%M')
    url += ".." + end_time.strftime('%Y-%jT%H:%M')
    if not latlon is None:
        # The API is currently forgiving enough to allow identical N/S and E/W
        # coordinates in a boundary box, which lets us query at a point without
        # worrying about overlapping a pole or the dateline.
        lat, lon = latlon
        url += f"&regions=[BBOX]W{lon} N{lat} E{lon} S{lat}"

    def recursive_page_get(url):
        """
        Small internal recursive method to aggregate all pages. This dubiously
        trusts that the API won't loop nextPageLinks, but whatever.
        """
        if debug: print(f"\033[32;1mQuerying new page: \033[34;0m{url}\033[0m")
        result = requests.get(url)
        if result.status_code != 200:
            raise ValueError(f"Invalid query. See response:\n{result.text}")
        res_dict = json.loads(result.text)
        next_page = res_dict.get('nextPageLink')
        this_result = [ {k:c[k] for k in ("downloadsLink", "illuminations")}
                       for c in res_dict["content"]
                       if str(c["archiveSets"])==archive ]
        if next_page is None or url == next_page:
            return this_result
        return this_result + recursive_page_get(next_page)
    return recursive_page_get(url)


def get_product_info(product_key:str, print_info:bool=False, debug=False):
    """
    Query the LAADS DAAC API for a specific product key, and return a dict
    of useful information about that product.

    https://ladsweb.modaps.eosdis.nasa.gov/api/v2/measurements/products

    :@param product_key: string key of the desired product. For valid options,
            see get_all_products(), or the LAADS DAAC link above.
    :@param print_info: Prints the dataset IDs and a link to the API download
            directory tree if True.
    """
    # Query the API with the provided product key
    if debug is True: print("Querying LAADS API...")
    resp = requests.get(f"{api_root}/measurements/products/{product_key}")
    if resp.status_code == 400:
        raise ValueError(f"Invalid product key: {product_key}\n" +
                         "See options with get_all_products(print_info=True).")

    # Return the product info as a dictionary, printing links to the data
    # download if requested.
    product_info = json.loads(resp.text)[product_key]
    product_info["archives"] = {
            c:f"{api_root}/content/details/allData/{c}/{product_key}"
            for c in list(product_info["archiveSet"].keys()) }
    product_info["descriptionLink"] = laads_root + \
            product_info["descriptionLink"]
    del product_info["archiveSet"]
    #del product_info["collection"]
    if print_info:
        print(f"\033[1m{product_key}\033[0m")
        for k in product_info["archives"].keys():
            print(f"\033[92m    {k} \033[96m\033[4m" + \
                    f"{product_info['archives'][k]}\033[0m")
    return product_info

