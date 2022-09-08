'''
The sarsar_admin module allows to deal with the postgis database in python: import s1 and s2 tiles, 
generate rasters by sar, compute stats, ...
'''
import psycopg2
import psycopg2.extras
import glob
import os
import decimal
import subprocess
import re
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.ndimage.filters as filters
import uuid
import tempfile
from osgeo import gdal

version = "8.4"

release_notes = (
    "8.4: add SAR sites management abilities (import/recompute)\n"
    "8.3: add SAR sites management abilities (activate/deactivate/delete)\n"
    "8.2: reproject sentinel-1 tiles when in SRID EPSG:32632 (UTM 32N)\n"
    "8.1: integrate database catalog to keep information about imported sentinel tiles\n"
    "8.0: integrate COMPO related abilities\n"
)

# ====================================================================================================================================
# S1 functions
# ====================================================================================================================================

def _generate_s1_tiles_entries_vvvh(paths, channel):
    entries = []
    for path in paths:
        tokens = path.split('/')
        entry = {"type": "s1", "date": tokens[5]+tokens[6]+tokens[7], "band": channel.lower(), "path": path}
        entries.append(entry)
    return entries


def _generate_s1_tiles_entries(paths, channel, orbit):
    entries = []
    for path in paths:
        tokens = path.split('/')
        entry = {"type": "s1", "date": tokens[5]+tokens[6]+tokens[7], "band": channel.lower()+orbit.lower(), "path": path}
        entries.append(entry)
    return entries


def find_s1_tiles_vvvh(yyyy, mm, dd, min_results=-1):
    '''
    Allow to find all S1 tiles on Terrascope disks that intersect with Wallonia and satisfy given parameters
    This is the version that returns 'vv' and 'vh' bands 
    
        Parameters:
            yyyy (str): the year for which we search tiles (in format `yyyy` or '*' to fetch all years)
            mm (str): the month for which we search tiles (in format `mm` or '*' to fetch all months)
            dd (str): the day for which we search tiles (in format `dd` or '*' to fetch all days)
            min_results (int): stop search as soon as a certain number of tiles are found (default: -1, 
                which means `take all`)
        
        Returns:
            A list of tiles (each tile is a dictionnary) that can serve as input for other functions
    '''
    orbits = ['8','37','59','88','110','139','161']
    channels = ['VH','VV']
    datadir = '/data/MTDA/CGS_S1/CGS_S1_GRD_SIGMA0_L1/'
    tiles = []
    for channel in channels:
        for orbit in orbits: 
            found = glob.glob(datadir + yyyy + '/' + mm + '/' + dd + '/*/*_' + orbit + '_*' + channel + '*', recursive = True)
            tiles += _generate_s1_tiles_entries_vvvh(found, channel)
            if min_results > 0 and len(tiles) >= min_results :
                return tiles[0:min_results]
    return tiles


def find_s1_tiles(yyyy, mm, dd, min_results=-1):
    '''
    Allow to find all S1 tiles on Terrascope disks that intersect with Wallonia and satisfy given parameters
    This is the version that returns separate tiles by orbits: 'vv8', 'vh8', 'vv33', 'vh33', etc
    
        Parameters:
            yyyy (str): the year for which we search tiles (in format `yyyy` or '*' to fetch all years)
            mm (str): the month for which we search tiles (in format `mm` or '*' to fetch all months)
            dd (str): the day for which we search tiles (in format `dd` or '*' to fetch all days)
            min_results (int): stop search as soon as a certain number of tiles are found (default: -1, 
                which means `take all`)
        
        Returns:
            A list of tiles (each tile is a dictionnary) that can serve as input for other functions
    '''
    orbits = ['8','37','59','88','110','139','161']
    channels = ['VH','VV']
    datadir = '/data/MTDA/CGS_S1/CGS_S1_GRD_SIGMA0_L1/'
    tiles = []
    for channel in channels:
        for orbit in orbits: 
            found = glob.glob(datadir + yyyy + '/' + mm + '/' + dd + '/*/*_' + orbit + '_*' + channel + '*', recursive = True)
            tiles += _generate_s1_tiles_entries(found, channel, orbit)
            if min_results > 0 and len(tiles) >= min_results :
                return tiles[0:min_results]
    return tiles

# ====================================================================================================================================
# S2 functions
# ====================================================================================================================================

def _find_s2_tiles_path_mgrs(yyyy, mm, dd):
    wal_mgrs_tiles = ['31UDS', '31UER', '31UES', '31UFQ', '31UFR', '31UFS', '31UGR', '31UGS']
    datadir = '/data/MTDA/TERRASCOPE_Sentinel2/TOC_V2/'
    ret = []
    for wal_mgrs_tile in wal_mgrs_tiles:
        found = glob.glob(datadir + yyyy + '/' + mm + '/' + dd + '/*' + wal_mgrs_tile + '*/')
        ret += found
    return sorted(ret)


def _fetch_md_xml_path(s2_tiles_path):
    xml_paths = []
    for path in s2_tiles_path:
        print('... scanning "%s" folder' % (path))
        for file in os.listdir(path):
            if file.endswith(".xml"):
                #print(os.path.join(path,file))
                xml_paths.append(os.path.join(path,file))
    return sorted(xml_paths)


def _extract_cloud_cov_percentage(xml_paths):
    print('> Cloud coverage percentage extraction, from %s metadata XML file...' % (len(xml_paths)))
    ret = []
    for xml_path in xml_paths:
        with open (xml_path, 'rt') as xml_file:
            content = xml_file.read()
            #print(xml_path)
            result = re.search('<gmd:cloudCoverPercentage><gco:Real>(.*)</gco:Real></gmd:cloudCoverPercentage>', content)
            #print(result.group(1))
            ret.append([os.path.dirname(xml_path),result.group(1)])
    return ret


def _find_s2_tiles_path(yyyy, mm, dd, min_cloud_cov, max_cloud_cov):
    print('> Be patient! This search can take a while...')
    xmls = _fetch_md_xml_path(_find_s2_tiles_path_mgrs(yyyy,mm,dd))
    percents = _extract_cloud_cov_percentage(xmls)
    ret = []
    for percent in percents:
        if float(percent[1]) >= min_cloud_cov and float(percent[1]) <= max_cloud_cov:
            #print('OK ! %s => %s' % (percent[0], percent[1]))
            ret.append(percent)
        #else:
        #    print('KO ! %s => %s' % (percent[0], percent[1]))
    return ret


def _generate_s2_tiles_entries(tiles_w_percent):
    band_names = {'B02_10M': 'b02', 'B03_10M': 'b03', 'B04_10M': 'b04', 'B08_10M': 'b08', 'SCENECLASSIFICATION_20M': 'scl'}
    entries = []
    for tile_w_percent in tiles_w_percent:
        path = tile_w_percent[0]
        tokens = path.split('/')
        for band_name in band_names.keys():
            found = sorted(glob.glob(path + '/*' + band_name + '*.tif'))
            for ent in found:
                entry = {"type": "s2", "date": tokens[5]+tokens[6]+tokens[7], "band": band_names[band_name], "path": ent, "cloud_cov": tile_w_percent[1]}
                #print (entry)
                entries.append(entry)
    return entries


def find_s2_tiles(yyyy, mm, dd, min_cloud_cov, max_cloud_cov):
    '''
    Allow to find all S2 tiles on Terrascope disks that intersect with Wallonia and satisfy given parameters
    
        Parameters:
            yyyy (str): the year for which we search tiles (in format `yyyy` or '*' to fetch all years)
            mm (str): the month for which we search tiles (in format `mm` or '*' to fetch all months)
            dd (str): the day for which we search tiles (in format `dd` or '*' to fetch all days)
            min_cloud_cov (int): the minimal cloud coverage of tiles we search (must be >= 0)
            max_cloud_cov (int): the maximal cloud coverage of tiles we search (must be <= 100)
        
        Returns:
            A list of tiles (each tile is a dictionnary) that can serve as input for other functions
    '''
    tiles_w_percent = _find_s2_tiles_path(yyyy, mm, dd, min_cloud_cov, max_cloud_cov)
    return _generate_s2_tiles_entries(tiles_w_percent)


# ====================================================================================================================================
# PostGIS prepare functions
# ====================================================================================================================================

_db_credentials = None
_db_conn = None


def prepare_env(db_credentials):
    '''
    The function allowing to prepare the connections to the database
    
        Parameters:
            db_credentials (dict): the credentials allowing to connect to the database
    '''
    global _db_credentials
    _db_credentials = db_credentials

    # before working with UUID objects in PostgreSQL
    psycopg2.extras.register_uuid()


def _create_or_get_db_connection():
    global _db_conn
    reconnection_required = True
    cur = None
    if _db_conn is not None:
        try:
            cur = _db_conn.cursor()
            cur.execute('SELECT 1')
            reconnection_required = False
        except psycopg2.InterfaceError:
            pass
        except psycopg2.errors.InFailedSqlTransaction:
            _db_conn.rollback()
        finally:
            if cur is not None:
                cur.close()
    if reconnection_required:
        _db_conn = psycopg2.connect(
            host=_db_credentials["host"],
            database=_db_credentials["db"],
            user=_db_credentials["user"],
            connect_timeout=5)
    return _db_conn;

def release_env():
    '''
    The function that should always be executed when the database interactions are finished, in order to 
    release all resources safely
    '''
    global _db_conn
    if _db_conn is not None:
        _db_conn.close()
        print('Database connection closed.')

# ====================================================================================================================================
# Process functions
# ====================================================================================================================================


def _check_table_exists(table_name):
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select exists(select from pg_tables where schemaname = 'public' and tablename = %s)", [table_name])
        result = cur.fetchone()[0]
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error: 
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def _add_additional_rasterid_column(table_name):
    conn = _create_or_get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            "alter table " + table_name + " add column raster_id uuid")
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()


def _populate_tile_catalog(tile, raster_id, schema_name, table_name, original_raster_epsg_proj):
    conn = _create_or_get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            "insert into sar_rasters_catalog(raster_id, raster_type, raster_band, acq_date, schema_name, "
            "table_name, tif_path, original_raster_epsg_proj) values (%s, %s, %s, %s, %s, %s, %s, %s)",
            (raster_id, tile["type"], tile["band"], tile["date"], schema_name, table_name, tile["path"], original_raster_epsg_proj))
        conn.commit()
        cur.execute(
            "update " + table_name + " set raster_id = %s where raster_id is null", (raster_id, )
        )
        conn.commit()
        cur.execute(
            "update sar_rasters_catalog set geom = compute_sentinel_raw_table_polygon(%s, %s) where raster_id = %s",
            (table_name, raster_id, raster_id,)
        )
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()


def _reproject_sentinel_tile_in_32631(original_tile_path, tmp_dir, dry_run):
    input_raster = gdal.Open(original_tile_path)
    output_raster = os.path.join(tmp_dir, os.path.basename(original_tile_path))
    if dry_run:
        print('> ... DRY RUN ! Should reproject in EPSG:32631 here')
    else:
        warp_opts = gdal.WarpOptions(
            dstSRS='EPSG:32631',
            xRes='10.000000000000000',
            yRes='-10.000000000000000',
            targetAlignedPixels=True
        )
        warp = gdal.Warp(output_raster, input_raster, options=warp_opts)
        warp = None  # Closes the files
        repr_raster_proj = int(gdal.Info(output_raster, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
        print("> ... Reprojected raster's projection is EPSG:%s" % repr_raster_proj)
    print('> ... Temp reprojected raster file is %s' % output_raster)
    return output_raster


def load_sentinel_tile(tile, dry_run=False): 
    '''
    Load the Sentinel `tile` (using the `raster2pgsql` command) in the database
    
        Parameters:
            tile (dict): a tile as returned by the `find_s1_tiles` or `find_s2_tiles` functions (which 
                returns list of tiles).
            dry_run (bool): if True, only print the `raster2pgsql` import command instead of executing 
                it (Default: False)
    '''
    with tempfile.TemporaryDirectory() as tmp_dir:
        print('> ... Create temporary working directory', tmp_dir)

        table_name = tile["type"] + '_' + tile["date"] + "_" + tile["band"]
        original_tile_path = tile["path"]
        toimport_tile_path = original_tile_path

        # Create table or append to existing ?
        table_exists = _check_table_exists(table_name)
        raster_opt = '-a' if table_exists else '-c -I'

        # get raster's projection
        # Renato's version
        # original_raster_epsg_proj = int(gdal.Info(original_tile_path, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
        # Gerard's version
        original_raster_epsg_proj = int(gdal.Info(original_tile_path, format='json')['coordinateSystem']['wkt'].rsplit("EPSG", 1)[-1].strip('",]'))

        if original_raster_epsg_proj != 32631:
            print("> ... WARNING ! Raster's projection is EPSG:%s. It will temporarily be reprojected in EPSG:32631 for import in DB" % original_raster_epsg_proj)
            toimport_tile_path = _reproject_sentinel_tile_in_32631(original_tile_path, tmp_dir, dry_run)
        else:
            print("> ... Raster's projection is EPSG:%s (the one expected)" % original_raster_epsg_proj)

        # Build cmd string
        raster2psql_exec = '/usr/pgsql-10/bin/raster2pgsql'
        # raster2psql_exec = 'raster2pgsql.exe'
        cmd = raster2psql_exec + ' -s 32631 {} -t 100x100 {} {} | psql -q -h {} -U {} -d {}'.format(raster_opt, toimport_tile_path, table_name, _db_credentials["host"], _db_credentials["user"], _db_credentials["db"])
        # print('command is %s' % (cmd))

        print('> ... Importing "%s" in DB table "%s" with options "%s"' % (os.path.basename(original_tile_path), table_name, raster_opt))

        # Execute
        if dry_run:
            print('> ... DRY RUN ! Command is "%s"' % (cmd))
        else:
            subprocess.run(cmd, shell=True)
            # Add additional column if ne table
            if not table_exists:
                _add_additional_rasterid_column(table_name)
                # insert entry in catalog
            print('> ... populate tile catalog')
            raster_id = uuid.uuid4()
            _populate_tile_catalog(tile, raster_id, "public", table_name, original_raster_epsg_proj)


def get_products(tiles):
    '''
    Produce a list of products (band for a given date, f.i. `b02` for `20210612`) based on a list of `tiles`
    
        Parameters:
            tiles (list): a list of tiles as produced by the `find_s1_tiles` or `find_s2_tiles` functions
        
        Returns:
            a list of products
    '''
    temp_set = set()
    products = []
    for tile in tiles: 
        key = tile["type"] + "_" + tile["date"] + "_" + tile["band"]
        if key not in temp_set:
            product = tile.copy()
            del product["path"]
            temp_set.add(key)
            product["key"] = key
            products.append(product)
    return products


def get_processing_dates(products):
    '''
    Produce a list of dates corresponding to the given products list
    
        Parameters:
            products (list): a list of products as produced by the `get_products` function
        
        Returns:
            a list of dates expressed in format `yyyymmdd`
    '''
    temp_set = set()
    proc_dates = []
    for product in products:
        key = product["type"] + "_" + product["date"]
        if key not in temp_set:
            proc_date = product.copy()
            del proc_date["band"]
            temp_set.add(key)
            proc_date["key"] = key
            proc_dates.append(proc_date)
    return proc_dates    


def clip_raster_on_wallonia(product):
    '''
    Trigger the clipping, in database, of the given product (a `band` for a given date) on the 
    Walloon region.
    This processing allow to drop all part of imagery that is clearly outside wallonia...
    
        Parameters:
            product (dict): a product as produced by the `get_products` function
    '''
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select crop_sar_raw_rasters(%s, %s, %s)", [product["date"], product["type"], product["band"]])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def compute_sar_raw_rasters(product):
    '''
    Trigger the computing, in database, of tiny raw rasters for each SAR intersecting the given product.
    
        Parameters:
            product (dict): a product as produced by the `get_products` function
    '''
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
        cur.execute("select compute_sar_raw_rasters(%s, %s, %s)", [product["date"], product["type"], product["band"]])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result



def compute_sar_index_stats(date, type, index, compute_cf_only):
    '''
    Trigger the computing of index statistics in the database.
    This function is usually called after the `compute_sar_raw_rasters` one.
    
        Parameters:
            date (str): the given date, in format `yyyymmdd`
            type (str): `s1` or `s2`, depending of the Sentinel data required to compute the 
                given index
            index (str): the index name (`NDVI`, `NDWI`, `BI`, `VH`, etc)
            compute_cf_only (bool): if true, trigger the computing only for SAR having 0 pixels 
                under cloud/shadow/snow
    '''
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select compute_sar_index_stats(%s, %s, %s, %s)", [date, type, index, compute_cf_only])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def _clean_stats_dict(stats_dict):
    stats_dict['pixel_count'] = int(stats_dict['pixel_count'])
    stats_dict['expected_pixel_count'] = int(stats_dict['expected_pixel_count'])
    stats_dict['index_max'] = float(stats_dict['index_max'])
    stats_dict['index_mean'] = float(stats_dict['index_mean'])
    stats_dict['index_min'] = float(stats_dict['index_min'])
    stats_dict['index_stddev'] = float(stats_dict['index_stddev'])
    return stats_dict

def fetch_sar_index_stats_by_sar_id(sar_id_segment, index, start_incl_date='2015-07-01', end_incl_date='2099-12-31'):
    '''
    Fetch all raw stats corresponding to the given SAR ID Segment for the specified index 
    (NDVI, etc), for whatever acquisition date
    
        Parameters:
            sar_id_segment (str): the SAR ID Segment identifier
            index (str): the index name (`NDVI`, `NDWI`, `BI`, `VH`, etc)
            start_incl_date (str): the given start date, in format `yyyy-mm-dd`
            end_incl_date (str): the given end date, in format `yyyy-mm-dd`
        
        Returns:
            a list of raw statistics (presented as dict), that can be processed in functions 
                `compute_raw_time_series_stats` and `compute_smoothed_time_series_stats`
    '''
    conn = _create_or_get_db_connection()
    cur = None
    dict_result = []
    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
        cur.execute("select * from fetch_sar_index_stats_by_sar_id(%s, %s) where acq_date between %s and %s;",
                    [sar_id_segment, index, start_incl_date, end_incl_date])
        result = cur.fetchall() 
        dict_result = []
        for row in result:
            dict_result.append(dict(row))
        cur.close()
        dict_result = list(map(_clean_stats_dict, dict_result))
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return dict_result


def fetch_sar_index_stats_by_acq_date(acq_date, index):
    '''
    Fetch all raw stats corresponding to the given acquisition date for the specified index 
    (NDVI, etc), for whatever SAR ID Segment
    
        Parameters:
            acq_date (str): the given date, in format `yyyymmdd`
            index (str): the index name (`NDVI`, `NDWI`, `BI`, etc)
        
        Returns:
            a list of raw statistics (presented as dict)
    '''
    conn = _create_or_get_db_connection()
    cur = None
    dict_result = []
    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
        cur.execute("select * from fetch_sar_index_stats_by_acq_date(%s, %s);", [acq_date, index])
        result = cur.fetchall() 
        dict_result = []
        for row in result:
            dict_result.append(dict(row))
        cur.close()
        dict_result = list(map(_clean_stats_dict, dict_result))
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return dict_result


def remove_outliers_from_stats(raw_stats_as_dict):
    '''
    Remove outliers from the list of raw statistics
    
        Parameters:
            raw_stats_as_dict (list): the list of raw statistics (presented as dict), returned by 
                the function `fetch_sar_index_stats_by_sar_id`
        
        Returns:
            the exact same list without any outliers (depend of the index)
    '''
    no_outlier_stats = []
    for stat in raw_stats_as_dict:
        stat_index_name = stat['index_name']
        stat_index_mean = stat['index_mean']
        if stat_index_name in ['BAI','NDVI','NDWI2']:
            if stat_index_mean != 0:
                no_outlier_stats.append(stat)
        elif stat_index_name in ['BI','BI2']:
            if stat_index_mean != 21000:
                no_outlier_stats.append(stat)
        elif 'SBI' == stat_index_name:
            if stat_index_mean <= 20000:
                no_outlier_stats.append(stat)
    return no_outlier_stats


def compute_raw_time_series_stats(raw_stats_as_dict):
    '''
    Compute a time series of raw statistics, based on a list of raw statistics (presented as dict) 
    coming from the database.
    
        Parameters:
            raw_stats_as_dict (list): the list of raw statistics (presented as dict), returned by 
                the function `fetch_sar_index_stats_by_sar_id`
        
        Returns:
            a tuple of 2 arrays: first is an array of dates, second is an array of corresponding 
            raw stats 
    '''
    raw_date = []
    raw_index_mean = []
    for stat in raw_stats_as_dict:
        if not math.isnan(stat['index_mean']):
            raw_date.append(stat['acq_date'])
            raw_index_mean.append(stat['index_mean'])
    return [raw_date, raw_index_mean]


def compute_smoothed_time_series_stats(raw_stats_as_dict, time_resampling='D', gaussian_sigma=61):
    '''
    Compute a time series of smoothed statistics, based on a list of raw statistics (presented as dict) 
    coming from the database.
    The smoothing is the result of two processings: first a time resampling, then a gaussian filter.

        Parameters:
            raw_stats_as_dict (list): the list of raw statistics (presented as dict), returned by the 
                function `fetch_sar_index_stats_by_sar_id`
            time_resampling (str): the unit of time resampling (default: 'D' for day resampling)
            gaussian_sigma (int): the sigma applied by the gaussian filter (default: 61)

        Returns:
            a tuple of 2 arrays: first is an array of interpolated dates, second is an array of 
            corresponding smoothed stats 
    '''
    raw_time_series = compute_raw_time_series_stats(raw_stats_as_dict)
    raw_date = raw_time_series[0]
    raw_index_mean = raw_time_series[1]
    
    # Time-serie interpolation
    timestamps = pd.to_datetime(raw_date)
    #print("VALUES : ", values)
    #print("TIMESTAMPS : ", timestamps)
    ts = pd.Series(raw_index_mean, index=timestamps)
    ts = ts.resample(time_resampling).mean()
    temp_data = ts.interpolate()

    interpolated_date = temp_data.index.to_pydatetime()
    interpolated_index_mean = temp_data.tolist()

    # Gaussian filter
    smoothed_index_mean = filters.gaussian_filter1d(interpolated_index_mean,sigma=gaussian_sigma)
    #print(smoothed_index_mean)
    
    return [interpolated_date, smoothed_index_mean]


def clear_full_import_date(acq_date):
    '''
    Clear all the computed data (rasters, stats, ...) in the database for a given date
    
        Parameters:
            acq_date (str): the given date, in format `yyyymmdd`
    '''
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select clear_full_import_date(%s)", [acq_date])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


# ====================================================================================================================================
# COMPO functions
# ====================================================================================================================================


def compute_sar_compo_stats_by_sar_id(sar_id_segment, start_incl_date, end_incl_date):
    """
    Trigger the computing of the compo statistics in the database, based on BI/BI2/SBI previously computed stats.

        Parameters:
            sar_id_segment (str): the SAR ID Segment identifier
            start_incl_date (str): the given start date, in format `yyyymmdd`
            end_incl_date (str): the given end date, in format `yyyymmdd`
    """
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select compute_sar_compo_stats_by_sar_id(%s, %s, %s)", [sar_id_segment, start_incl_date, end_incl_date])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def _clean_compo_stats_dict(stats_dict):
    for key in stats_dict.keys():
        if type(stats_dict[key]) is decimal.Decimal:
            stats_dict[key] = float(stats_dict[key])
    return stats_dict


def fetch_sar_compo_stats_by_sar_id(sar_id_segment, start_incl_date, end_incl_date):
    """
    Fetch all compo stats corresponding to the given SAR ID Segment computed between the two provided dates

        Parameters:
            sar_id_segment (str): the SAR ID Segment identifier
            start_incl_date (str): the given start date, in format `yyyymmdd`
            end_incl_date (str): the given end date, in format `yyyymmdd`

        Returns:
            a list of raw statistics (presented as dict), that can be processed in functions
                `compute_compo_time_series_stats` and `compute_smoothed_compo_time_series_stats`
    """
    conn = _create_or_get_db_connection()
    cur = None
    dict_result = []
    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("select * from fetch_sar_compo_stats_by_sar_id(%s, %s, %s);", [sar_id_segment, start_incl_date, end_incl_date])

        result = cur.fetchall()
        dict_result = []
        for row in result:
            dict_result.append(dict(row))
        cur.close()
        dict_result = list(map(_clean_compo_stats_dict, dict_result))
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return dict_result


def compute_compo_time_series_stats(compo_stats_as_dict):
    """
    Compute a time series of compo statistics, based on a list of compo statistics (presented as dict)
    coming from the database.

        Parameters:
            compo_stats_as_dict (list): the list of compo statistics (presented as dict), returned by
                the function `fetch_sar_compo_stats_by_sar_id`

        Returns:
            a tuple of 2 arrays: first is an array of dates, second is an array of corresponding
            compo stats
    """
    compo_date = []
    compo_value = []
    for stat in compo_stats_as_dict:
        if not math.isnan(stat['compo_value']):
            compo_date.append(stat['acq_date'])
            compo_value.append(stat['compo_value'])
    return [compo_date, compo_value]


def compute_smoothed_compo_time_series_stats(compo_stats_as_dict, time_resampling='D', gaussian_sigma=61):
    """
    Compute a time series of smoothed compo statistics, based on a list of raw compo statistics (presented as dict)
    coming from the database.
    The smoothing is the result of two processings: first a time resampling, then a gaussian filter.

        Parameters:
            compo_stats_as_dict (list): the list of raw compo statistics (presented as dict), returned by the
                function `fetch_sar_compo_stats_by_sar_id`
            time_resampling (str): the unit of time resampling (default: 'D' for day resampling)
            gaussian_sigma (int): the sigma applied by the gaussian filter (default: 61)

        Returns:
            a tuple of 2 arrays: first is an array of interpolated dates, second is an array of
            corresponding smoothed stats
    """
    raw_time_series = compute_compo_time_series_stats(compo_stats_as_dict)
    raw_date = raw_time_series[0]
    raw_index_mean = raw_time_series[1]

    # Time-serie interpolation
    timestamps = pd.to_datetime(raw_date)
    ts = pd.Series(raw_index_mean, index=timestamps)
    ts = ts.resample(time_resampling).mean()
    temp_data = ts.interpolate()

    interpolated_date = temp_data.index.to_pydatetime()
    interpolated_index_mean = temp_data.tolist()

    # Gaussian filter
    smoothed_index_mean = filters.gaussian_filter1d(interpolated_index_mean, sigma=gaussian_sigma)

    return [interpolated_date, smoothed_index_mean]

# ====================================================================================================================================
# CATALOG functions
# ====================================================================================================================================


def find_sentinel_tile_in_catalog(sar_id_segment, acq_date, raster_type):
    """
    Find the path of the Sentinel tile (s1 or s2) from which data have been used to compute stats on the given
    SAR Id Segment, for the given acquisition date.

    Note: The tile must absolutely have been imported in the catalog before the call!

        Parameters:
            sar_id_segment (str): the SAR ID Segment identifier
            acq_date (str): the given date, in format `yyyymmdd`
            raster_type (str): `s1` or `s2`

        Returns:
            the path to the tif file if found
    """
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select find_sentinel_tile_in_catalog(%s, %s, %s);", [sar_id_segment, acq_date, raster_type])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result

# ====================================================================================================================================
# SAR Management functions
# ====================================================================================================================================


def manage_sar_enable(sar_id_segment):
    """
    Mark a SAR as 'active' in the database

        Parameters:
            sar_id_segment (str): the id_segment of the SAR
    """
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("update sar_def set is_active = TRUE where id_segment = %s", [sar_id_segment])
        result = cur.rowcount
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def manage_sar_disable(sar_id_segment):
    """
    Mark a SAR as 'inactive' in the database

        Parameters:
            sar_id_segment (str): the id_segment of the SAR
    """
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("update sar_def set is_active = FALSE where id_segment = %s", [sar_id_segment])
        result = cur.rowcount
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def manage_sar_delete_and_purge_database(sar_id_segment):
    """
    Delete a SAR and all its related rasters and statistics in the database, based on the SAR Id Segment
    This is a dangerous command, that cannot be reverted (data are definitively lost !)

        Parameters:
            sar_id_segment (str): the SAR ID Segment identifier
    """
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select clear_full_sar_data(%s);", [sar_id_segment])
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result


def manage_sar_import_shapefile(shp_file_path):
    """
        Import a shapefile with SAR definitions into the 'sar_def_temporary' table in the database
        This function is usually followed by a 'manage_sar_recompute_history_for_new_sar_sites' function call...

            Parameters:
                shp_file_path (str): the shapefile (.shp) full path
        """
    # gdal.UseExceptions()
    print(shp_file_path) #DEBUG
    gdal.SetConfigOption('CPL_DEBUG', 'ON')    
    conn_string = 'PG:host={} user={} dbname={}'.format(_db_credentials["host"], _db_credentials["user"], _db_credentials["db"])
    ds = gdal.OpenEx(conn_string, gdal.OF_VECTOR | gdal.OF_UPDATE)
    print("debug 1") #DEBUG
    gdal.VectorTranslate(
        ds,
        shp_file_path,
        options='-nln sar_def_temporary -nlt MULTIPOLYGON -append'
    )
    print(ds) #DEBUG
    ds = None
    print("debug 2") #DEBUG

def manage_sar_recompute_history_for_new_sar_sites():
    """
    Trigger, in the database, the full history computing for all SAR imported into the 'sar_def_temporary' table
    Once the history is fully computed, SAR are moved to the main 'sar_def' table

    This function is usually called after the 'manage_sar_import_shapefile'...
    """
    conn = _create_or_get_db_connection()
    cur = None
    result = None
    try:
        cur = conn.cursor()
        cur.execute("select recompute_history_for_new_sar_sites();")
        result = cur.fetchone()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
    return result