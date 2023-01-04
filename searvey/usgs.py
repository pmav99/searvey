"""
_`dataretrieval` package developed by USGS is used for the USGS stations

:: _`dataretrieval`: https://usgs-python.github.io/dataretrieval/

This pacakge is a think wrapper around NWIS _REST API:

:: _REST: https://waterservices.usgs.gov/rest/

We take the return values from `dataretrieval` to be the original data
"""

from __future__ import annotations

import datetime
import functools
import logging
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from itertools import product

import geopandas as gpd
import limits
import pandas as pd
import xarray as xr
from dataretrieval import nwis
from dataretrieval.codes import state_codes
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry import mapping
from shapely.geometry import shape

from .multi import multiprocess
from .multi import multithread
from .rate_limit import RateLimit
from .rate_limit import wait
from .utils import get_region
from .utils import merge_datasets


logger = logging.getLogger(__name__)

# TODO: Fix documentation

# TODO
#ADD a dict to be populated dynamically that gets the parameter code 
# for parameters include key words, e.g. "elevation" or "flow rate".
# these codes can be further filtered based on 'group', 'parm_unit', etc.
# These codes, when passed to what_sites or get_iv will return 
# only relevant values or stations that have these parameters


USGS_OUTPUT_OF_INTEREST = ("elevation", "flow rate")

# constants
USGS_RATE_LIMIT = limits.parse("5/second")
USGS_MAX_DAYS_PER_REQUEST = 30 # WHAT IS THE ACTUAL MAX?
USGS_STATIONS_KWARGS = [
    {"output": "general", "skip_table_rows": 4},
    {"output": "contacts", "skip_table_rows": 4},
    {"output": "performance", "skip_table_rows": 8},
]
USGS_STATION_DATA_COLUMNS_TO_DROP = [
    "bat",
    "sw1",
    "sw2",
    "(m)",
]
USGS_STATION_DATA_COLUMNS = {
}


def _filter_parameter_codes(param_cd_df: pd.DataFrame) -> pd.DataFrame:

    # Should we filter based on units?
    param_cd_df = param_cd_df[
        (param_cd_df.group == "Physical")
    ]

    return param_cd_df


@functools.cache
def _get_usgs_output_codes() -> Dict[str, str]:

    output_codes = {}
    for var in USGS_OUTPUT_OF_INTEREST:
        df_param_cd, _ = nwis.get_pmcodes(var)
        df_param_cd = _filter_parameter_codes(df_param_cd)
        output_codes[var] = df_param_cd.parameter_cd.values.tolist()

    return output_codes


def get_usgs_stations_by_output(output: str) -> pd.DataFrame:
    # TODO: Do we need this public functionality?
    return None

def _get_usgs_stations_by_output(output: List[str], **kwargs) -> pd.DataFrame:

    sites, sites_md = nwis.what_sites(
        parameterCd=output, 
        **kwargs
    )
    # TODO: Embed the info from sites_md into the main dataframe? md
    # object cannot be pickled due to lambda func
    return sites


def normalize_usgs_stations(df: pd.DataFrame) -> gpd.GeoDataFrame:
    # TODO: Process necessary fields

    gdf = gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(df.dec_long_va, df.dec_lat_va, crs="EPSG:4326"),
    )
    return gdf


@functools.cache
def _get_all_usgs_stations() -> gpd.GeoDataFrame:
    """
    Return USGS station metadata

    :return: ``geopandas.GeoDataFrame`` with the station metadata
    """


    usgs_stations_results = multiprocess(
        func=_get_usgs_stations_by_output,
        func_kwargs=[
            {'stateCd': st, 'output': out}
            for st, out in product(state_codes, _get_usgs_output_codes().values())
        ]
    )

    usgs_stations = functools.reduce(
        functools.partial(pd.merge, how='outer'),
        (r.result for r in usgs_stations_results)
    )
    usgs_stations = normalize_usgs_stations(usgs_stations)

    return usgs_stations



@functools.cache
def _get_usgs_stations_by_region(**region_json: Any) -> gpd.GeoDataFrame:

    region = shape(region_json)
    
    bBox = list(region.bounds)
    if (bBox[2] - bBox[0]) * (bBox[3] - bBox[1]) > 25:
        raise ValueError("Product of lat range and lon range cannot exceed 25!")

    usgs_stations_results = multiprocess(
        func=_get_usgs_stations_by_output,
        func_kwargs=[
            {'bBox': bBox, 'output': out}
            for out in _get_usgs_output_codes().values()
        ]
    )
    usgs_stations = functools.reduce(
        functools.partial(pd.merge, how='outer'),
        (r.result for r in usgs_stations_results)
    )
    usgs_stations = normalize_usgs_stations(usgs_stations)

    usgs_stations = usgs_stations[usgs_stations.within(region)]

    return usgs_stations


def get_usgs_stations(
    region: Optional[Union[Polygon, MultiPolygon]] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Return IOC station metadata from: http://www.ioc-sealevelmonitoring.org/list.php?showall=all

    If `region` is defined then the stations that are outside of the region are
    filtered out.. If the coordinates of the Bounding Box are defined then
    stations outside of the BBox are filtered out. If both ``region`` and the
    Bounding Box are defined, then an exception is raised.

    Note: The longitudes of the IOC stations are in the [-180, 180] range.

    :param region: ``Polygon`` or ``MultiPolygon`` denoting region of interest
    :param lon_min: The minimum Longitude of the Bounding Box.
    :param lon_max: The maximum Longitude of the Bounding Box.
    :param lat_min: The minimum Latitude of the Bounding Box.
    :param lat_max: The maximum Latitude of the Bounding Box.
    :return: ``pandas.DataFrame`` with the station metadata
    """
    region = get_region(
        region=region,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        symmetric=True,
    )

    if region:
        # NOTE: Polygon is unhashable and cannot be used for a 
        # cached function input
        region_json = mapping(region)
        usgs_stations = _get_usgs_stations_by_region(**region_json)
    else:
        usgs_stations = _get_all_usgs_stations()

    return usgs_stations


def normalize_usgs_station_data(usgs_code: str, df: pd.DataFrame, truncate_seconds: bool) -> pd.DataFrame:
    # Each station may have more than one sensors.
    # Some of the sensors have nothing to do with sea level height. We drop these sensors
    df = df.rename(columns=IOC_STATION_DATA_COLUMNS)
    logger.debug("%s: df contains the following columns: %s", usgs_code, df.columns)
    df = df.drop(columns=IOC_STATION_DATA_COLUMNS_TO_DROP, errors="ignore")
    if len(df.columns) == 1:
        # all the data columns have been dropped!
        msg = f"{usgs_code}: The table does not contain any sensor data!"
        logger.info(msg)
        raise ValueError(msg)
    df = df.assign(
        usgs_code=usgs_code,
        time=pd.to_datetime(df.time),
    )
    if truncate_seconds:
        # Truncate seconds from timestamps: https://stackoverflow.com/a/28783971/592289
        # WARNING: This can potentially lead to duplicates!
        df = df.assign(time=df.time.dt.floor("min"))
        if df.time.duplicated().any():
            # There are duplicates. Keep the first datapoint per minute.
            msg = f"{usgs_code}: Duplicate timestamps have been detected after the truncation of seconds. Keeping the first datapoint per minute"
            warnings.warn(msg)
            df = df.iloc[df.time.drop_duplicates().index].reset_index(drop=True)
    return df


def get_usgs_station_data(
    usgs_code: str,
    endtime: Union[str, datetime.date] = datetime.date.today(),
    period: float = USGS_MAX_DAYS_PER_REQUEST,
    truncate_seconds: bool = True,
    rate_limit: Optional[RateLimit] = None,
) -> pd.DataFrame:
    """Retrieve the TimeSeries of a single IOC station."""

    # TODO: Code can be a list for USGS -- ALSO limit on list length,
    # maybe 20/25?
    if rate_limit:
        while rate_limit.reached(identifier="IOC"):
            wait()

    endtime = pd.to_datetime(endtime).date().isoformat()
    url = IOC_BASE_URL.format(usgs_code=usgs_code, endtime=endtime, period=period)
    logger.info("%s: Retrieving data from: %s", usgs_code, url)
    try:
        df = pd.read_html(url, header=0)[0]
    except ValueError as exc:
        if str(exc) == "No tables found":
            logger.info("%s: No data", usgs_code)
        else:
            logger.exception("%s: Something went wrong", usgs_code)
        raise
    df = normalize_usgs_station_data(usgs_code=usgs_code, df=df, truncate_seconds=truncate_seconds)
    return df


def get_usgs_data(
    usgs_metadata: pd.DataFrame,
    endtime: Union[str, datetime.date] = datetime.date.today(),
    period: float = 1,  # one day
    truncate_seconds: bool = True,
    rate_limit: RateLimit = RateLimit(),
    disable_progress_bar: bool = False,
) -> xr.Dataset:
    """
    Return the data of the stations specified in ``usgs_metadata`` as an ``xr.Dataset``.

    ``truncate_seconds`` needs some explaining. IOC has more than 1000 stations.
    When you retrieve data from all (or at least most of) these stations, you
    end up with thousands of timestamps that only contain a single datapoint.
    This means that the returned ``xr.Dataset`` will contain a huge number of ``NaN``
    which means that you will need a huge amount of RAM.

    In order to reduce the amount of the required RAM we reduce the number of timestamps
    by truncating the seconds. This is how this works:

        2014-01-03 14:53:02 -> 2014-01-03 14:53:00
        2014-01-03 14:53:32 -> 2014-01-03 14:53:00
        2014-01-03 14:53:48 -> 2014-01-03 14:53:00
        2014-01-03 14:54:09 -> 2014-01-03 14:54:00
        2014-01-03 14:54:48 -> 2014-01-03 14:54:00

    Nevertheless this approach has a downside. If a station returns multiple datapoints
    within the same minute, then we end up with duplicate timestamps. When this happens
    we only keep the first datapoint and drop the subsequent ones. So potentially you
    may not retrieve all of the available data.

    If you don't want this behavior, set ``truncate_seconds`` to ``False`` and you
    will retrieve the full data.

    :param usgs_metadata: A ``pd.DataFrame`` returned by ``get_usgs_stations``.
    :param endtime: The date of the "end" of the data.
    :param period: The number of days to be requested. IOC does not support values greater than 30
    :param truncate_seconds: If ``True`` then timestamps are truncated to minutes (seconds are dropped)
    :param rate_limit: The default rate limit is 5 requests/second.
    :param disable_progress_bar: If ``True`` then the progress bar is not displayed.

    """
    if period > USGS_MAX_DAYS_PER_REQUEST:
        msg = (
            f"Unsupported period. Please choose a period smaller than {USGS_MAX_DAYS_PER_REQUEST}: {period}"
        )
        raise ValueError(msg)

    func_kwargs = []
    for usgs_code in usgs_metadata.usgs_code:
        func_kwargs.append(
            dict(
                period=period,
                endtime=endtime,
                usgs_code=usgs_code,
                rate_limit=rate_limit,
                truncate_seconds=truncate_seconds,
            ),
        )

    results = multithread(
        func=get_usgs_station_data,
        func_kwargs=func_kwargs,
        n_workers=5,
        print_exceptions=False,
        disable_progress_bar=disable_progress_bar,
    )

    datasets = []
    for result in results:
        if result.result is not None:
            df = result.result
            meta = usgs_metadata[usgs_metadata.usgs_code == result.kwargs["usgs_code"]]  # type: ignore[index]
            ds = df.set_index(["usgs_code", "time"]).to_xarray()
            ds["lon"] = ("usgs_code", meta.lon)
            ds["lat"] = ("usgs_code", meta.lat)
            ds["country"] = ("usgs_code", meta.country)
            ds["location"] = ("usgs_code", meta.location)
            datasets.append(ds)

    # in order to keep memory consumption low, let's group the datasets
    # and merge them in batches
    while len(datasets) > 5:
        datasets = merge_datasets(datasets)
    # Do the final merging
    ds = xr.merge(datasets)
    return ds
