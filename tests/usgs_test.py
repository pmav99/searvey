import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import geometry

from searvey import usgs


def test_get_usgs_stations_exception():
    lon_min = -76
    lat_min = 39
    lon_max = -70
    lat_max = 43
    geom = geometry.box(lon_min, lat_min, lon_max, lat_max)

    with pytest.raises(Exception):
        usgs.get_usgs_stations(
            region=geom,
            lon_min=lon_min,
            lat_min=lat_min,
            lon_max=lon_max,
            lat_max=lat_max,
        )


@pytest.mark.vcr
def test_get_usgs_stations():
    stations = usgs.get_usgs_stations()
    assert isinstance(stations, pd.DataFrame)
    assert isinstance(stations, gpd.GeoDataFrame)
    assert len(stations) > 242000
    # check that the DataFrame has the right columns
    df_columns = {col for col in stations.columns}
    expected_columns = {col for col in usgs.USGS_STATIONS_COLUMN_NAMES}
    assert df_columns.issuperset(expected_columns)


@pytest.mark.vcr
@pytest.mark.parametrize(
    "truncate_seconds,no_records",
    [
        pytest.param(True, 1022, id="truncate_seconds=True"),
        pytest.param(False, 1022, id="truncate_seconds=False"),
    ],
)
def test_get_usgs_station_data(truncate_seconds, no_records):
    """Truncate_seconds=False returns more datapoints compared to `=True`"""
    df = usgs.get_usgs_station_data(
        usgs_code="301112085500201",
        endtime=datetime.date(2022, 10, 1),
        period=1,
        truncate_seconds=truncate_seconds,
    )
    assert len(df) == no_records


_USGS_METADATA_MINIMAL = pd.DataFrame.from_dict(
    {
        "agency_cd": {0: "USGS", 1: "USGS"},
        "alt_datum_cd": {0: "NAVD88", 1: "NAVD88"},
        "alt_va": {0: 0.0, 1: 0.0},
        "begin_date": {0: "2022-09-26", 1: "2022-09-27"},
        "dec_coord_datum_cd": {0: "NAD83", 1: "NAD83"},
        "dec_lat_va": {0: 27.97775, 1: 30.18655556},
        "dec_long_va": {0: -82.8322778, 1: -85.8339722},
        "end_date": {0: "2022-10-03", 1: "2022-10-04"},
        "parm_cd": {0: "62622", 1: "62622"},
        "site_no": {0: "275840082495601", 1: "301112085500201"},
        "station_nm": {
            0: "GULF OF MEXICO AT CLEARWATER BEACH, FL",
            1: "GULF OF MEXICO NEAR PANAMA CITY BEACH, FL",
        },
    }
)


@pytest.mark.parametrize(
    "truncate_seconds",
    [
        pytest.param(True, id="truncate_seconds=True"),
        pytest.param(False, id="truncate_seconds=False"),
    ],
)
def test_get_usgs_data(truncate_seconds):
    # in order to speed up the execution time of the test,
    # we don't retrieve the USGS metadata from the internet,
    # but we use a hardcoded dict instead
    usgs_metadata = _USGS_METADATA_MINIMAL
    ds = usgs.get_usgs_data(
        usgs_metadata=usgs_metadata,
        endtime="2022-09-29",
        period=2,
        truncate_seconds=truncate_seconds,
    )
    assert isinstance(ds, xr.Dataset)
    # TODO: Check if truncate_seconds does work
    assert len(ds.datetime) == 720
    assert len(ds.site_no) == len(usgs_metadata)
    # Check that usgs_code, lon, lat, country and location are not modified
    assert set(ds.site_no.values).issubset(usgs_metadata.site_no.values)
    assert set(ds.lon.values).issubset(usgs_metadata.dec_long_va.values)
    assert set(ds.lat.values).issubset(usgs_metadata.dec_lat_va.values)
    # assert set(ds.country.values).issubset(usgs_metadata.country.values)
    # assert set(ds.location.values).issubset(usgs_metadata.location.values)
    # Check that actual data has been retrieved
    assert ds.sel(site_no="275840082495601").value.mean() == pytest.approx(-0.17697674, rel=1e-3)
    assert ds.sel(site_no="301112085500201").value.mean() == pytest.approx(-0.0492796, rel=1e-3)
