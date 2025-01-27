import dask
from dask.distributed import LocalCluster, Client
import h5py
import netCDF4
from numcodecs import Blosc
import pandas as pd
import xarray as xr
import zarr.codecs

from mylib import *


def dev(h5filename, output_path, demo=False):
    """Big mess. Code in development"""

    ds = xr.open_mfdataset(h5filename, mask_and_scale=False, engine="netcdf4")
    ds = fix_time(ds)

    ds["lat"] = ds.coordinates.isel(phony_dim_1=0)
    ds["lon"] = ds.coordinates.isel(phony_dim_1=1)
    ds = ds.drop_vars(["coordinates"])
    ds = ds.rename_dims({"phony_dim_0": "location"})
    ds = ds.set_coords(["lat", "lon"])

    for v in extract_meta(h5filename):
        ds[v.name] = v

    for v in ds:
        ds[v] = fix_variable(ds[v])

    ds.temperature_10m.attrs["standard_name"] = "air_temperature"
    ds.windspeed_10m.attrs["standard_name"] = "wind_speed"
    ds.winddirection_10m.attrs["standard_name"] = "wind_to_direction"

    encoding_per_type = {
        "inversemoninobukhovlength": {
            "dtype": "i2",
            "scale_factor": 0.01,
        },
        "precipitationrate": {
            "dtype": "u2",
            "scale_factor": 0.1,
        },
        "pressure": {
            "dtype": "u2",
            "scale_factor": 0.1,
        },
        "relativehumidity": {
            "dtype": "u2",
            "scale_factor": 0.01,
        },
        "temperature": {
            "dtype": "i2",
            "scale_factor": 0.01,
            # "add_offset": -100,
        },
        "windspeed": {
            "dtype": "u2",
            "scale_factor": 0.01,
        },
        "winddirection": {
            "dtype": "u2",
            "scale_factor": 0.01,
        },
    }

    varnames = [v for v in ds if v.split("_")[0] in encoding_per_type]
    encoding = {}

    # from numcodecs import Blosc
    # compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.shuffle)

    for v in varnames:
        encoding[v] = encoding_per_type[v.split("_")[0]]
        encoding[v]["_FillValue"] = netCDF4.default_fillvals[encoding[v]["dtype"]]
        encoding[v]["compressor"] = compressor

    # subset for development
    # ds = ds[[v for v in ds if v.startswith("temperature") or v.startswith("pressure")]]
    # ds = ds.isel(point=range(0, ds.point.size, 100_000))
    if demo:
        ds = ds.isel(location=range(10_000))
    # 1_000 x 1_000 ~= 1MB
    # ds.isel(point=range(0, ds.point.size, 1_000)
    # ds[varnames].isel(point=range(100_000))
    ds.chunk({"time": -1, "location": 2_000}).to_zarr(
        output_path,
        zarr_format=2,
        encoding=encoding,
    )


if __name__ == "__main__":
    from pathlib import Path

    cluster = LocalCluster(
        n_workers=2,
        memory_limit=0.15,
        processes=True,
        threads_per_worker=4,
        # security=True,
    )
    ## client = Client(cluster.scheduler.address)
    client = cluster.get_client()

    h5filename = "/datasets/WIND/conus/v1.0.0/wtk_conus_2013.h5"

    zarr_path = Path("/scratch/gpimenta/zarr")

    dev(h5filename, output_path=zarr_path / "WTK_dev.zarr", demo=True)
