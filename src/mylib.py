import h5py
import pandas as pd
import xarray as xr


def extract_meta(filename):
    """Extract variables embeded in 'meta'"""
    with h5py.File(filename, "r") as h5f:
        for vname in h5f["meta"].dtype.names:
            yield xr.DataArray(
                h5f["meta"][vname][:],
                name=vname,
                dims=("location",),
                attrs=dict(
                    description="Extracted from meta variable",
                ),
            )


def fix_variable(da):
    attrs = da.attrs
    encoding = da.encoding
    encoding = {}
    if "fill_value" in da.attrs:
        print(f"Fixing fill value for {da.name}, {da.attrs['fill_value']}")
        da = da.where(da != da.attrs["fill_value"])
        attrs.pop("fill_value")
        # encoding["_FillValue"] = da.attrs.pop("fill_value")

    if "scale_factor" in da.attrs:
        if "adder" in da.attrs:
            da.attrs["offset"] = da.attrs["adder"]
            da = da + da.attrs["offset"]

        else:
            print(f"Fixing scaling factor for {da.name}, {da.attrs['scale_factor']}")
            da = da / da.attrs["scale_factor"]
            # da.encoding["scale_factor"] = 1 / attrs.pop("scale_factor")

        attrs.pop("scale_factor")

    da.attrs = attrs
    # da.encoding = encoding
    return da

standard_attributes = {
    "temperature": {
        "standard_name": "air_temperature",
        "units": "K",
    },
    "windspeed": {
        "standard_name": "wind_speed",
    },
    "winddirection": {
        "standard_name": "wind_to_direction",
    },
    "pressure": {
        "standard_name": "air_pressure",
    },
    "relativehumidity": {
        "standard_name": "relative_humidity",
    },
}

def add_standard_name(ds):
    for v in ds:
        if attrs:=v.split("_")[0] in standard_attributes:
            for k, v in standard_attributes[attrs].items():
                ds[v].attrs[k] = v

