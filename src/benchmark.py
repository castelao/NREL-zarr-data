
import numpy as np
from rex import Resource
from scipy.interpolate import interp1d
import time
import xarray as xr



da = ds["windspeed_100m"]


def mean_random_location(da, N=1000):
    idx = np.random.permutation(da.point.size)[:N]
    t0 = time.time()
    m = da.isel(point=idx).mean().compute()
    t_tot = time.time() - t0
    return t_tot


def mean_sequential_location(da, N=1000):
    start = np.random.randint(0, da.point.size - N)
    end = start + N
    t0 = time.time()
    m = da.isel(point=slice(start, end)).mean().compute()
    t_tot = time.time() - t0
    return t_tot

PC_POWER = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 16, 49, 81, 122, 163, 211,
 259, 319, 378, 441, 504, 574, 643, 726, 808, 896, 984, 1072, 1159, 1236, 1312,
1369, 1426, 1473, 1519, 1545, 1571, 1583, 1594, 1602, 1609, 1614, 1619, 1620,
            1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620,
            1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620,
            1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620,
            1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620,
            1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 1620, 0]
PC_WS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25,
         3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75,
         7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25,
         10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25,
         13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25,
         16.5, 16.75, 17, 17.25, 17.5, 17.75, 18, 18.25, 18.5, 18.75, 19, 19.25,
         19.5, 19.75, 20, 20.25, 20.5, 20.75, 21, 21.25, 21.5, 21.75, 22, 22.25,
         22.5, 22.75, 23, 23.25, 23.5, 23.75, 24, 24.25, 24.5, 24.75, 25, 25.25]

f = interp1d(PC_WS, PC_POWER, kind="linear", fill_value="extrapolate")

def myfunc(x):
    return np.interp(x, PC_WS, PC_POWER).sum()


def interp_random_location_zarr(dataset_path, N=1_000):
    ds = xr.open_zarr(dataset_path)
    da = ds["windspeed_100m"]
    idx = np.random.permutation(2488136)[:N]
    idx = np.sort(idx)
    t0 = time.time()
    result = xr.apply_ufunc(
        f,
        da.isel(point=idx),
        input_core_dims=(da.dims,),
        output_core_dims=(da.dims,),
        dask="allowed",
    ).sum(dim="time").compute()
    t_tot = time.time() - t0
    return {"runtime": t_tot, "result": result}

def interp_random_location_rex(dataset_path, N=100):
    idx = np.random.permutation(2488136)[:N]
    idx = np.sort(idx)
    t0 = time.time()
    with Resource(dataset_path) as res:
        data = res["windspeed_100m", :, idx]
    result = f(data).sum(axis=0)
    t_tot = time.time() - t0
    return {"runtime": t_tot, "result": result}




log_zarr = [interp_random_location_zarr(zarr_path, N=10_000) for i in range(3)]
log_rex = [interp_random_location_rex(h5f_path, N=250_000) for i in range(3)]

t_zarr = [t["runtime"] for t in log_zarr]
print(t_zarr)
t_rex = [t["runtime"] for t in log_rex]
print(t_rex)

np.mean(t_zarr), np.std(t_zarr)
np.mean(t_rex), np.std(t_rex)



