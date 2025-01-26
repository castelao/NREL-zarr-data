
import time
import xarray as xr


ds = xr.open_zarr("/kfs3/scratch/gpimenta/zarr/WTK.zarr")
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

log = [mean_random_location(da, N=1000) for i in range(10)]




