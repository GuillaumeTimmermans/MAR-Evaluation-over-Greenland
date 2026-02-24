from sklearn.neighbors import BallTree
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt

def build_interpol(lat_on_which_we_interpolate,
                   lon_on_which_we_interpolate,
                   lat_source,
                   lon_source,
                   N=4,
                   msk_source=None):

    # Flatten source grid
    lat_flat = lat_source.ravel()
    lon_flat = lon_source.ravel()

    if msk_source is not None:
        mask_flat = msk_source.ravel().astype(bool)

        # Keep only valid points
        lat_valid = lat_flat[mask_flat]
        lon_valid = lon_flat[mask_flat]

        # Store mapping to original indices
        valid_indices = np.where(mask_flat)[0]
    else:
        lat_valid = lat_flat
        lon_valid = lon_flat
        valid_indices = np.arange(lat_flat.size)

    # Build BallTree on valid points only
    bt = BallTree(
        np.deg2rad(np.column_stack((lat_valid, lon_valid))),
        metric='haversine'
    )

    # Target grid
    coords_rad = np.deg2rad(
        np.column_stack((lat_on_which_we_interpolate.ravel(),
                         lon_on_which_we_interpolate.ravel()))
    )

    distances, indices_valid = bt.query(coords_rad, k=N)

    # Convert back to original source indices
    indices = valid_indices[indices_valid]

    return distances, indices


def interpol(lat_on_which_we_interpolate, lon_on_which_we_interpolate, lat_source,lon_source,data_to_interpol, N = 4, p = 2, msk_source = None):
    """
    data_to_interpol doit avoir les dimensions suivantes : (time, Nx, Ny) ou (Nx, Ny)
    """
    distances,indices = build_interpol(lat_on_which_we_interpolate, lon_on_which_we_interpolate, lat_source,lon_source)
    time_flag = False
    if len(np.shape(data_to_interpol)) ==3:
        time_flag = True
    if time_flag:
        output = np.zeros((np.shape(data_to_interpol)[0], np.shape(lat_on_which_we_interpolate)[0],np.shape(lat_on_which_we_interpolate)[1]))
        for temps in range(np.shape(data_to_interpol)[0]):
            val = data_to_interpol[temps,:,:].ravel()

            M = distances.shape[0]
            weighted_sum = np.zeros(M)#np.zeros_like(data_to_interpol)
            #weighted_sum = np.zeros(M)
            if temps==0:
                weight_sum = np.zeros(M)

            for i in range(N):
                d = distances[:, i]
                w = (1 / d)**p
                vals = val[indices[:, i]]

                weighted_sum += vals * w
                if temps==0:
                    weight_sum += w

            interp = weighted_sum / weight_sum
            interp = interp.reshape(lat_on_which_we_interpolate.shape)
            output[temps,:,:]=interp.copy()
        return output
    else:
        val = data_to_interpol.ravel()

        M = distances.shape[0]

        weighted_sum = np.zeros(M)
        weight_sum = np.zeros(M)

        for i in range(N):
            d = distances[:, i]
            w = (1 / d)**p
            vals = val[indices[:, i]]

            weighted_sum += vals * w
            weight_sum += w

        interp = weighted_sum / weight_sum
        interp = interp.reshape(lat_on_which_we_interpolate.shape)
    return interp
