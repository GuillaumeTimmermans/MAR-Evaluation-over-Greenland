# This code checks the idw smb value in python; "official computation" is made with an old "obscur" fortran program If you want to be 100 % precise you should use the haversine distance


import numpy as np
from netCDF4 import Dataset
#from sklearn.neighbors import KDTree

nb_yr = float(len(range(1979,2025)))

THRESHOLD_MSK = 50
epsilon=1e-0



import numpy as np
from sklearn.neighbors import KDTree

# --- Conversion lat/lon -> xyz ---
def latlon_to_xyz(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack((x, y, z))

# --- Trouver k plus proches voisins ---
def knn_latlon(A_lats, A_lons, B_lats, B_lons, k=4):
    # Convert A and B to 3D coords
    A_xyz = latlon_to_xyz(A_lats, A_lons)
    B_xyz = latlon_to_xyz(B_lats, B_lons)

    # Build KDTree on dataset B
    tree = KDTree(B_xyz, metric='euclidean')

    # Query k nearest neighbors of A in B
    dist3d, idx = tree.query(A_xyz, k=k)

    # Convert chord distances to great-circle distances (km)
    R = 6371.0
    dist_km = 2 * R * np.arcsin(np.clip(dist3d / 2, 0, 1))

    return idx, dist_km




def smb_computation_from_annual_array(smb_anual, msk, lon, lat,area):
    # data_nc = Dataset(file)
    # try :
    #     smb, msk, lon, lat, area = [np.array(data_nc[i]) for i in ["MBTO", "MSK","LON","LAT", "AREA"]]
    # except( KeyError, IndexError) as e:
    #     smb, msk, lon, lat, area = [np.array(data_nc[i]) for i in ["MBto", "MSK","LON","LAT", "AREA"]]
    # smb = smb [:,0,:,:]
    m1  = (~ ((lat >= 75 ) & (lon <=-75))).astype(int)
    m1 *= (~ ((lat >=79.5) & (lon <=-67))).astype(int)
    m1 *= (~ ((lat >=81.2) & (lon <=-63))).astype(int)
    m1 *= (msk > 50).astype(int)
    m1 = m1.astype(float)
    m1*= msk/100.
    km3 = area / (1000*1000) 
    #plt.imshow(m1)
    #plt.show()
    return np.sum(smb_anual*m1*km3)
    # return np.sum(np.sum(smb,axis=0)*m1*km3)





for g, resol in enumerate(["5","10","15","20","30"]):
    with Dataset(f"{resol}-grid.nc") as ds:
        lon30,lat30,msk30,sh30 = [np.array(ds.variables[iii]) for iii in ["LON","LAT","MSK","SH"]]
    with Dataset("5-grid.nc") as ds:
        lon5,lat5,msk5,sh5,area5 = [np.array(ds.variables[iii]) for iii in ["LON","LAT","MSK","SH","AREA"]]
    msk30flat = np.ravel(msk30)
    lat_array30 = np.ravel(lat30)[msk30flat > THRESHOLD_MSK]
    lon_array30 = np.ravel(lon30)[msk30flat > THRESHOLD_MSK]
    msk5flat = np.ravel(msk5)
    lat_array5 = np.ravel(lat5)[msk5flat > THRESHOLD_MSK]
    lon_array5 = np.ravel(lon5)[msk5flat > THRESHOLD_MSK]
    # break
    #Pour chaque point de 5 (A), trouver les 4 plus proches voisins dans 30 (B) et leurs distances
    A_lats = lat_array5
    A_lons = lon_array5
    B_lats = lat_array30
    B_lons = lon_array30
    indices, distances = knn_latlon(
        A_lats, A_lons,     # dataset A
        B_lats, B_lons,     # dataset B
        k=4
    )
    # indices 1D
    mask = msk5flat > THRESHOLD_MSK
    idx_1d = np.where(mask)[0]
    
    # indices 2D
    rows, cols = np.unravel_index(idx_1d, msk5.shape)
    # distances_2d = np.ones_like(msk5)*-999
    # distances_2d[rows,cols] = distances[:,0]

    smb30 = np.load(f"./mean-annual-smb-{resol}.npy")
    
    smb30 = np.ravel(smb30)[msk30flat > THRESHOLD_MSK]
    
    smb_2d = np.ones_like(msk5)*-1e36
    for gims in range(4):
        if gims==0:
            smb_2d[rows,cols] = smb30[indices[:,gims]] * 1./(distances[:,gims]+epsilon)**2  # distances[:,0]
        else:
            smb_2d[rows,cols] += smb30[indices[:,gims]]* 1./(distances[:,gims]+epsilon)**2
    total_dist = np.sum(1/(distances+epsilon)**2,axis=1)
    smb_2d[rows,cols]/=total_dist
    smb_2d[smb_2d < -1e20]=0
    # print(np.sum(smb_2d))
    integ = smb_computation_from_annual_array(smb_2d, msk5, lon5, lat5,area5)
    print(resol,':',integ)
    # for u, year in enumerate(range(1979,2025)):
    #     print(resol, year)
