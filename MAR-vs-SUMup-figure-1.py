# You will need to change file paths for this code to work





import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import griddata, interp1d

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import get_cmap

fast =  False







# fig = plt.figure(figsize=(14.5,8.5))  # Taille globale
fig = plt.figure(figsize=(12.,8.5))  # Taille globale
# Grille 2x2, mais la grande figure occupe les deux lignes à gauche
# gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1., 0.03, 1.2],  # large gauche / fine colonne pour colorbar / droite
    height_ratios=[1, 1],
    wspace=0.01, hspace=0.3
)
# --- Sous-figures ---
ax_left = fig.add_subplot(gs[:, 0])
ax = ax_left# grande à gauche
ax_cb   = fig.add_subplot(gs[:, 1])        # pour la colorbar
ax_top_right_true  = fig.add_subplot(gs[0, 2])        # en haut à droite
ax_bottom_right  = fig.add_subplot(gs[1, 2])        # en bas à droite
# # C

letters = ['(a)', '(b)', '(c)']

for axii, letter in zip([ax_left, ax_top_right_true, ax_bottom_right], letters):
    axii.text(0.94, 0.98, letter, transform=axii.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCATTER PLOTS (1) et (2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# points = ['X', "v", "^", "s", "P", "*", ".", "D", "p", "o"]
# colors = ['b','orange','g','r', 'c', 'm', 'k', 'y', 'crimson', 'peru','dodgerblue']
colors = [
    "purple",  # vert profond
    "green",  # jaune doré doux
    "black"#"#332288"   # violet indigo
]

# Formes distinctes, bien visibles même en petit
markers = ["o", "s", "D"]

# Pour les stations PROMICE–GC-NET
color_promice = "black"# "black"  # vert olive doux
marker_promice = "X"
label_promice = "PROMICE\n & GC-NET"

# Taille
taille_points_carte = 45
method_names = ["firn/ice core", "stake measurements", "estim. from m.b. profile", "snow pits", "snow pit / do18 dating", "snowfox", 'ice cores from XF']
min_axes, max_axes = -9. ,5.
# markers = ["o", "D"]#, 'X']
# colors = ["dodgerblue", "crimson"]#, 'green']
# color_promice = 'green'
# marker_promice = 'X'
# label_promice = 'PROMICE-GC NET'
# taille_points_carte = 45
 # --- Styles distincts pour les deux groupes ---
names_source = ["SUMup", "Ohmura & Reeh"] #"Bales et al."]
group_styles = {
    f"{names_source[0]}": {"color": colors[0], "marker": markers[0]},
    f"{names_source[1]}": {"color": colors[1], "marker": markers[1]},
}
for axindex, ax_top_right in enumerate([ax_bottom_right, ax_top_right_true]):
    for rrr, res in enumerate([5]):#,10,15,20,30]):
        if axindex==0:
            path = f"./data/mar-3.14.3/{int(res)}km"
        else:
            path = f"./no-interpol/data/mar-3.14.3-no-interpol/{int(res)}km"

        if (rrr != 0) & (rrr != 1) & (rrr != 2) & (rrr != 3) & (rrr != 4):  # A faire mais en fait je ne suis plus sur des chemins vers les sorties de xavier a ces resolutions, je pense que les sorties
            #de la derniere version du mar sont dispos...
            ax[iis[rrr],jjs[rrr]].text(0,0,'le fichier ICE.2019.01-08.m90.nc n est pas complet')
        else:
            nbroutliers = 0
            NBRLINES = 0
            tout1, tout2 = [],[]
            for ggg, i in enumerate([3,4,6,12,14,15,-999]):#enumerate([3,4,6,12,14,15,13,16]):#,"ice_cores_xavier"])
                stat, index, obs1,obs2 = [],[],[],[]
                fichier_dat2 = f"{path}/method-{int(i)}.dat2"
                if i < -900:
                    fichier_dat2 = f"{path}/ice_cores_xavier.dat2"
                with open(fichier_dat2) as file:
                        lignes = file.readlines()
                        for g, li in enumerate(lignes):
                            NBRLINES+=1
                            # print(l)
                            # print(float(l[3]))
                            l = li.split()
                            try:
                                if str(l[0])[0]=='@':
                                    print(l)
                                # stat.append(str(l[0]))
                                # index.append(int(l[0]))
                                obs1.append(float(l[1]))
                                obs2.append(float(l[2]))
                            except ValueError as e:
                                print(e)
                                nbroutliers+=1
                                print("error")
                                #print("outlier:",li)
                                mini = len(obs2)
                                #for gg, x in enumerate([stat, index, obs1, obs2]):
                                for gg, x in enumerate([obs1, obs2]):
                                    if len(x) > mini:
                                        x.pop()
                            # if np.abs(float(l[2]) - float(l[3])) > 2:
                            #     print(g)
                            #     print(li)
                tout1 += obs1
                tout2 += obs2
                if i == -999:
                    ax_top_right.scatter(obs1,obs2, color = colors[1], marker=markers[1], s = 15)#, edgecolor = 'k')#, label = method_names[ggg])
                else:
                    ax_top_right.scatter(obs1,obs2, color = colors[0], marker=markers[0],s=15)#, edgecolor = 'k')#, label = method_names[ggg])

            ax_top_right.set_title("Elevation correction")
            if axindex ==1:
                ax_top_right.set_title("Raw")
            ax_top_right.set_aspect('equal')
            ax_top_right.set_xlim(min_axes,max_axes)
            ax_top_right.set_ylim(min_axes,max_axes)
            ax_top_right.set_xlabel("Measured SMB [m]")
            ax_top_right.set_ylabel("MAR SMB [m]")
            ax_top_right.plot(np.array([min_axes-0.1,max_axes+0.1]), np.array([min_axes-0.1,max_axes+0.1]), ls = '-.', color = 'grey', alpha = 0.5)
            # if rrr==0:
            #     ax[iis[rrr],jjs[rrr]].legend(loc = 'upper left')
            obs = np.array(tout1)
            mod = np.array(tout2)
            N = len(obs)
            # Biais (moyenne de la différence mod - obs)
            bias = np.mean(mod - obs)
            # RMSE
            rmse = np.sqrt(mean_squared_error(obs, mod))
            # R²
            r2 = r2_score(obs, mod)
            # Régression linéaire (y = slope * x + intercept)
            reg = LinearRegression().fit(obs.reshape(-1, 1), mod)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            corr, _ = pearsonr(obs, mod)
            ax_top_right.plot(np.array([min_axes-0.1,max_axes+0.1]), intercept + slope * np.array([min_axes-0.1,max_axes+0.1]), color = 'red', alpha = 0.5)
            # ax_top_right.text(0.,-6.,f"N = {N}\nBias = {bias:.3f}m\nRMSE = {rmse:.3f}m\nR² = {r2:.3f}\nSlope = {slope:.3f}\nIntercept = {intercept:.3f}m")
# --------------------------------------------------------------------
            textstr = '\n'.join((
                f'Bias = {bias:.2f} m',
                f'RMSE = {rmse:.2f} m',
                f'Corr = {corr:.2f}',# <-------------------------------------------
                f'Slope = {slope:.2f}',
                f'Int. = {intercept:.2f}m',
                f'N = {N}'
            ))
            ax_top_right.text(0.02, 0.98, textstr, transform=ax_top_right.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            # Text with metrics
            # textstr = '\n'.join((
            #     f'Bias = {bias:.2f} {units[uuu]}',
            #     f'RMSE = {rmse:.2f} {units[uuu]}',
            #     f'Corr = {corr:.2f}',
            #     f'N = {len(np.array(temperatures_mars))}'
            # ))
            # ax[iis[uuu],jjs[uuu]].text(0.05, 0.95, textstr, transform=ax[iis[uuu],jjs[uuu]].transAxes,
            #         fontsize=10, verticalalignment='top',
            #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
# --------------------------------------------------------------------
        # ax_top_right.axis("off")   # on cache les axes inutilisés
    handles, labels = ax_top_right.get_legend_handles_labels()
    # if axindex == 0:
    #     ax_top_right.legend(handles, labels, loc="center", fontsize=10, frameon=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CARTE             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ax.legend(handles, labels, loc="center", fontsize=10, frameon=False)
ax.legend(
    handles,
    labels,
    loc="lower right",      # position en bas à droite
    fontsize=10,
    frameon=True,           # affiche un encadré
    framealpha=0.8,         # transparence (0 = transparent, 1 = opaque)
    facecolor="white",      # fond blanc (ou autre couleur douce)
    edgecolor="gray"        # bord fin et neutre
)

# ds = Dataset("./grid_info.nc")
ds = Dataset("./5kgrid.nc")

# lonmar,latmar,mskmar, xmar, ymar= [np.array(ds.variables[xxx]) for xxx in ["LON", "LAT", "MSK", "X", "Y"]]
lonmar,latmar,mskmar, xmar, ymar= [np.array(ds.variables[xxx]) for xxx in ["LON", "LAT", "MSK", "X14_311", "Y21_560"]]
msk = mskmar
xgrid, ygrid = np.meshgrid(xmar, ymar)
# plt.pcolormesh(xgrid, ygrid, mskmar)
ax.contour(xgrid, ygrid, mskmar,[0.1],colors='black')
smb_5km = np.load("./somme5km.npy")

# ax.pcolormesh(xgrid, ygrid, smb_5km)
bounds = [100,250,500,1000,1500,2000,3000,4000,5000,6000]
bounds = (-np.flip(np.array(bounds))).tolist() + bounds
colors = plt.get_cmap("bwr_r", len(bounds)-1)(range(len(bounds)-1))
colors = np.array(colors)
# Remplacer la tranche correspondant à [-100,100] par blanc
i0 = bounds.index(-100)
colors[i0] = (1,1,1,1)  # RGBA blanc
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)
m = ax.pcolormesh(xgrid,ygrid,np.ma.masked_where(msk<50.,msk*0.01*smb_5km/float(len(range(1979,2025)))), cmap = cmap,norm = norm)
cb=fig.colorbar(m,cax=ax_cb, label = "SMB [mmwe / year]")
cb.set_ticks(bounds)
ax.set_aspect('equal')
lon_flat = lonmar.flatten()
lat_flat = latmar.flatten()
x_flat = xgrid.flatten()
y_flat = ygrid.flatten()
# On interpole pour trouver les coordonnées x/y correspondant à chaque (lon, lat)
points = np.column_stack((lon_flat, lat_flat))
values_x = x_flat
values_y = y_flat

# Contours de lonmar en pointillés avec étiquettes
cont_lon = ax.contour(xgrid, ygrid, np.ma.masked_where(msk > 50,lonmar), colors='black', levels = [-55,-45,-35,-25,-15,-5,5,15,25,35,45,55], linestyles = 'dashdot',alpha = 0.5)
ax.clabel(cont_lon, inline=True, fontsize=8)
# Contours de latmar en noir plein avec étiquettes
cont_lat = ax.contour(xgrid, ygrid, np.ma.masked_where(msk > 50,latmar), colors='black',levels = [60,65,70,75,80,85],linestyles = 'dashdot', alpha = 0.5)
ax.clabel(cont_lat, inline=True, fontsize=8)




lon_label = [-55,-50,-45,-40,-35]#,0,30,60]
lon_line = lonmar[0, :]
x_line = xgrid[0, :]  # x projeté
interp_func = interp1d(lon_line, x_line, bounds_error=False, fill_value='extrapolate')
x_ticks = interp_func(lon_label)
x_tick_coords = x_ticks#[-400, -0, 400]
ax.set_xticks(x_ticks)
ax.set_xticklabels(lon_label)

lat_label = [60,65,70,75,80]#[-55,-50,-45,-40,-35,-30]#,0,30,60]
lat_line = latmar[:, 0]
y_line = ygrid[:, 0]  # x projeté
interp_func = interp1d(lat_line, y_line, bounds_error=False, fill_value='extrapolate')
y_ticks = interp_func(lat_label)
y_tick_coords = y_ticks#[-400, -0, 400]
ax.set_yticks(y_ticks)
ax.set_yticklabels(lat_label)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

if fast is False:
    # --- Fichiers des deux groupes ---
    ids = [3, 4, 6, 12, 14, 15, -999]
    group_points = {f"{names_source[0]}": {"lons": [], "lats": []},
                    f"{names_source[1]}": {"lons": [], "lats": []}}

    # --- Lecture des fichiers ---
    for i in ids:
        if i == -999:
            group = f"{names_source[1]}"
            file = "./data/mar-3.14.3/5km/ice_cores_xavier.dat1"
        else:
            group = f"{names_source[0]}"
            file = f"./data/mar-3.14.3/5km/method-{i}.dat1"

        seen_points = set()
        with open(file) as fichier:
            for line in fichier:
                ll = line.strip().split(',')
                lon, lat = float(ll[2]), float(ll[1])
                if (lon, lat) not in seen_points:
                    group_points[group]["lons"].append(lon)
                    group_points[group]["lats"].append(lat)
                seen_points.add((lon, lat))

    # --- Interpolation et tracé sur la carte ---
    for group, style in group_styles.items():
        lon_pts = np.array(group_points[group]["lons"])
        lat_pts = np.array(group_points[group]["lats"])
        x_pts = griddata(points, values_x, (lon_pts, lat_pts), method='linear')
        y_pts = griddata(points, values_y, (lon_pts, lat_pts), method='linear')
        ax.scatter(x_pts, y_pts,
                   color=style["color"],
                   marker=style["marker"],
                   s=taille_points_carte,
                   # edgecolors='k',
                   label=group, zorder=10)

    # --- Légende propre ---
    # ax.legend(
    #     loc="lower right",
    #     fontsize=10,
    #     frameon=True,
    #     framealpha=0.85,
    #     facecolor="white",
    #     edgecolor="gray"
    # )

### Enfin, il faut ajouter les stations PROMICE-GC Net
lons , lats = [], []
with open("./file-stations-promice.dat") as file:
    lignes = file.readlines()
    for g, li in enumerate(lignes):
        # print(li)
        if g > 1:
            # print(li)
            lii = li.split()
            # print(li.split())
            if len(lii)> 1:
                # stations.append(lii[0])
                lons.append(-float(lii[2]))
                lats.append(float(lii[1]))

lon_pts = np.array(lons)
lat_pts = np.array(lats)
x_pts = griddata(points, values_x, (lon_pts, lat_pts), method='linear')
y_pts = griddata(points, values_y, (lon_pts, lat_pts), method='linear')
ax.scatter(x_pts, y_pts,
           color=color_promice,
           marker=marker_promice,
           s=taille_points_carte,
           # edgecolors='k',
           label=label_promice, zorder=10)
ax.legend(
    loc="lower right",
    fontsize=10,
    frameon=True,
    framealpha=0.85,
    facecolor="white",
    edgecolor="gray"
)

fig.savefig("FIGURE1.png", dpi = 550, bbox_inches='tight')
plt.show()
