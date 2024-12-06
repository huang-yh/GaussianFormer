import os
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(2560, 1440))
        display.start()
        offscreen = True
    except:
        print("Failed to start virtual display.")

try:
    from mayavi import mlab
    import mayavi
    mlab.options.offscreen = offscreen
    print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
except:
    print("No Mayavi installation found.")

import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid
import os

from model.utils.safe_ops import safe_sigmoid


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def save_occ(
        save_dir, 
        gaussian, 
        name,
        sem=False,
        cap=2,
        dataset='nusc'
    ):
    if dataset == 'nusc':
        # voxel_size = [0.4] * 3
        # vox_origin = [-40.0, -40.0, -1.0]
        # vmin, vmax = 0, 16
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
        vmin, vmax = 0, 16
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 19
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 18

    voxels = gaussian[0].cpu().to(torch.int)
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1 
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        voxels.shape, voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    if not sem:
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
        ]
    else:
        if dataset == 'nusc':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)
            ]
        elif dataset == 'kitti360':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 19)
            ]
        else:
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
            ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=vmin,
            vmax=vmax, # 16
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    if sem:
        if dataset == 'nusc':
            colors = np.array(
                [
                    [  0,   0,   0, 255],       # others
                    [255, 120,  50, 255],       # barrier              orange
                    [255, 192, 203, 255],       # bicycle              pink
                    [255, 255,   0, 255],       # bus                  yellow
                    [  0, 150, 245, 255],       # car                  blue
                    [  0, 255, 255, 255],       # construction_vehicle cyan
                    [255, 127,   0, 255],       # motorcycle           dark orange
                    [255,   0,   0, 255],       # pedestrian           red
                    [255, 240, 150, 255],       # traffic_cone         light yellow
                    [135,  60,   0, 255],       # trailer              brown
                    [160,  32, 240, 255],       # truck                purple                
                    [255,   0, 255, 255],       # driveable_surface    dark pink
                    # [175,   0,  75, 255],       # other_flat           dark red
                    [139, 137, 137, 255],
                    [ 75,   0,  75, 255],       # sidewalk             dard purple
                    [150, 240,  80, 255],       # terrain              light green          
                    [230, 230, 250, 255],       # manmade              white
                    [  0, 175,   0, 255],       # vegetation           green
                    # [  0, 255, 127, 255],       # ego car              dark cyan
                    # [255,  99,  71, 255],       # ego car
                    # [  0, 191, 255, 255]        # ego car
                ]
            ).astype(np.uint8)
        elif dataset == 'kitti360':
            colors = (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8)
        else:
            colors = (get_kitti_colormap()[1:, :] * 255).astype(np.uint8)

        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    scene = figure.scene
    scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.azimuth(-5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(-5)
    scene.render()
    scene.camera.position = [-138.7379881436844, -0.008333206176756428, 99.5084646673331]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [104.37185230017721, 252.84608651497263]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.position = [-114.65804807470022, -0.008333206176756668, 82.48137575398867]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [75.17498702830105, 222.91192666552377]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.position = [-94.75727115818437, -0.008333206176756867, 68.40940144543957]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [51.04534630774225, 198.1729515833347]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.6463156430702276, -6.454925414290924e-18, 0.7630701733934554]
    scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
    scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555]
    scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(-5)
    mlab.pitch(-8)
    mlab.move(up=15)
    scene.camera.orthogonalize_view_up()
    scene.render()

    # scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
    # scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
    # scene.camera.view_angle = 40.0
    # scene.camera.view_up = [0.0, 0.0, 1.0]
    # scene.camera.clipping_range = [0.01, 300.]
    # scene.camera.compute_view_plane_normal()
    # scene.render()

    filepath = os.path.join(save_dir, f'{name}.png')
    if offscreen:
        mlab.savefig(filepath)
    else:
        mlab.show()
    mlab.close()

def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.float32) / 255.
    return colors

def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False):

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

    means = gaussian.means[0].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy() # g
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.75)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk
        
        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask


    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    # number of ellipsoids 
    ellipNumber = means.shape[0]

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=46, azim=-180)

    # compute each and plot each ellipsoid iteratively
    border = np.array([
        [-50.0, -50.0, 0.0],
        [-50.0, 50.0, 0.0],
        [50.0, -50.0, 0.0],
        [50.0, 50.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in range(ellipNumber):
        
        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("equal")
    # plt.gca().set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_axis_off()    

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath)

    plt.cla()
    plt.clf()

def save_gaussian_topdown(save_dir, anchor_init, gaussian, name):
    init_means = safe_sigmoid(anchor_init[:, :2]) * 100 - 50
    means = [init_means] + [g.means[0, :, :2] for g in gaussian]

    plt.clf(); plt.cla()
    fig = plt.figure(figsize=(24., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 5),  # creates 2x2 grid of Axes
                    axes_pad=0.,  # pad between Axes in inch.
                    share_all=True
                    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    for ax, im in zip(grid, means):
        im = im.cpu()
        # Iterating over the grid returns the Axes.
        ax.scatter(im[:, 0], im[:, 1], s=0.1, marker='o')
    plt.savefig(os.path.join(save_dir, f"{name}.jpg"))
    plt.clf(); plt.cla()
