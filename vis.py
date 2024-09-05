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
