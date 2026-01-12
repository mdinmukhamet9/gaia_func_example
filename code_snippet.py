import open3d as o3d
import numpy as np
import time
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def simulate_scan(mesh: o3d.geometry.TriangleMesh,
                    eye_pos = [0, -5000, 2500],        
                    lookat = [0, 0, 100],
                    up = [0, 0, 1],  
                    fov_deg = 90,
                    width = 1280,
                    height = 1024,
                    scene = None) -> o3d.geometry.PointCloud:
    
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    if len(tensor_mesh.vertex["positions"]) == 0 or len(tensor_mesh.triangle["indices"]) == 0:
        raise ValueError("Mesh is empty or has no triangles.")

    if scene is None:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        eye = eye_pos,
        center = lookat,
        up = up,
        width_px = width,
        height_px = height,
        fov_deg = fov_deg
    )
    ans = scene.cast_rays(rays)
    t_hit = ans['t_hit']
    valid = t_hit.isfinite()

    origins = rays[valid][:,:3]
    directions = rays[valid][:,3:]
    distances = t_hit[valid].reshape((-1,1))
    hit_points = origins + directions*distances
    
    scan_pcd = o3d.t.geometry.PointCloud(hit_points)
    return scan_pcd.to_legacy()


def angled_dual_flyover_scan(mesh,
                            z_height=800,
                            lookat_z=100,
                            lookat_offset_x=0,
                            width=640,
                            height=480,
                            fov_deg=55,
                            step=300,
                            x_offset=100,
                            y_start = -6000,
                            y_end = 1500) -> o3d.geometry.PointCloud:
    t0= time.time()
    vis_geoms = []

    def make_flyover(x_eye, x_look_offset, y_start, y_end, label):
        scans = []
        for y in range(y_start, y_end, step if y_start < y_end else -step):
            eye = [x_eye, y, z_height]
            lookat = [x_look_offset, y, lookat_z]

            # Cam vis
            cam = o3d.geometry.TriangleMesh.create_sphere(radius=15)
            cam.translate(eye)
            cam.paint_uniform_color([1, 0.5, 0.5])
            vis_geoms.append(cam)

            # Scan vis
            line = o3d.geometry.LineSet()
            points = np.array([eye, lookat])
            lines = np.array([[0, 1]])
            colors = np.array([[0, 1, 0]])
            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector(lines)
            line.colors = o3d.utility.Vector3dVector(colors)
            vis_geoms.append(line)

            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(tensor_mesh)

            scan = simulate_scan(mesh,
                                eye_pos=eye,
                                lookat=lookat,
                                up=[0, 0, 1],
                                width=width,
                                height=height,
                                fov_deg=fov_deg,
                                scene=scene)
            scans.append(scan)
        return scans

    # left and right cameras
    left_scans = make_flyover(x_eye=-x_offset,
                            x_look_offset=lookat_offset_x,
                            y_start= y_start,
                            y_end=y_end,
                            label="Left-angled")

    right_scans = make_flyover(x_eye=+x_offset,
                            x_look_offset=-lookat_offset_x,
                            y_start=y_start,
                            y_end=y_end,
                            label="Right-angled")

    # add scans
    all_pcd = left_scans[0]
    for p in left_scans[1:] + right_scans:
        all_pcd += p
    all_pcd = all_pcd.voxel_down_sample(2.0)

    # to see how cameras are placed and how they got the scan
    duration = time.time()-t0
    vis_geoms.append(all_pcd.paint_uniform_color([0.3, 0.3, 0.7]))
    print(f"Time taken to create a Point Cloud: {duration}")
    o3d.visualization.draw_geometries(vis_geoms, window_name="Angled Multi-View Beam Scan")

    return all_pcd