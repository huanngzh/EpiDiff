import argparse
import os

import imageio
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm


# lookAt function implementation
def look_at(eye, target, up):
    normalize = lambda v: v / np.linalg.norm(v)
    lookat = normalize(target - eye)
    right = normalize(np.cross(lookat, up))
    up = np.cross(right, lookat)

    # Construct a rotation matrix from the right, new_up, and forward vectors
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack((right, up, -lookat)).T
    c2w[:3, 3] = eye

    return c2w


def look_at_ndarray(eye, target, up):
    normalize = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    lookat = normalize(target - eye)
    right = normalize(np.cross(lookat, up))
    up = np.cross(right, lookat)

    # Construct a rotation matrix from the right, new_up, and forward vectors
    c2w = np.eye(4).reshape(1, 4, 4).repeat(eye.shape[0], axis=0)
    c2w[:, :3, :3] = np.stack((right, up, -lookat), axis=-1)
    c2w[:, :3, 3] = eye

    return c2w


def prepare_mesh(mesh, normalize_mesh=False, radius=0.7, source="glb", **kwargs):
    if normalize_mesh:
        # Set the scene to be centered at the coordinate
        mesh.apply_transform(trimesh.transformations.translation_matrix(-mesh.centroid))

        # Normalize the scene into a sphere with radius of 0.7
        mesh.apply_scale(radius / mesh.bounding_sphere.primitive.radius)

    # Transform the coordinates of the scene
    if source == "glb":
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        )
    elif source == "meshlab":
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )

    return mesh


def prepare_pyrender_scene(
    mesh, light_intensity=4.0, bg_intensity=255, preprocess_mesh=True, **kwargs
):
    if preprocess_mesh:
        mesh = prepare_mesh(mesh, **kwargs)

    pyrender_scene = pyrender.Scene.from_trimesh_scene(mesh)

    # Create a Pyrender camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))

    # Create a full ambient lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    nl = pyrender.Node(light=light, matrix=np.eye(4))

    # Prepare the pyrender scene
    pyrender_scene.bg_color = np.repeat(bg_intensity / 255, 3)
    pyrender_scene.add_node(nc)
    pyrender_scene.add_node(nl)

    return pyrender_scene, nc, nl


def prepare_test_camera_poses(
    n_views, elevation_deg=15.0, fovy_deg=70.0, camera_dist=1.5
):
    elevation_deg = np.full(n_views, elevation_deg)
    azimuth_deg = np.linspace(0, 360, n_views)
    camera_dists = np.full(n_views, camera_dist)
    fovy_deg = np.full(n_views, fovy_deg)

    # Convert to radian
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    fovy = np.deg2rad(fovy_deg)

    # Convert spherical coordinates to cartesian coordinates
    camera_positions = np.stack(
        [
            camera_dists * np.cos(elevation) * np.cos(azimuth),
            camera_dists * np.cos(elevation) * np.sin(azimuth),
            camera_dists * np.sin(elevation),
        ],
        axis=-1,
    )
    light_positions = camera_positions

    # Default scene center at origin
    center = np.zeros_like(camera_positions)
    # Default camera up direction as +z
    up = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3).repeat(n_views, axis=0)

    # Create camera poses
    camera_poses = look_at_ndarray(camera_positions, center, up)

    return camera_poses, fovy, light_positions


def render_scene(
    pyrender_scene: pyrender.Scene,
    nc: pyrender.Node,
    nl: pyrender.Node,
    c2w: np.ndarray,
    fovy: float,
    light_pose=None,
    renderer=None,
    width=768,
    height=768,
    **kwargs,
):
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(width, height)
    if light_pose is None:
        light_pose = c2w

    # Set camera pose and light pose
    nc.camera.yfov = fovy
    pyrender_scene.set_pose(nc, pose=c2w)

    # Set light pose
    pyrender_scene.set_pose(nl, pose=light_pose)

    # Render the scene
    color, depth = renderer.render(pyrender_scene)

    return color, depth


def render_test_video(
    obj_path, out_video, n_views=60, fps=15, width=768, height=768, **kwargs
):
    # Prepare image list
    images = []

    # Load obj/glb file
    scene = trimesh.load(obj_path, force="scene", merge_primitives=True)
    print(f"Found {len(scene.geometry.values())} geometry")

    pyrender_scene, nc, nl = prepare_pyrender_scene(
        scene, preprocess_mesh=True, **kwargs
    )

    # Prepare camera poses
    camera_poses, fovy, _ = prepare_test_camera_poses(n_views)

    # Create a Pyrender offscreen renderer
    renderer = pyrender.OffscreenRenderer(width, height)

    # For each perspective, rotate the object, render the scene and save the image
    for i in tqdm(range(n_views)):
        # Create a copy of the scene for rendering
        c2w = camera_poses[i]
        color, _ = render_scene(
            pyrender_scene, nc, nl, c2w, fovy[i], c2w, renderer=renderer, **kwargs
        )
        images.append(color)

    # Save the video
    imageio.mimsave(out_video, images, fps=fps)
    print(f"Saved to {out_video}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_in", type=str, help="path to mesh file")
    parser.add_argument("--pth_out", type=str, help="path to output render video")
    parser.add_argument("--n_views", type=int, default=60, help="number of views")
    parser.add_argument("--fps", type=int, default=15, help="fps of the video")
    parser.add_argument("--normalize_mesh", action="store_true", help="normalize mesh")
    parser.add_argument(
        "--source", choices=["glb", "shape", "meshlab"], default="shape"
    )
    parser.add_argument("--width", type=int, default=768, help="width of the video")
    parser.add_argument("--height", type=int, default=768, help="height of the video")
    parser.add_argument(
        "--light_intensity", type=float, default=4.0, help="light intensity"
    )
    parser.add_argument(
        "--bg_intensity", type=int, default=255, help="background intensity"
    )
    return parser.parse_args()


def main(cfg):
    if os.path.isdir(cfg.pth_in):
        os.makedirs(cfg.pth_out, exist_ok=True)
        for f in os.listdir(cfg.pth_in):
            if f.endswith(".obj") or f.endswith(".glb"):
                obj_path = os.path.join(args.pth_in, f)
                out_video = os.path.join(args.pth_out, f + ".mp4")
                print(f"Rendering {obj_path}")

                try:
                    render_test_video(
                        obj_path=obj_path, out_video=out_video, **vars(args)
                    )
                except Exception as e:
                    print(f"Failed to render {obj_path}: {e}")
    else:
        if "/" in cfg.pth_out:
            os.makedirs(os.path.dirname(cfg.pth_out), exist_ok=True)

        render_test_video(obj_path=args.pth_in, out_video=args.pth_out, **vars(args))


if __name__ == "__main__":
    args = parse_args()
    main(args)
