import viser
import torch
import os
import imageio
import numpy as np
from PIL import Image

from typing import Optional, Union, Tuple
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import time
from pathlib import Path
from worldgen.utils.splat_utils import SplatFile
from worldgen import WorldGen
import open3d as o3d
import trimesh

def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    q1 = np.array(q1)
    q2 = np.array(q2)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.sum(q1 * q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = min(1.0, max(-1.0, dot))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return q1 * (1 - t) + q2 * t

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return q1 * s1 + q2 * s2


class ViserServer:
    def __init__(self, args):
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("-y")
        self.server.scene.enable_default_lights(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_mesh = False

        if args.return_mesh:
            self.mesh = None
            self.return_mesh = True
            assert (not args.inpaint_bg), "inpaint_bg is not supported when return_mesh is True"

        if args.use_sharp:
            print("\033[92m" + "=" * 80 + "\033[0m")
            print("\033[95m 🚀 INFO: You are using the ml-sharp experimental feature to generate the world \033[0m")
            print("\033[92m" + "=" * 80 + "\033[0m")

        if args.inpaint_bg:
            print("\033[93m" + "!" * 70 + "\033[0m")
            print("\033[93m⚠️  WARNING: You are using the inpaint_bg experimental feature ⚠️\033[0m")
            print("\033[93m⚠️  This feature may produce worse results than the default mode ⚠️\033[0m")
            print("\033[93m" + "!" * 70 + "\033[0m")
        
        mode = "t2s" if args.image is None else "i2s"
        self.worldgen = WorldGen(mode=mode, 
                                 use_sharp=args.use_sharp,
                                 inpaint_bg=args.inpaint_bg, 
                                 resolution=args.resolution,
                                 device=self.device,
                                 low_vram=args.low_vram)
        self.args = args
        self.frames = []
        self.start_camera = None

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.2,
        position: Tuple[float, float, float] = (0, 0, 0),
        wxyz: Tuple[float, float, float, float] = (1, 0, 0, 0),
        color: Tuple[int, int, int] = (0, 255, 0),
        visible: bool = True,
    ):
        return self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=aspect,
            scale=scale,
            position=position,
            wxyz=wxyz,
            color=color,
            visible=visible,
        )

    def add_gs(self, splat: SplatFile):
        if self.args.save_scene:
            splat.save(os.path.join(self.args.output_dir, "splat.ply"))
        self.scene_gs_handle = self.server.scene.add_gaussian_splats(
            "/scene_gs",
            centers=splat.centers,
            rgbs=splat.rgbs,
            opacities=splat.opacities,
            covariances=splat.covariances,
        )
    
    def add_mesh(self, mesh: o3d.geometry.TriangleMesh):
        if self.args.save_scene:
            file_path = os.path.join(self.args.output_dir, "mesh.glb")
            o3d.io.write_triangle_mesh(file_path, mesh)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        trimesh_mesh.visual.vertex_colors = colors
        self.scene_mesh_handle = self.server.scene.add_mesh_trimesh(name="/scene_mesh", mesh=trimesh_mesh)

    def add_original_camera(self):
        h, w = 1080, 1920
        fov = np.deg2rad(90)
        aspect = w / h
        self.original_camera = self.server.scene.add_camera_frustum("original_camera", fov, aspect)
        self.init_h, self.init_w = h, w
        self.original_camera.visible = False

    def prepare_render_visibility(self):
        self.original_camera.visible = False
        for frame in self.frames:
            frame.visible = False
        
        if hasattr(self, 'gs_transform_controls'):
            self.gs_transform_controls.scale = 0.0

    def restore_render_visibility(self):
        self.original_camera.visible = True
        for frame in self.frames:
            frame.visible = True
        if hasattr(self, 'gs_transform_controls'):
            self.gs_transform_controls.scale = 2.0

    def save_novel_views(self, client):
        # Get render parameters from UI
        render_h = self.render_height_input.value
        render_w = self.render_width_input.value
        render_fov_deg = self.render_fov_input.value
        render_fov_rad = np.deg2rad(render_fov_deg)

        print(f"Starting to save novel views ({render_h}x{render_w}, FoV: {render_fov_deg}°)")
        image_dir = os.path.join(self.args.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Render all cameras
        start_idx = 0
        self.prepare_render_visibility()
        rgb_video_writer = imageio.get_writer(os.path.join(self.args.output_dir, "rgb.mp4"), fps=30)
        for i, frame in tqdm(enumerate(self.frames), total=len(self.frames)):
            # Use values from UI for rendering
            rendered_image = client.get_render(
                height=render_h,
                width=render_w,
                wxyz=frame.wxyz,
                position=frame.position,
                fov=render_fov_rad # Use FoV from UI (in radians)
            )
            imageio.imwrite(f"{image_dir}/{i+start_idx:04d}.png", rendered_image)
            rgb_video_writer.append_data(rendered_image)
        rgb_video_writer.close()
        self.restore_render_visibility()

    def add_interpolated_cameras(self, client):
        current_wxyz = client.camera.wxyz
        current_position = client.camera.position

        steps = self.interpolation_steps.value
        current_fov = self.original_camera.fov
        current_aspect = self.original_camera.aspect

        # Add click handler to teleport to this camera view
        def create_click_handler(f):
            def click_handler(_):
                with client.atomic():
                    client.camera.wxyz = f.wxyz
                    client.camera.position = f.position
                    client.camera.fov = f.fov
            return click_handler

        if self.start_camera is None:
            self.start_camera = self.add_camera_frustum(
                "/start_camera",
                fov=current_fov,
                aspect=current_aspect,
                position=current_position,
                wxyz=current_wxyz,
                color=(0, 0, 0),
            )
            self.start_camera.on_click(create_click_handler(self.start_camera))
            self.frames.append(self.start_camera)
            return 

        start_wxyz = self.start_camera.wxyz
        start_position = self.start_camera.position
        camera_counter = [0]

        # Create interpolated cameras
        for i in range(1, steps + 1):
            # Calculate interpolation factor (0 to 1)
            t = i / steps
            
            # Apply ease-in-out effect (cubic bezier)
            t_eased = t * t * (3 - 2 * t)  # Smooth step function for ease-in-out

            interp_position = t_eased * current_position + (1 - t_eased) * start_position
            interp_wxyz = quaternion_slerp(start_wxyz, current_wxyz, t_eased)

            c2w = torch.eye(4, dtype=torch.float64, device=self.device)
            c2w[:3, :3] = torch.from_numpy(
                R.from_quat(interp_wxyz, scalar_first=True).as_matrix()
            )
            c2w[:3, 3] = torch.tensor(interp_position, device=self.device)

            # Create a new camera frustum at the interpolated position
            camera_name = f"/{camera_counter[0]}"
            camera_counter[0] += 1

            # Color gradient from blue to green
            r = int(0 * (1 - t) + 0 * t)
            g = int(150 * (1 - t) + 255 * t)
            b = int(255 * (1 - t) + 0 * t)

            # Add camera frustum
            frustum = self.server.scene.add_camera_frustum(
                camera_name,
                fov=current_fov,
                aspect=current_aspect,
                scale=0.2,
                wxyz=interp_wxyz,
                position=interp_position,
                color=(r, g, b),  # Color gradient from blue to green
            )

            frustum.on_click(create_click_handler(frustum))
            self.frames.append(frustum)

        print(f"Added camera path with {steps+1} cameras")

    def create_ui(self, client):
        initial_fov_rad = self.original_camera.fov
        initial_fov_deg = np.rad2deg(initial_fov_rad)
        client.camera.position = (0, 0, 0)
        client.camera.wxyz = (1, 0, 0, 0)
        client.camera.fov = initial_fov_rad
        client.camera.far = 10000
        client.camera.near = 0.01

        with client.gui.add_folder("Camera Path"):
            self.interpolation_steps = client.gui.add_slider(
                "Interpolation Steps", min=1, max=1000, step=1, initial_value=120
            )
            self.add_camera_path_button = client.gui.add_button("Generate Camera Path")

        with client.gui.add_folder("Render Settings"):
            self.render_fov_input = client.gui.add_number(
                "Render FoV (deg)", initial_value=initial_fov_deg, min=1.0, max=179.0, step=5
            )
            self.render_height_input = client.gui.add_number(
                "Render Height", initial_value=self.init_h, min=64, max=4096, step=1
            )
            self.render_width_input = client.gui.add_number(
                "Render Width", initial_value=self.init_w, min=64, max=4096, step=1
            )
            self.save_button = client.gui.add_button("Save Novel Views")

            # Update client camera FoV when the input changes
            @self.render_fov_input.on_update
            def _(value):
                client.camera.fov = np.deg2rad(self.render_fov_input.value)

    def generate_world(self):
        if self.args.pano_image is not None:
            pano_image = Image.open(self.args.pano_image).convert("RGB")
            pano_image = pano_image.resize((2048, 1024))
            scene = self.worldgen._generate_world(pano_image, return_mesh=self.return_mesh)
        elif self.args.image is not None:
            image = Image.open(self.args.image).convert("RGB")
            scene = self.worldgen.generate_world(self.args.prompt, image, return_mesh=self.return_mesh)
        else:
            scene = self.worldgen.generate_world(self.args.prompt, return_mesh=self.return_mesh)
        return scene

    def set_bg(self, splat: SplatFile):
        # Use black background
        bg_img = np.zeros((1, 1, 3))
        self.server.scene.set_background_image(bg_img)

    def run(self):
        print("\033[92m" + "=" * 70 + "\033[0m")
        print("\033[95m🌍 Generating the world... Please wait 🌍\033[0m")
        print("\033[92m" + "=" * 70 + "\033[0m")

        scene = self.generate_world()
        if self.return_mesh:
            self.add_mesh(scene)
        else:
            self.add_gs(scene)
            self.set_bg(scene)
        self.add_original_camera()

        print("\033[92m" + "=" * 70 + "\033[0m")
        print("\033[95m✨ World successfully generated! ✨\033[0m")
        print("\033[95mOpen your browser at http://localhost:8080 to view the scene\033[0m")
        print("\033[92m" + "=" * 70 + "\033[0m")

        @self.server.on_client_connect
        def connect(client: viser.ClientHandle) -> None:
            self.create_ui(client)
            @self.original_camera.on_click
            def _(_):
                with client.atomic():
                    client.camera.wxyz = self.original_camera.wxyz
                    client.camera.position = self.original_camera.position
                    client.camera.fov = self.original_camera.fov

            @self.save_button.on_click
            def _(_):
                self.save_novel_views(client)

            @self.add_camera_path_button.on_click
            def _(_):
                self.add_interpolated_cameras(client)


        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="World Generation Demo with Viser")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt for world generation")
    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Path to output directory")
    parser.add_argument("--resolution", "-r", type=int, default=1600, help="Resolution of the generated world")
    parser.add_argument("--pano_image", type=str, default=None, help="Path to input Panorama image")
    parser.add_argument("--use_sharp", action="store_true", help="Whether to use the ml-sharp experimental feature")
    parser.add_argument("--inpaint_bg", action="store_true", help="Whether to inpaint the background")
    parser.add_argument("--return_mesh", action="store_true", help="Whether to return the mesh")
    parser.add_argument("--save_scene", action="store_true", help="Whether to save the scene")
    parser.add_argument("--low_vram", action="store_true", help="Whether to use low VRAM")
    args = parser.parse_args()

    if torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) < 24:
        print("Detected GPU VRAM less than 24GB, setting low_vram to True")
        args.low_vram = True

    server = ViserServer(args)
    server.run()
