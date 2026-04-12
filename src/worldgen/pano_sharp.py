import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.utils.data
from pytorch3d.transforms import quaternion_to_matrix
from PIL import Image

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    unproject_gaussians,
)

from worldgen.utils.equirectangular import (
    extract_cubemap_from_equirectangular,
    get_cubemap_face_params,
    get_cubemap_extrinsics,
    rotate_quaternions,
)

from worldgen.utils.splat_utils import SplatFile

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

CUBEMAP_FACE_NAMES = ["front", "back", "right", "left", "up", "down"]


def build_sharp_model(device: torch.device) -> RGBGaussianPredictor:
    """Build and load the pretrained Sharp model."""
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor.to(device)


@torch.no_grad()
def predict_cubemap_face(
    predictor: RGBGaussianPredictor,
    face_rgb: torch.Tensor,
    face_depth: torch.Tensor,
    face_size: int,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from a cubemap face RGB, then align depth to DA-2.

    Args:
        predictor: The Sharp predictor model.
        face_rgb: (3, H, W) RGB tensor in [0, 1].
        face_depth: (1, H, W) depth from DA-2 for this face.
        face_size: Size of the cubemap face.
        device: Device.

    Returns:
        Gaussians3D in camera space, with depth aligned to DA-2.
    """
    internal_shape = (1536, 1536)

    # 90 deg FOV cubemap → focal = face_size / 2
    f_px = face_size / 2.0

    image_resized = F.interpolate(
        face_rgb.unsqueeze(0),
        size=internal_shape,
        mode="bilinear",
        align_corners=True,
    )

    disparity_factor = torch.tensor([f_px / face_size]).float().to(device)
    gaussians_ndc = predictor(image_resized, disparity_factor)

    f_resized = f_px * internal_shape[0] / face_size
    intrinsics_resized = torch.tensor([
        [f_resized, 0, internal_shape[0] / 2, 0],
        [0, f_resized, internal_shape[1] / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)

    # Unproject to camera-space metric coordinates using identity extrinsics
    gaussians_cam = unproject_gaussians(
        gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, internal_shape
    )

    # Align Sharp's depth to DA-2 depth
    # Sharp positions are in camera space: z is depth along the view axis
    sharp_depths = gaussians_cam.mean_vectors[0, :, 2]  # (N,) z-values
    valid = sharp_depths > 0.01

    # Compute median ratio between DA-2 depth and Sharp depth
    da2_median = face_depth[face_depth > 0.01].median()
    sharp_median = sharp_depths[valid].median()

    if sharp_median > 1e-6 and da2_median > 1e-6:
        scale = da2_median / sharp_median
        scaled_positions = gaussians_cam.mean_vectors * scale
        scaled_sv = gaussians_cam.singular_values * scale
    else:
        scaled_positions = gaussians_cam.mean_vectors
        scaled_sv = gaussians_cam.singular_values

    return Gaussians3D(
        mean_vectors=scaled_positions,
        singular_values=scaled_sv,
        quaternions=gaussians_cam.quaternions,
        colors=gaussians_cam.colors,
        opacities=gaussians_cam.opacities,
    )


@torch.no_grad()
def predict_equirectangular(
    predictor: RGBGaussianPredictor,
    equirect_image: Image.Image,
    device: torch.device,
    face_size: int = 768,
    depth_predictions: dict = None,
) -> SplatFile:
    """Predict Gaussians from an equirectangular image using 6 cubemap faces.

    Uses DA-2 equirectangular depth to align Sharp's per-face depth predictions
    for globally consistent geometry.

    Args:
        predictor: The Sharp Gaussian predictor model.
        equirect_image: Equirectangular PIL image.
        device: Device to run inference on.
        face_size: Size of each cubemap face.
        depth_predictions: Dict from pred_pano_depth with 'distance' key.

    Returns:
        SplatFile with merged Gaussians from all 6 cubemap faces.
    """
    equirect_np = np.array(equirect_image)
    equirect_pt = torch.from_numpy(equirect_np.copy()).float().permute(2, 0, 1) / 255.0
    equirect_pt = equirect_pt.to(device)

    # Extract 6 cubemap faces from equirectangular RGB
    cubemap_rgb = extract_cubemap_from_equirectangular(equirect_pt, face_size)

    # Extract 6 cubemap faces from equirectangular depth (DA-2)
    distance = depth_predictions["distance"]  # (H, W)
    distance_pt = distance.unsqueeze(0).to(device)  # (1, H, W)
    cubemap_depth = extract_cubemap_from_equirectangular(distance_pt, face_size)

    face_params = get_cubemap_face_params(device)

    all_positions = []
    all_singular_values = []
    all_quaternions = []
    all_colors = []
    all_opacities = []

    print(f"Processing 6 cubemap faces through Sharp model...")
    for i, face_name in enumerate(CUBEMAP_FACE_NAMES):
        print(f"  Face {i+1}/6: {face_name}...")
        face_rgb = getattr(cubemap_rgb, face_name)    # (3, H, W)
        face_depth = getattr(cubemap_depth, face_name)  # (1, H, W)

        # Run Sharp on this face and align to DA-2 depth
        gaussians_cam = predict_cubemap_face(
            predictor, face_rgb, face_depth.squeeze(0), face_size, device
        )

        # Transform from camera space to world space
        extrinsics = get_cubemap_extrinsics(face_name, device)
        world_from_camera = torch.linalg.inv(extrinsics)
        rotation = world_from_camera[:3, :3]

        positions_world = gaussians_cam.mean_vectors @ rotation.T
        quaternions_world = rotate_quaternions(gaussians_cam.quaternions, rotation)

        all_positions.append(positions_world.squeeze(0))
        all_singular_values.append(gaussians_cam.singular_values.squeeze(0))
        all_quaternions.append(quaternions_world.squeeze(0))
        all_colors.append(gaussians_cam.colors.squeeze(0))
        all_opacities.append(gaussians_cam.opacities.squeeze(0))

    print("Merging Gaussians from all faces...")
    positions = torch.cat(all_positions, dim=0)
    scales = torch.cat(all_singular_values, dim=0)
    quats = torch.cat(all_quaternions, dim=0)
    colors = torch.cat(all_colors, dim=0)
    opacities = torch.cat(all_opacities, dim=0).unsqueeze(-1)

    R = quaternion_to_matrix(quats)
    S = torch.diag_embed(scales)
    covariances = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    # Convert colors from linearRGB to sRGB for proper display
    colors_srgb = cs_utils.linearRGB2sRGB(colors)

    return SplatFile(
        centers=positions.cpu().numpy(),
        rgbs=colors_srgb.cpu().numpy(),
        opacities=opacities.cpu().numpy(),
        covariances=covariances.cpu().numpy(),
        rotations=quats.cpu().numpy(),
        scales=scales.cpu().numpy(),
    )
