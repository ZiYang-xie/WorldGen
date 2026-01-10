import click
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
from sharp.utils import io
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)
from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)

from worldgen.utils.equirectangular import (
    extract_overlapping_views,
    get_view_extrinsics,
    merge_with_consensus,
    rotate_quaternions
)

from worldgen.utils.splat_utils import SplatFile

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

def build_sharp_model(device: torch.device) -> RGBGaussianPredictor:
    """Build and load the pretrained Sharp model."""
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor.to(device)

@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians

@torch.no_grad()
def predict_equirectangular(
    predictor: RGBGaussianPredictor,
    equirect_image: Image.Image,
    device: torch.device,
    face_size: int = 768,
    fov_deg: float = 95.0,
    num_horizontal: int = 6,
    use_consensus: bool = True,
) -> tuple[Gaussians3D, float]:
    """Predict Gaussians from an equirectangular (360°) image.

    Uses overlapping views with consensus-based merging for better seam handling.

    Args:
        predictor: The Gaussian predictor model.
        equirect_image: (H, W, 3) equirectangular image as numpy array.
        device: Device to run inference on.
        face_size: Size of each extracted perspective view.
        fov_deg: Field of view for each view (>90° creates overlap for consensus).
        num_horizontal: Number of horizontal views around the horizon.
        use_consensus: Whether to use consensus-based merging (reduces seam artifacts).

    Returns:
        Tuple of (merged Gaussians, focal length in pixels).
    """

    # Convert image to tensor (C, H, W)
    equirect_image = np.array(equirect_image)
    equirect_pt = torch.from_numpy(equirect_image.copy()).float().permute(2, 0, 1) / 255.0
    equirect_pt = equirect_pt.to(device)
    views = extract_overlapping_views(
        equirect_pt,
        view_size=face_size,
        fov_deg=fov_deg,
        num_horizontal=num_horizontal,
        num_polar_rings=1,  # No extra polar rings - just up/down views
    )

    f_px = face_size / (2 * math.tan(math.radians(fov_deg / 2)))

    all_gaussians = []
    all_forwards = []

    for view in views:
        view_np = (view.image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        gaussians_view = predict_image(predictor, view_np, f_px, device)

        extrinsics = get_view_extrinsics(view.forward, view.up, device)
        world_from_camera = torch.linalg.inv(extrinsics)
        rotation = world_from_camera[:3, :3]

        mean_vectors = gaussians_view.mean_vectors @ rotation.T
        quaternions = rotate_quaternions(gaussians_view.quaternions, rotation)

        transformed_gaussians = Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=gaussians_view.singular_values,
            quaternions=quaternions,
            colors=gaussians_view.colors,
            opacities=gaussians_view.opacities,
        )

        all_gaussians.append(transformed_gaussians)
        all_forwards.append(view.forward)

    # Merge all Gaussians
    if use_consensus:
        merged_gaussians = merge_with_consensus(
            all_gaussians,
            all_forwards,
            fov_deg=fov_deg,
            voxel_size=0.02,
            depth_tolerance=0.15,
        )
    else:
        merged_gaussians = _merge_gaussians(all_gaussians)

    positions = merged_gaussians.mean_vectors.squeeze(0)
    scales = merged_gaussians.singular_values.squeeze(0)
    quats = merged_gaussians.quaternions.squeeze(0)
    colors = merged_gaussians.colors.squeeze(0)
    opacities = merged_gaussians.opacities.squeeze(0).unsqueeze(-1)


    R = quaternion_to_matrix(quats)
    S = torch.diag_embed(scales)
    covariances = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    # Convert colors from linearRGB to sRGB for proper display
    # (Sharp outputs linearRGB, but viser/displays expect sRGB)
    colors_srgb = cs_utils.linearRGB2sRGB(colors)

    return SplatFile(
        centers=positions.cpu().numpy(),
        rgbs=colors_srgb.cpu().numpy(),
        opacities=opacities.cpu().numpy(),
        covariances=covariances.cpu().numpy(),
        rotations=quats.cpu().numpy(),
        scales=scales.cpu().numpy(),
    )
