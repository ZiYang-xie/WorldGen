from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class CubemapFaces(NamedTuple):
    """Container for 6 cubemap face images."""

    front: torch.Tensor  # +Z direction
    back: torch.Tensor   # -Z direction
    right: torch.Tensor  # +X direction
    left: torch.Tensor   # -X direction
    up: torch.Tensor     # -Y direction (looking up)
    down: torch.Tensor   # +Y direction (looking down)


class PerspectiveView(NamedTuple):
    """A perspective view extracted from equirectangular."""

    image: torch.Tensor  # (C, H, W) image tensor
    direction: torch.Tensor  # (3,) forward direction vector
    up: torch.Tensor  # (3,) up vector
    fov_deg: float  # field of view in degrees


def equirectangular_to_direction(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert equirectangular pixel coordinates to 3D direction vectors.

    Args:
        x: Horizontal pixel coordinates (0 to width-1).
        y: Vertical pixel coordinates (0 to height-1).
        width: Image width.
        height: Image height.

    Returns:
        Tuple of (dx, dy, dz) direction components.
        Uses convention: +X right, +Y down, +Z forward.
    """

    theta = (x / (width - 1)) * 2 * math.pi - math.pi  # longitude: -π to π
    phi = math.pi / 2 - (y / (height - 1)) * math.pi  # latitude: +π/2 to -π/2

    dx = torch.cos(phi) * torch.sin(theta)
    dy = -torch.sin(phi)  # Negate because +Y is down in OpenCV convention
    dz = torch.cos(phi) * torch.cos(theta)

    return dx, dy, dz

def rotate_quaternions(quaternions: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    from sharp.utils.linalg import (
        quaternions_from_rotation_matrices,
        rotation_matrices_from_quaternions,
    )
    rot_matrices = rotation_matrices_from_quaternions(quaternions)
    rot_matrices_world = rotation @ rot_matrices
    return quaternions_from_rotation_matrices(rot_matrices_world)


def direction_to_equirectangular(
    dx: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D direction vectors to equirectangular pixel coordinates.

    Args:
        dx, dy, dz: Direction vector components.
        width: Output image width.
        height: Output image height.

    Returns:
        Tuple of (x, y) pixel coordinates.
    """
    theta = torch.atan2(dx, dz)  # longitude: -π to π
    phi = torch.asin((-dy).clamp(-1, 1))  # latitude: -π/2 to +π/2

    x = (theta + math.pi) / (2 * math.pi) * (width - 1)
    y = (math.pi / 2 - phi) / math.pi * (height - 1)

    return x, y


def create_rotation_matrix(
    forward: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Create a rotation matrix from forward and up vectors.

    The rotation matrix transforms from camera coordinates to world coordinates.
    Camera convention: +X right, +Y down, +Z forward.

    Args:
        forward: (3,) forward direction vector in world space.
        up: (3,) up direction vector in world space (will be orthogonalized).

    Returns:
        (3, 3) rotation matrix where columns are [right, down, forward].
    """
    forward = forward / forward.norm()
    right = torch.cross(forward, up)
    right = right / right.norm()

    down = torch.cross(forward, right)
    rotation = torch.stack([right, down, forward], dim=1)
    return rotation


def get_cubemap_face_params(device: torch.device) -> dict:
    """Get the face parameters (forward, up vectors) for all cubemap faces.

    Returns:
        Dictionary mapping face name to (forward, up) tensor tuples.
    """
    return {
        "front": (
            torch.tensor([0.0, 0.0, 1.0], device=device),   # Looking at +Z
            torch.tensor([0.0, -1.0, 0.0], device=device),  # World -Y is up in image
        ),
        "back": (
            torch.tensor([0.0, 0.0, -1.0], device=device),  # Looking at -Z
            torch.tensor([0.0, -1.0, 0.0], device=device),  # World -Y is up in image
        ),
        "right": (
            torch.tensor([1.0, 0.0, 0.0], device=device),   # Looking at +X
            torch.tensor([0.0, -1.0, 0.0], device=device),  # World -Y is up in image
        ),
        "left": (
            torch.tensor([-1.0, 0.0, 0.0], device=device),  # Looking at -X
            torch.tensor([0.0, -1.0, 0.0], device=device),  # World -Y is up in image
        ),
        "up": (
            torch.tensor([0.0, -1.0, 0.0], device=device),  # Looking at -Y (up in world)
            torch.tensor([0.0, 0.0, -1.0], device=device),  # World -Z is up in image
        ),
        "down": (
            torch.tensor([0.0, 1.0, 0.0], device=device),   # Looking at +Y (down in world)
            torch.tensor([0.0, 0.0, 1.0], device=device),   # World +Z is up in image
        ),
    }


def extract_perspective_from_equirectangular(
    equirect: torch.Tensor,
    direction: torch.Tensor,
    up: torch.Tensor,
    fov_deg: float = 90.0,
    output_size: int = 512,
) -> torch.Tensor:
    """Extract a perspective view from an equirectangular image.

    Args:
        equirect: (C, H, W) equirectangular image tensor.
        direction: (3,) forward direction vector for the view.
        up: (3,) up vector for the view.
        fov_deg: Field of view in degrees.
        output_size: Size of the output square image.

    Returns:
        (C, output_size, output_size) perspective image.
    """
    device = equirect.device
    dtype = equirect.dtype
    _, eq_h, eq_w = equirect.shape

    # Create rotation matrix from view direction
    rotation = create_rotation_matrix(direction.to(device), up.to(device))

    # Create pixel grid for output image
    fov_rad = fov_deg * math.pi / 180
    focal = output_size / (2 * math.tan(fov_rad / 2))

    y_coords = torch.arange(output_size, device=device, dtype=dtype)
    x_coords = torch.arange(output_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Convert to camera-space directions
    cx = (output_size - 1) / 2
    cy = (output_size - 1) / 2
    dx_cam = (xx - cx) / focal
    dy_cam = (yy - cy) / focal
    dz_cam = torch.ones_like(dx_cam)

    # Normalize directions
    norm = torch.sqrt(dx_cam**2 + dy_cam**2 + dz_cam**2)
    dx_cam = dx_cam / norm
    dy_cam = dy_cam / norm
    dz_cam = dz_cam / norm

    # Rotate to world space
    dirs_cam = torch.stack([dx_cam, dy_cam, dz_cam], dim=-1)  # (H, W, 3)
    dirs_world = torch.einsum("ij,hwj->hwi", rotation, dirs_cam)

    # Convert world directions to equirectangular coordinates
    dx = dirs_world[..., 0]
    dy = dirs_world[..., 1]
    dz = dirs_world[..., 2]

    eq_x, eq_y = direction_to_equirectangular(dx, dy, dz, eq_w, eq_h)

    # Normalize to [-1, 1] for grid_sample
    grid_x = (eq_x / (eq_w - 1)) * 2 - 1
    grid_y = (eq_y / (eq_h - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    # Sample from equirectangular image
    equirect_batch = equirect.unsqueeze(0)
    perspective = F.grid_sample(
        equirect_batch,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return perspective[0]


def extract_cubemap_from_equirectangular(
    equirect: torch.Tensor,
    face_size: int = 512,
) -> CubemapFaces:
    """Extract 6 cubemap faces from an equirectangular image.

    Args:
        equirect: (C, H, W) equirectangular image tensor.
        face_size: Size of each cubemap face.

    Returns:
        CubemapFaces containing 6 perspective views.
    """
    device = equirect.device
    face_params = get_cubemap_face_params(device)

    faces = {}
    for name, (direction, up) in face_params.items():
        faces[name] = extract_perspective_from_equirectangular(
            equirect, direction, up, fov_deg=90.0, output_size=face_size
        )

    return CubemapFaces(**faces)


def cubemap_to_equirectangular(
    faces: CubemapFaces,
    output_width: int = 2048,
    output_height: int = 1024,
) -> torch.Tensor:
    """Convert cubemap faces to equirectangular projection.

    Args:
        faces: CubemapFaces containing 6 perspective views.
        output_width: Width of output equirectangular image.
        output_height: Height of output equirectangular image.

    Returns:
        (C, output_height, output_width) equirectangular image.
    """
    device = faces.front.device
    dtype = faces.front.dtype
    channels = faces.front.shape[0]
    face_size = faces.front.shape[1]

    # Create output pixel grid
    y_coords = torch.arange(output_height, device=device, dtype=dtype)
    x_coords = torch.arange(output_width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Convert to 3D directions in world space
    dx, dy, dz = equirectangular_to_direction(xx, yy, output_width, output_height)

    # Stack into direction vectors (H, W, 3)
    world_dirs = torch.stack([dx, dy, dz], dim=-1)

    # Determine which face each pixel belongs to
    abs_dx = torch.abs(dx)
    abs_dy = torch.abs(dy)
    abs_dz = torch.abs(dz)

    # Initialize output
    equirect = torch.zeros(channels, output_height, output_width, device=device, dtype=dtype)

    # Face data
    face_data = [
        ("front", faces.front, (abs_dz >= abs_dx) & (abs_dz >= abs_dy) & (dz > 0)),
        ("back", faces.back, (abs_dz >= abs_dx) & (abs_dz >= abs_dy) & (dz < 0)),
        ("right", faces.right, (abs_dx >= abs_dy) & (abs_dx >= abs_dz) & (dx > 0)),
        ("left", faces.left, (abs_dx >= abs_dy) & (abs_dx >= abs_dz) & (dx < 0)),
        ("up", faces.up, (abs_dy >= abs_dx) & (abs_dy >= abs_dz) & (dy < 0)),
        ("down", faces.down, (abs_dy >= abs_dx) & (abs_dy >= abs_dz) & (dy > 0)),
    ]

    for face_name, face_img, mask in face_data:
        if not mask.any():
            continue

        # Get the rotation matrix for this face (camera to world)
        # We need world to camera, which is the transpose
        extrinsics = get_cubemap_extrinsics(face_name, device)
        R_world_to_cam = extrinsics[:3, :3]

        # Transform world directions to camera space
        masked_dirs = world_dirs[mask]  # (N, 3)
        cam_dirs = masked_dirs @ R_world_to_cam.T  # (N, 3)

        # Project to UV (perspective projection)
        # u = cam_x / cam_z, v = cam_y / cam_z
        cam_z = cam_dirs[:, 2].clamp(min=1e-6)  # Avoid division by zero
        u = cam_dirs[:, 0] / cam_z
        v = cam_dirs[:, 1] / cam_z

        # Convert UV to grid coordinates for grid_sample
        # UV is in range [-1, 1] for 90° FOV
        grid_x = u.clamp(-1, 1)
        grid_y = v.clamp(-1, 1)

        # Create grid for sampling
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        face_batch = face_img.unsqueeze(0)  # (1, C, H, W)

        sampled = F.grid_sample(
            face_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        equirect[:, mask] = sampled[0, :, 0, :]

    return equirect


def get_cubemap_extrinsics(face_name: str, device: torch.device) -> torch.Tensor:
    """Get the extrinsics matrix for a cubemap face.

    The extrinsics matrix transforms from world coordinates to camera coordinates.
    This is the inverse of the rotation matrix used in extract_cubemap_from_equirectangular.

    Args:
        face_name: One of 'front', 'back', 'right', 'left', 'up', 'down'.
        device: Device to create tensor on.

    Returns:
        (4, 4) extrinsics matrix in OpenCV format (world to camera transform).
    """
    face_params = get_cubemap_face_params(device)

    if face_name not in face_params:
        raise ValueError(f"Unknown face name: {face_name}")

    forward, up = face_params[face_name]
    R_cam_to_world = create_rotation_matrix(forward, up)
    R_world_to_cam = R_cam_to_world.T

    extrinsics = torch.eye(4, device=device, dtype=torch.float32)
    extrinsics[:3, :3] = R_world_to_cam
    return extrinsics


def get_cubemap_intrinsics(face_size: int, device: torch.device) -> torch.Tensor:
    """Get intrinsics matrix for a cubemap face (90° FOV).

    Args:
        face_size: Size of the cubemap face in pixels.
        device: Device to create tensor on.

    Returns:
        (4, 4) intrinsics matrix.
    """
    # For 90° FOV, focal length = face_size / 2
    f_px = face_size / 2.0
    cx = (face_size - 1) / 2.0
    cy = (face_size - 1) / 2.0

    intrinsics = torch.tensor([
        [f_px, 0.0, cx, 0.0],
        [0.0, f_px, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], device=device, dtype=torch.float32)

    return intrinsics


# =============================================================================
# Overlapping Views for Better Seam Handling
# =============================================================================


class OverlappingView(NamedTuple):
    """An overlapping perspective view with metadata."""

    name: str
    image: torch.Tensor  # (C, H, W)
    forward: torch.Tensor  # (3,) forward direction
    up: torch.Tensor  # (3,) up direction
    fov_deg: float


def get_overlapping_view_params(
    device: torch.device,
    num_horizontal: int = 8,
    num_polar_rings: int = 1,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Generate overlapping view parameters for better seam handling.

    Uses more views with overlapping coverage to get consensus in overlap regions.

    Args:
        device: Device to create tensors on.
        num_horizontal: Number of views around the horizon (default 8).
        num_polar_rings: Number of additional rings above/below horizon (default 1).

    Returns:
        Dictionary mapping view name to (forward, up) tensor tuples.
    """
    views = {}

    # Horizontal ring at horizon (elevation = 0)
    for i in range(num_horizontal):
        angle = 2 * math.pi * i / num_horizontal
        forward = torch.tensor([
            math.sin(angle),
            0.0,
            math.cos(angle),
        ], device=device, dtype=torch.float32)
        up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
        views[f"h{i}"] = (forward, up)

    # Add polar views (up and down)
    views["up"] = (
        torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32),
        torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32),
    )
    views["down"] = (
        torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
    )

    # Optional: additional rings at ±45° elevation for better polar coverage
    if num_polar_rings >= 1:
        elevation = math.radians(45)
        cos_el = math.cos(elevation)
        sin_el = math.sin(elevation)

        # Upper ring
        for i in range(num_horizontal // 2):
            angle = 2 * math.pi * i / (num_horizontal // 2)
            forward = torch.tensor([
                cos_el * math.sin(angle),
                -sin_el,  # Looking upward
                cos_el * math.cos(angle),
            ], device=device, dtype=torch.float32)
            up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
            views[f"upper{i}"] = (forward, up)

        # Lower ring
        for i in range(num_horizontal // 2):
            angle = 2 * math.pi * i / (num_horizontal // 2)
            forward = torch.tensor([
                cos_el * math.sin(angle),
                sin_el,  # Looking downward
                cos_el * math.cos(angle),
            ], device=device, dtype=torch.float32)
            up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
            views[f"lower{i}"] = (forward, up)

    return views


def extract_overlapping_views(
    equirect: torch.Tensor,
    view_size: int = 768,
    fov_deg: float = 100.0,
    num_horizontal: int = 8,
    num_polar_rings: int = 1,
) -> list[OverlappingView]:
    """Extract overlapping perspective views from an equirectangular image.

    Args:
        equirect: (C, H, W) equirectangular image tensor.
        view_size: Size of each extracted view.
        fov_deg: Field of view in degrees (>90 creates overlap).
        num_horizontal: Number of views around the horizon.
        num_polar_rings: Number of additional elevation rings.

    Returns:
        List of OverlappingView containing extracted views with metadata.
    """
    device = equirect.device
    view_params = get_overlapping_view_params(device, num_horizontal, num_polar_rings)

    views = []
    for name, (forward, up) in view_params.items():
        image = extract_perspective_from_equirectangular(
            equirect, forward, up, fov_deg=fov_deg, output_size=view_size
        )
        views.append(OverlappingView(
            name=name,
            image=image,
            forward=forward,
            up=up,
            fov_deg=fov_deg,
        ))

    return views


def get_view_extrinsics(
    forward: torch.Tensor,
    up: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Get extrinsics matrix for an arbitrary view direction.

    Args:
        forward: (3,) forward direction vector.
        up: (3,) up direction vector.
        device: Device to create tensor on.

    Returns:
        (4, 4) extrinsics matrix (world to camera transform).
    """
    R_cam_to_world = create_rotation_matrix(forward.to(device), up.to(device))
    R_world_to_cam = R_cam_to_world.T

    extrinsics = torch.eye(4, device=device, dtype=torch.float32)
    extrinsics[:3, :3] = R_world_to_cam
    return extrinsics


def compute_view_weight(
    positions: torch.Tensor,
    view_forward: torch.Tensor,
    fov_deg: float,
    falloff_start: float = 0.7,
) -> torch.Tensor:
    """Compute per-Gaussian weights based on angular distance from view center.

    Gaussians near the center of a view get higher weight than those at edges.

    Args:
        positions: (N, 3) Gaussian positions in world space.
        view_forward: (3,) forward direction of the view.
        fov_deg: Field of view of the view in degrees.
        falloff_start: Start of falloff as fraction of half-FOV (0.7 = 70% from center).

    Returns:
        (N,) weight tensor, 1.0 at center, falling to 0 at edges.
    """
    # Normalize positions to get directions
    directions = positions / (positions.norm(dim=-1, keepdim=True) + 1e-8)

    cos_angle = (directions @ view_forward).clamp(-1, 1)
    angle_rad = torch.acos(cos_angle)
    half_fov_rad = math.radians(fov_deg / 2)

    falloff_angle = half_fov_rad * falloff_start
    weight = torch.where(
        angle_rad < falloff_angle,
        torch.ones_like(angle_rad),
        1.0 - (angle_rad - falloff_angle) / (half_fov_rad - falloff_angle + 1e-8),
    )
    weight = weight.clamp(0, 1)

    # Apply smoothstep for smoother transition
    weight = weight * weight * (3 - 2 * weight)

    return weight


def merge_with_consensus(
    gaussians_list: list,  # list of Gaussians3D
    view_forwards: list[torch.Tensor],
    fov_deg: float,
    voxel_size: float = 0.02,
    depth_tolerance: float = 0.15,
) -> "Gaussians3D":
    """Merge Gaussians from overlapping views using depth consensus.

    For regions covered by multiple views, keeps Gaussians whose depth
    is consistent with predictions from other views.

    This implementation uses vectorized operations for speed.

    Args:
        gaussians_list: List of Gaussians3D from each view.
        view_forwards: List of (3,) forward direction tensors for each view.
        fov_deg: Field of view used for extraction.
        voxel_size: Size of voxels for spatial grouping.
        depth_tolerance: Relative depth difference tolerance for consensus.

    Returns:
        Merged Gaussians3D with consensus-filtered opacities.
    """
    from sharp.utils.gaussians import Gaussians3D

    if len(gaussians_list) == 0:
        raise ValueError("Empty gaussians list")

    device = gaussians_list[0].mean_vectors.device

    # Collect all Gaussians with their source view info
    all_positions = []
    all_singular_values = []
    all_quaternions = []
    all_colors = []
    all_opacities = []
    all_view_weights = []

    for view_id, (gaussians, view_forward) in enumerate(zip(gaussians_list, view_forwards)):
        positions = gaussians.mean_vectors[0]  # (N, 3)

        # Compute weight based on distance from view center
        weights = compute_view_weight(positions, view_forward, fov_deg, falloff_start=0.6)

        all_positions.append(positions)
        all_singular_values.append(gaussians.singular_values[0])
        all_quaternions.append(gaussians.quaternions[0])
        all_colors.append(gaussians.colors[0])
        all_opacities.append(gaussians.opacities[0])
        all_view_weights.append(weights)

    # Concatenate all
    positions = torch.cat(all_positions, dim=0)  # (N_total, 3)
    singular_values = torch.cat(all_singular_values, dim=0)
    quaternions = torch.cat(all_quaternions, dim=0)
    colors = torch.cat(all_colors, dim=0)
    opacities = torch.cat(all_opacities, dim=0)
    view_weights = torch.cat(all_view_weights, dim=0)

    n_total = positions.shape[0]
    
    depths = positions.norm(dim=-1)
    voxel_indices = (positions / voxel_size).long()
    voxel_min = voxel_indices.min(dim=0).values
    voxel_indices = voxel_indices - voxel_min
    voxel_max = voxel_indices.max(dim=0).values + 1
    voxel_1d = (
        voxel_indices[:, 0] * voxel_max[1] * voxel_max[2] +
        voxel_indices[:, 1] * voxel_max[2] +
        voxel_indices[:, 2]
    )

    # Get unique voxels and inverse mapping
    unique_voxels, inverse_indices = torch.unique(voxel_1d, return_inverse=True)
    n_voxels = len(unique_voxels)

    weighted_depths = depths * view_weights
    voxel_weighted_depth_sum = torch.zeros(n_voxels, device=device, dtype=positions.dtype)
    voxel_weighted_depth_sum.scatter_add_(0, inverse_indices, weighted_depths)
    voxel_weight_sum = torch.zeros(n_voxels, device=device, dtype=positions.dtype)
    voxel_weight_sum.scatter_add_(0, inverse_indices, view_weights)
    voxel_count = torch.zeros(n_voxels, device=device, dtype=torch.long)
    voxel_count.scatter_add_(0, inverse_indices, torch.ones(n_total, device=device, dtype=torch.long))

    voxel_consensus_depth = voxel_weighted_depth_sum / (voxel_weight_sum + 1e-8)
    consensus_depth_per_gaussian = voxel_consensus_depth[inverse_indices]
    count_per_gaussian = voxel_count[inverse_indices]

    rel_diff = torch.abs(depths - consensus_depth_per_gaussian) / (consensus_depth_per_gaussian + 1e-6)
    depth_weight = (1.0 - rel_diff / depth_tolerance).clamp(0, 1)
    depth_weight = depth_weight * depth_weight * (3 - 2 * depth_weight)

    consensus_weights = torch.where(count_per_gaussian > 1, depth_weight, torch.ones_like(depth_weight))

    final_opacities = opacities * view_weights * consensus_weights

    merged = Gaussians3D(
        mean_vectors=positions.unsqueeze(0),
        singular_values=singular_values.unsqueeze(0),
        quaternions=quaternions.unsqueeze(0),
        colors=colors.unsqueeze(0),
        opacities=final_opacities.unsqueeze(0),
    )

    return merged
