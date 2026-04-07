"""
============================================================
SO(3) Rotation Scan for L^p-Norm Condition Numbers
============================================================

This script generates a dataset of condition numbers for a 
tetrahedral measurement frame under rotations in SO(3).

Rotation strategy:
    1. Rotate around Y-axis
    2. Then sweep X and Z rotations

Final rotation order:
    R = Rz @ Rx @ Ry

For each rotated configuration, we compute:
    - L1 norm condition number
    - L2 norm condition number
    - L-infinity norm condition number
    - Equally Weighted Variance (EWV)

The resulting dataset is saved as a CSV file.

Author: Yuxi Cai (University of Oxford)
============================================================
"""

import numpy as np
import csv
from tqdm import tqdm

# ============================================================
# Condition number definitions under different norms
# ============================================================

def l1_norm_condition_number(A):
    """
    Compute the L1-norm condition number of matrix A.

    ||A||_1 = max column sum
    """
    norm_A = np.max(np.sum(np.abs(A), axis=0))
    try:
        A_inv = np.linalg.inv(A)
        norm_A_inv = np.max(np.sum(np.abs(A_inv), axis=0))
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular.")
    return norm_A * norm_A_inv


def linf_norm_condition_number(A):
    """
    Compute the L-infinity norm condition number of matrix A.

    ||A||_∞ = max row sum
    """
    norm_A = np.max(np.sum(np.abs(A), axis=1))
    try:
        A_inv = np.linalg.inv(A)
        norm_A_inv = np.max(np.sum(np.abs(A_inv), axis=1))
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular.")
    return norm_A * norm_A_inv


def l2_norm_condition_number(A):
    """
    Compute the L2-norm (spectral) condition number via SVD.
    """
    singular_values = np.linalg.svd(A, compute_uv=False)
    return singular_values[0] / singular_values[-1]


def calculate_ewv(matrix):
    """
    Compute Equally Weighted Variance (EWV).

    This measures the variance of equally weighted combinations
    of the measurement vectors.
    """
    cov_matrix = np.cov(matrix.T)
    n = matrix.shape[1]
    weights = np.ones(n) / n
    return weights @ cov_matrix @ weights.T


# ============================================================
# Rotation matrices in SO(3)
# ============================================================

def rotation_matrix(axis, theta):
    """
    Generate a 3D rotation matrix.

    Parameters:
        axis  : 'x', 'y', or 'z'
        theta : rotation angle (radians)
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    elif axis == 'y':
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


# ============================================================
# Tetrahedral vertices on the Poincaré sphere
# ============================================================

# Regular tetrahedron (normalized Stokes vectors)
vertices = np.array([
    [1, 0, 0],
    [-1/3,  2*np.sqrt(2)/3, 0],
    [-1/3, -np.sqrt(2)/3,  np.sqrt(2/3)],
    [-1/3, -np.sqrt(2)/3, -np.sqrt(2/3)]
]).T  # shape = (3, 4)


# ============================================================
# Scan parameters
# ============================================================

step_deg = 1  # angular resolution (degrees)
output_file = "LAM_all_data.csv"

x_angles = np.arange(0, 360, step_deg)
y_angles = np.arange(0, 360, step_deg)
z_angles = np.arange(0, 360, step_deg)


# ============================================================
# Write CSV header
# ============================================================

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Rotation X (deg)',
        'Rotation Y (deg)',
        'Rotation Z (deg)',
        'L1 Norm Condition Number',
        'L2 Norm Condition Number',
        'L∞ Norm Condition Number',
        'EWV'
    ])


# ============================================================
# Main loop: SO(3) sampling
# ============================================================

"""
Rotation order:
    1. Rotate about Y-axis
    2. Then rotate about X-axis
    3. Then rotate about Z-axis

Matrix composition:
    R = Rz @ Rx @ Ry
"""

with open(output_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    for theta_y_deg in tqdm(y_angles, desc="Y-axis rotation"):
        rot_y = rotation_matrix('y', np.radians(theta_y_deg))

        for theta_x_deg in tqdm(x_angles, desc="X-axis rotation", leave=False):
            rot_x = rotation_matrix('x', np.radians(theta_x_deg))

            for theta_z_deg in tqdm(z_angles, desc="Z-axis rotation", leave=False):
                rot_z = rotation_matrix('z', np.radians(theta_z_deg))

                # Apply rotations to tetrahedron
                rotated_vertices = rot_z @ rot_x @ rot_y @ vertices

                # Construct 4×4 instrument matrix
                # First row: intensity component (S0 = 1)
                instrument_matrix = np.vstack([np.ones(4), rotated_vertices]).T * 0.5

                try:
                    # Compute condition numbers
                    l1 = l1_norm_condition_number(instrument_matrix)
                    l2 = l2_norm_condition_number(instrument_matrix)
                    l_inf = linf_norm_condition_number(instrument_matrix)
                    ewv = calculate_ewv(instrument_matrix)

                    writer.writerow([
                        theta_x_deg,
                        theta_y_deg,
                        theta_z_deg,
                        l1,
                        l2,
                        l_inf,
                        ewv
                    ])

                except ValueError:
                    # Skip singular configurations
                    continue


print(f"Computation complete. Results saved to {output_file}.")