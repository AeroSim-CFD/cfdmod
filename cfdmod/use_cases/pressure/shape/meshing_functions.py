from typing import Tuple

import numpy as np


def project_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using normal vector

    Args:
        points (np.ndarray): Point cloud
        normal (np.ndarray): Surface normal containing points from point cloud

    Returns:
        np.ndarray: Points projected from point cloud into surface using its normal vector
    """
    projected = points - np.dot(points, normal)[:, np.newaxis] * normal
    return projected


def remove_colinear(vertices: np.ndarray, tolerance: float = 1e-4) -> np.ndarray:
    """Given a point cloud, remove all points that are colinear to another 2 points

    Args:
        vertices (np.ndarray): Point cloud
        tolerance (_type_, optional): Threshold to determine whether the points are colinear. Defaults to 1e-4.

    Returns:
        np.ndarray: _description_
    """
    i = 0
    while i < len(vertices) - 1:
        if i == len(vertices) - 2:
            p1, p2, p3 = (
                np.array(vertices[i]),
                np.array(vertices[i + 1]),
                np.array(vertices[0]),
            )
        else:
            p1, p2, p3 = (
                np.array(vertices[i]),
                np.array(vertices[i + 1]),
                np.array(vertices[i + 2]),
            )
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p3 - p1
        # Calculate cross product
        cross_product = np.cross(v1, v2)
        # If the magnitude of the cross product vector is close to zero,
        # the points are colinear
        if np.linalg.norm(cross_product) / np.linalg.norm(v3) < tolerance:
            del vertices[i + 1]  # Remove the middle vertex
        else:
            i += 1

    return np.array(vertices)


def find_intersection_point(
    line1_points: Tuple[np.ndarray, ...], line2_points: Tuple[np.ndarray, ...]
) -> np.ndarray:
    """Given two lines, find the intersection point from the lines crossing

    Args:
        line1_points (Tuple[np.ndarray, ...]): Tuple containing the points that form the line 1
        line2_points (Tuple[np.ndarray, ...]): Tuple containing the points that form the line 2

    Returns:
        np.ndarray: Point of intersection
    """
    point1, point2 = line1_points
    point3, point4 = line2_points

    direction1 = np.array(point2) - np.array(point1)
    direction2 = np.array(point4) - np.array(point3)

    # Check if the lines are parallel
    if np.all(np.cross(direction1, direction2) == 0):
        # Check if the lines are coincident
        # if np.all(np.cross(np.array(point3) - np.array(point1), direction1) == 0):
        #     return

        return None  # The lines are parallel but not coincident
    # Calculate the parameters t1 and t2 for the intersection point
    t2 = (
        np.dot(
            np.cross(np.array(point1) - np.array(point3), direction1),
            np.cross(direction2, direction1),
        )
        / np.linalg.norm(np.cross(direction2, direction1)) ** 2
    )
    t1 = (
        np.dot(np.array(point3) - np.array(point1) + t2 * direction2, direction1)
        / np.linalg.norm(direction1) ** 2
    )

    # Calculate the intersection point coordinates
    intersection_point = np.array(point1) + t1 * direction1

    return intersection_point


def remove_duplicates_3d_points(points_tuple: Tuple[np.ndarray, ...]) -> Tuple[Tuple, ...]:
    """Removes any duplicate points in a 3D point cloud

    Args:
        points_tuple (Tuple[np.ndarray, ...]): Tuple of 3D points

    Returns:
        Tuple[np.ndarray, ...]: Tuple of unique 3D points
    """
    unique_points_set = set(map(tuple, points_tuple))
    unique_points_array = np.array(list(unique_points_set))
    return tuple(map(tuple, unique_points_array))


def is_in_range(n: float, start: float, end: float) -> bool:
    """Check if a given number is in range

    Args:
        n (float): Target number
        start (float): Start of the range
        end (float): End of the range

    Returns:
        bool: Check if a given number is in range, returns True if it is
    """
    lower = min(start, end)
    upper = max(start, end)
    return lower <= n <= upper


def line_plane_intersection(
    P0: Tuple[float, float, float],
    N: Tuple[int, int, int],
    L1: Tuple[np.ndarray, np.ndarray],
    L2: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Defines the point of intersection between two lines and projeted onto a plane

    Args:
        P0 (Tuple[float, float, float]): Point in the plane
        N (Tuple[int, int, int]): Plane normal
        L1 (Tuple[np.ndarray, np.ndarray]): Line 1 containing 2 points
        L2 (Tuple[np.ndarray, np.ndarray]): Line 2 containing 2 points

    Returns:
        np.ndarray: Intersection point projeted onto plane
    """
    # Make sure the inputs are numpy arrays
    P0 = np.array(P0)
    N = np.array(N)
    L1 = np.array(L1)
    L2 = np.array(L2)
    # Calculate the direction vector of the line
    D = L2 - L1
    # Calculate the value of t
    if np.dot(N, D) != 0:
        t = np.dot(N, P0 - L1) / np.dot(N, D)
    else:
        t = 0
    # Calculate the point of intersection
    intersection = L1 + t * D

    return intersection
