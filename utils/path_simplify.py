from shapely.geometry import LineString
import numpy as np
def adjust_path_points(simplified_points, target_points):
    while len(simplified_points) < target_points:
        # Find indices of the furthest consecutive points
        furthest_idx = max(range(len(simplified_points) - 1),
                           key=lambda i: np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Insert a midpoint
        midpoint = tuple(np.mean([simplified_points[furthest_idx], simplified_points[furthest_idx + 1]], axis=0))
        simplified_points.insert(furthest_idx + 1, midpoint)

    while len(simplified_points) > target_points:
        # Find indices of the closest consecutive points
        closest_idx = min(range(len(simplified_points) - 1),
                          key=lambda i: np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Remove one of the closest points
        del simplified_points[closest_idx]

def simplify_path_to_target_points(path, target_points, tolerance=0.1, tolerance_increment=0.05):
    if len(path) < 2 or target_points >= len(path):
        return path

    line = LineString(path)
    simplified_line = line.simplify(tolerance)
    simplified_points = list(simplified_line.coords)

    while len(simplified_points) != target_points:
        if len(simplified_points) > target_points:
            tolerance += tolerance_increment
        else:
            tolerance -= tolerance_increment
            tolerance_increment /= 2

        simplified_line = line.simplify(tolerance)
        simplified_points = list(simplified_line.coords)

        if tolerance <= 0 or tolerance_increment < 1e-5:
            break

    # Directly adjust the number of points to match the target
    adjust_path_points(simplified_points, target_points)

    return simplified_points