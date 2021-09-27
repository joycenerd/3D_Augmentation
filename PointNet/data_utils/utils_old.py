import numpy as np
import open3d as o3d
from data_utils.z_order import *

multizorder_ranges = {5000: 500, 1024: 300, 64: 20, 32: 10}

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd = np.array(pcd.points).astype(np.float32)

    return pcd

def pcd_normalize(pcd):
    centroids = np.mean(pcd, axis=0)
    pcd = pcd - centroids
    max_dist = np.max(np.sqrt(np.sum(pcd**2, axis=1)))
    pcd = pcd / max_dist

    return pcd

def random_sample(points, num_points=1024):
    points = np.random.permutation(points)
    points = points[:num_points, :]

    return points

def farthest_point_sample(points, num_points=1024):
    """
    Input:
        points: a point set, in the format of NxM, where N is the number of points, and M is the point dimension
        num_points: required number of sampled points
    """
    def compute_dist(centroid, points):
        return np.sum((centroid - points) ** 2, axis=1)

    farthest_pts = np.zeros((num_points, points.shape[1]))
    farthest_pts[0] = points[np.random.randint(len(points))] # Random choose one point as starting point
    distances = compute_dist(farthest_pts[0], points)
    
    for idx in range(1, num_points):
        farthest_pts[idx] = points[np.argmax(distances)]
        distances = np.minimum(distances, compute_dist(farthest_pts[idx], points))
    
    return farthest_pts.astype(np.float32)

def get_z_values(points):
    # Get z values of points
    points_round = round_to_int_32(points) # convert to int
    z_values = get_z_order(points_round[:, 0], points_round[:, 1], points_round[:, 2])

    return z_values

def keep_zorder(points, num_points=1024):
    # Random a start index of z-order sequence
    sample_idx = np.random.randint(len(points) - num_points)
    points = points[sample_idx:sample_idx+num_points]

    return points

def keep_multizorder(points, num_points=1024):
    remain = num_points
    keep = np.array([], dtype=int)
    multizorder_range = multizorder_ranges[num_points]

    while len(keep) < num_points:
        keep_range = remain if remain <= multizorder_range else np.random.randint(multizorder_range, remain)
        keep_idx = np.random.randint(len(points) - keep_range)
        keep = np.append(keep, np.arange(keep_idx, keep_idx+keep_range, dtype=int), axis=0)
        keep = np.array(list(set(keep)), dtype=int)
        remain = num_points - len(keep)

    points = points[keep]
    
    return points

def discard_zorder(points, num_in_points=5000, num_out_points=8192):
    num_discard_points = num_out_points - num_in_points

    idx = np.random.randint(num_in_points)
    points = np.concatenate((points[:idx], points[idx+num_discard_points:]), axis=0)

    return points

def discard_multizorder(points, num_in_points=5000, num_out_points=8192):
    num_discard_points = num_out_points - num_in_points
    multizorder_range = multizorder_ranges[num_points]

    remain = num_discard_points
    discard = np.array([])
    while len(discard) < num_discard_points:
        discard_range = remain if remain <= multizorder_range else np.random.randint(multizorder_range, remain)
        discard_idx = np.random.randint(self.npoints - discard_range)
        discard = np.append(discard, np.arange(discard_idx, discard_idx + discard_range), axis=0)
        discard = np.array(list(set(discard)))
        remain = num_discard_points - len(discard)

    keep = np.arange(0, self.npoints)
    points= points[np.array(list(set(keep) - set(discard)))]

    return points
