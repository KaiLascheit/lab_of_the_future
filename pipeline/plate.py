"""96-well plate geometry: box from the two plate markers, ray-box intersection,
and which grid cell a 3D point falls in."""
import numpy as np

import settings as S


def plate_corners_cam(T_cam_top, T_cam_bottom):
    """8 plate box corners in the camera frame, from the poses of the top-right and
    bottom-right plate markers:
      [0:4]  top face    (top-right, bottom-right, bottom-left, top-left)
      [4:8]  bottom face (same order)"""
    p_t, p_b = T_cam_top[:3, 3], T_cam_bottom[:3, 3]
    v = p_b - p_t                       # along the right edge, top -> bottom
    v /= np.linalg.norm(v)
    n = T_cam_top[:3, 2] + T_cam_bottom[:3, 2]   # average marker normal (plate "up")
    if np.dot(n, (p_t + p_b) / 2) > 0:  # make it point towards the camera
        n = -n
    n -= np.dot(n, v) * v
    n /= np.linalg.norm(n)
    u = np.cross(v, n)                  # in the plate plane, pointing into the plate body

    a = p_t - S.PLATE_EDGE_EXTENSION_M * v
    b = p_b + S.PLATE_EDGE_EXTENSION_M * v
    right = -S.PLATE_RIGHT_MARGIN_M * u
    left = (S.PLATE_LENGTH_M - S.PLATE_RIGHT_MARGIN_M) * u
    top = [a + right, b + right, b + left, a + left]
    bottom = [c - S.PLATE_HEIGHT_M * n for c in top]
    return np.array(top + bottom)


def ray_box_intersection(origin, direction, corners, eps=1e-9):
    """Intersect a ray with the solid box from plate_corners_cam() (slab method).
    Returns (t_enter, t_exit) along the unit `direction` (t_enter clamped to 0 if
    the ray starts inside the box), or None if it misses."""
    c0 = corners[0]
    t_min, t_max = 0.0, np.inf
    for e in (corners[1] - c0, corners[3] - c0, corners[4] - c0):
        L = np.linalg.norm(e)
        if L < 1e-9:              # degenerate axis (PLATE_HEIGHT_M = 0): flat slab
            e_hat = np.cross(corners[1] - c0, corners[3] - c0)
            e_hat /= np.linalg.norm(e_hat)
        else:
            e_hat = e / L
        s0 = np.dot(origin - c0, e_hat)
        ds = np.dot(direction, e_hat)
        if abs(ds) < eps:
            if s0 < -eps or s0 > L + eps:
                return None
            continue
        t1, t2 = (0.0 - s0) / ds, (L - s0) / ds
        if t1 > t2:
            t1, t2 = t2, t1
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min, t_max


def grid_cell_at(p, plate_points):
    """(row, col) of the grid cell on the top face containing point p (projected
    onto the top face; points on the box sides map to the edge cells)."""
    g0, g1, _, g3 = plate_points[0:4]
    ev, eu = g1 - g0, g3 - g0
    a = np.clip(np.dot(p - g0, ev) / np.dot(ev, ev), 0.0, 1.0 - 1e-9)
    b = np.clip(np.dot(p - g0, eu) / np.dot(eu, eu), 0.0, 1.0 - 1e-9)
    return int(a * S.GRID_ROWS), int(b * S.GRID_COLS)


def grid_cell_quad(plate_points, row, col):
    """The 4 camera-frame corners of one grid cell on the top face."""
    g0, g1, _, g3 = plate_points[0:4]
    ev, eu = (g1 - g0) / S.GRID_ROWS, (g3 - g0) / S.GRID_COLS
    c00 = g0 + row * ev + col * eu
    return np.array([c00, c00 + ev, c00 + ev + eu, c00 + eu])


def well_name(row, col):
    """Grid cell -> well name such as "A1" (orientation set in settings)."""
    r = row if S.ROW_A_AT_TOP_MARKER else S.GRID_ROWS - 1 - row
    c = col if S.COLUMN_1_AT_MARKER_EDGE else S.GRID_COLS - 1 - col
    return f"{chr(ord('A') + r)}{c + 1}"
