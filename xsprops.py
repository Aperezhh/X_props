# xsprops.py
"""
xsprops - Cross-section geometry and properties library.
Tekla-compatible coordinate system.

Coordinate System (Tekla Convention):
    Origin (0, 0) = Centroid
    +U = Front face direction (right when looking Start -> End)
    +V = Top face direction (up)

Example:
    >>> from xsprops import i_beam, props
    >>> sec = i_beam(b=200, h=300, tw=8, tf=12)
    >>> p = props(sec)
    >>> print(f"Area: {p['A']:.0f} mm^2")

See README.md for usage and conventions.

Dependencies: numpy, shapely, matplotlib (optional)
"""
from typing import List, Tuple, Dict, Optional, Any, Sequence, TypedDict, cast, Mapping
import math
import numpy as np
from shapely.geometry import Polygon, Point, box, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from shapely.affinity import rotate, translate

__all__ = [
    "i_beam", "channel", "angle", "t_beam", "rect_tube", "circ_tube",
    "rectangle", "plate", "circle", "grid_circles", "series_holes",
    "props", "props_at_point", "pretty", "plot_section",
    "profile_from_dict", "apply_tekla_position", "apply_tekla_offsets_only",
    "create_bolt_group_geometry", "calculate_bolt_group_properties",
    "calculate_weld_polar_inertia"
]


class SectionProps(TypedDict):
    A: float
    Cx: float
    Cy: float
    Ix: float
    Iy: float
    Ixy: float
    Ip: float
    rx: float
    ry: float
    Sx_max: float
    Sy_max: float
    Wx_plus: float
    Wx_minus: float
    Wy_plus: float
    Wy_minus: float


class SectionPropsAtPoint(TypedDict):
    A: float
    Cx: float
    Cy: float
    Ix_0: float
    Iy_0: float
    Ixy_0: float
    Ip_0: float
    rx_0: float
    ry_0: float

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _arc_ccw(
    cx: float, cy: float, r: float,
    p_start: Tuple[float, float],
    p_end: Tuple[float, float],
    nseg: int
) -> List[Tuple[float, float]]:
    """Generate arc points counter-clockwise from p_start to p_end."""
    if r <= 0:
        return [p_end]
    a0 = math.atan2(p_start[1] - cy, p_start[0] - cx)
    a1 = math.atan2(p_end[1] - cy, p_end[0] - cx)
    if a0 < 0:
        a0 += 2 * math.pi
    if a1 < 0:
        a1 += 2 * math.pi
    if a1 < a0:
        a1 += 2 * math.pi
    angles = np.linspace(a0, a1, nseg + 1)[1:]
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


def _centroid(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute centroid of polygon defined by coordinates (shoelace formula)."""
    n = len(coords)
    if n < 3:
        return (0.0, 0.0)
    
    # Close polygon if needed
    if coords[0] != coords[-1]:
        coords = list(coords) + [coords[0]]
    
    area = 0.0
    cx = 0.0
    cy = 0.0
    
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    
    area *= 0.5
    if abs(area) < 1e-12:
        # Fallback: simple average
        xs = [c[0] for c in coords[:-1]]
        ys = [c[1] for c in coords[:-1]]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return (cx, cy)


def _center_to_centroid(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Shift coordinates so centroid is at (0, 0)."""
    cx, cy = _centroid(coords)
    return [(x - cx, y - cy) for (x, y) in coords]


def _collect_polygons(g: BaseGeometry) -> List[Polygon]:
    """Return a list of Polygon objects from a geometry."""
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if isinstance(g, GeometryCollection):
        return [p for p in g.geoms if isinstance(p, Polygon)]
    return []


# ============================================================
# PROFILE GENERATORS (Tekla coordinate system)
# ============================================================
# All profiles:
#   - Centroid at (0, 0)
#   - +U = Front (right), +V = Top (up)
#   - Return Shapely Polygon
# ============================================================

def i_beam(
    b: float,
    h: float, 
    tw: float,
    tf: float,
    r1: float = 0.0,
    nseg: int = 8
) -> Polygon:
    """
    Create I-beam / H-section with centroid at (0, 0).
    
    Orientation (Tekla convention):
        Top (+V)    = upper flange
        Below (-V)  = lower flange
        Front (+U)  = right flange edge
        Back (-U)   = left flange edge
    
    Args:
        b:  Flange width (mm)
        h:  Total height (mm)
        tw: Web thickness (mm)
        tf: Flange thickness (mm)
        r1: Fillet radius between web and flange (mm)
        nseg: Number of segments for fillet arcs
    
    Returns:
        Shapely Polygon with centroid at origin
    """
    if h <= 0 or b <= 0:
        raise ValueError(f"h and b must be positive: h={h}, b={b}")
    if tw <= 0 or tf <= 0:
        raise ValueError(f"tw and tf must be positive: tw={tw}, tf={tf}")
    if tw >= b:
        raise ValueError(f"tw must be less than b: tw={tw}, b={b}")
    if 2 * tf >= h:
        raise ValueError(f"2*tf must be less than h: tf={tf}, h={h}")
    max_r1 = min((h - 2 * tf) / 2.0, (b - tw) / 2.0)
    if r1 > max_r1 + 1e-6:
        raise ValueError(f"r1 too large: r1={r1}, max={max_r1:.1f}")

    # Half dimensions
    hb = b / 2.0
    htw = tw / 2.0
    
    # Key Y coordinates (building from top)
    y_top = h / 2.0
    y_flange_inner = y_top - tf
    y_fillet_end = y_flange_inner - r1
    y_bot_fillet_start = -y_fillet_end
    y_bot_flange_inner = -y_flange_inner
    y_bot = -y_top
    
    # X coordinates
    x_web = htw
    x_fillet = htw + r1
    x_flange = hb
    
    # Build contour CCW starting from top-left corner
    coords = []
    
    # Top flange - left side
    coords.append((-hb, y_top))
    
    # Top flange - right side  
    coords.append((hb, y_top))
    coords.append((hb, y_flange_inner))
    
    # Right top fillet
    if r1 > 0:
        p0 = (x_fillet, y_flange_inner)
        p1 = (x_web, y_fillet_end)
        coords.append(p0)
        coords.extend(_arc_ccw(x_fillet, y_fillet_end, r1, p0, p1, nseg))
    else:
        coords.append((x_web, y_flange_inner))
    
    # Right web
    coords.append((x_web, y_bot_fillet_start))
    
    # Right bottom fillet
    if r1 > 0:
        p0 = (x_web, y_bot_fillet_start)
        p1 = (x_fillet, y_bot_flange_inner)
        coords.extend(_arc_ccw(x_fillet, y_bot_fillet_start, r1, p0, p1, nseg))
    
    # Bottom flange - right side
    coords.append((hb, y_bot_flange_inner))
    coords.append((hb, y_bot))
    
    # Bottom flange - left side
    coords.append((-hb, y_bot))
    coords.append((-hb, y_bot_flange_inner))
    
    # Left bottom fillet
    if r1 > 0:
        p0 = (-x_fillet, y_bot_flange_inner)
        p1 = (-x_web, y_bot_fillet_start)
        coords.append(p0)
        coords.extend(_arc_ccw(-x_fillet, y_bot_fillet_start, r1, p0, p1, nseg))
    else:
        coords.append((-x_web, y_bot_flange_inner))
    
    # Left web
    coords.append((-x_web, y_fillet_end))
    
    # Left top fillet
    if r1 > 0:
        p0 = (-x_web, y_fillet_end)
        p1 = (-x_fillet, y_flange_inner)
        coords.extend(_arc_ccw(-x_fillet, y_fillet_end, r1, p0, p1, nseg))
    
    # Back to top flange
    coords.append((-hb, y_flange_inner))
    
    # I-beam is doubly symmetric, centroid is at geometric center
    # Already built symmetric about (0, 0)
    
    return Polygon(coords)


def channel(
    h: float,
    b: float,
    tw: float,
    tf: float,
    r1: float = 0.0,
    nseg: int = 8
) -> Polygon:
    """
    Create channel (C/U section) with centroid at (0, 0).
    
    Orientation (Tekla convention):
        Top (+V)    = upper flange
        Below (-V)  = lower flange
        Front (+U)  = toes (open side)
        Back (-U)   = web (closed side)
    
    Args:
        h:  Total height (mm)
        b:  Flange width (mm)  
        tw: Web thickness (mm)
        tf: Flange thickness (mm)
        r1: Inner fillet radius (mm)
        nseg: Number of segments for fillet arcs
    
    Returns:
        Shapely Polygon with centroid at origin
    """
    # Build channel with web on left side, toes on right
    # Then shift to centroid
    
    # Y coordinates
    y_top = h / 2.0
    y_flange_inner_top = y_top - tf
    y_fillet_top = y_flange_inner_top - r1
    y_fillet_bot = -y_fillet_top
    y_flange_inner_bot = -y_flange_inner_top
    y_bot = -y_top
    
    # X coordinates (web on left at x=0, toes extend to right)
    x_web_outer = -tw / 2.0
    x_web_inner = tw / 2.0
    x_fillet = x_web_inner + r1
    x_toe = -tw / 2.0 + b  # toe edge
    
    coords = []
    
    # Start at top-left (web outer, top)
    coords.append((x_web_outer, y_top))
    
    # Top flange outer edge
    coords.append((x_toe, y_top))
    coords.append((x_toe, y_flange_inner_top))
    
    # Top inner fillet
    if r1 > 0:
        p0 = (x_fillet, y_flange_inner_top)
        p1 = (x_web_inner, y_fillet_top)
        coords.append(p0)
        coords.extend(_arc_ccw(x_fillet, y_fillet_top, r1, p0, p1, nseg))
    else:
        coords.append((x_web_inner, y_flange_inner_top))
    
    # Web inner surface
    coords.append((x_web_inner, y_fillet_bot))
    
    # Bottom inner fillet
    if r1 > 0:
        p0 = (x_web_inner, y_fillet_bot)
        p1 = (x_fillet, y_flange_inner_bot)
        coords.extend(_arc_ccw(x_fillet, y_fillet_bot, r1, p0, p1, nseg))
    
    # Bottom flange inner
    coords.append((x_toe, y_flange_inner_bot))
    coords.append((x_toe, y_bot))
    
    # Bottom to web outer
    coords.append((x_web_outer, y_bot))
    
    # Compute centroid and shift
    coords = _center_to_centroid(coords)
    
    return Polygon(coords)


def angle(
    h: float,
    b: float,
    t: float,
    r1: float = 0.0,
    nseg: int = 8
) -> Polygon:
    """
    Create angle (L-section) with centroid at (0, 0).
    
    Orientation (Tekla convention, for h >= b):
        Top (+V)    = end of vertical (long) leg
        Below (-V)  = corner region
        Front (+U)  = end of horizontal (short) leg
        Back (-U)   = corner region
    
    For equal leg angle (h == b), orientation is symmetric.
    
    Args:
        h:  Vertical leg length (mm) - typically longer leg
        b:  Horizontal leg length (mm) - typically shorter leg
        t:  Leg thickness (mm)
        r1: Inner fillet radius at corner (mm)
        nseg: Number of segments for fillet arc
    
    Returns:
        Shapely Polygon with centroid at origin
    """
    # Build angle with corner at origin, then shift to centroid
    # Vertical leg goes up (+Y), horizontal leg goes right (+X)
    
    coords = []
    
    # Start at outer corner (0, 0) going CCW
    # Outer vertical leg
    coords.append((0, 0))
    coords.append((0, h))
    coords.append((t, h))
    
    # Inner corner with optional fillet
    if r1 > 0:
        # Fillet center
        fc_x = t + r1
        fc_y = t + r1
        # Arc from vertical inner to horizontal inner
        p0 = (t, t + r1)
        p1 = (t + r1, t)
        coords.append(p0)
        coords.extend(_arc_ccw(fc_x, fc_y, r1, p0, p1, nseg))
    else:
        coords.append((t, t))
    
    # Horizontal leg inner
    coords.append((b, t))
    
    # Horizontal leg outer
    coords.append((b, 0))
    
    # Compute centroid and shift
    coords = _center_to_centroid(coords)
    
    return Polygon(coords)


def t_beam(
    b: float,
    h: float,
    tw: float,
    tf: float,
    r1: float = 0.0,
    nseg: int = 8
) -> Polygon:
    """
    Create T-section with centroid at (0, 0).
    
    Orientation (Tekla convention):
        Top (+V)    = flange top surface
        Below (-V)  = stem end
        Front (+U)  = right side
        Back (-U)   = left side
    
    Args:
        b:  Flange width (mm)
        h:  Total height (mm)
        tw: Stem (web) thickness (mm)
        tf: Flange thickness (mm)
        r1: Fillet radius between flange and stem (mm)
        nseg: Number of segments for fillet arcs
    
    Returns:
        Shapely Polygon with centroid at origin
    """
    # Half dimensions
    hb = b / 2.0
    htw = tw / 2.0
    
    # Build T with flange at top, stem going down
    # Reference: flange top at y = 0
    y_flange_top = 0
    y_flange_bot = -tf
    y_fillet_end = y_flange_bot - r1
    y_stem_bot = -h
    
    x_fillet = htw + r1
    
    coords = []
    
    # Top left of flange
    coords.append((-hb, y_flange_top))
    
    # Top right of flange
    coords.append((hb, y_flange_top))
    coords.append((hb, y_flange_bot))
    
    # Right fillet
    if r1 > 0:
        p0 = (x_fillet, y_flange_bot)
        p1 = (htw, y_fillet_end)
        coords.append(p0)
        coords.extend(_arc_ccw(x_fillet, y_fillet_end, r1, p0, p1, nseg))
    else:
        coords.append((htw, y_flange_bot))
    
    # Right side of stem
    coords.append((htw, y_stem_bot))
    
    # Bottom of stem
    coords.append((-htw, y_stem_bot))
    
    # Left side of stem
    coords.append((-htw, y_fillet_end))
    
    # Left fillet
    if r1 > 0:
        p0 = (-htw, y_fillet_end)
        p1 = (-x_fillet, y_flange_bot)
        coords.extend(_arc_ccw(-x_fillet, y_fillet_end, r1, p0, p1, nseg))
    
    # Back to flange
    coords.append((-hb, y_flange_bot))
    
    # Shift to centroid (T is NOT symmetric about horizontal axis)
    coords = _center_to_centroid(coords)
    
    return Polygon(coords)


def rect_tube(
    b: float,
    h: float,
    t: float,
) -> BaseGeometry:
    """
    Create rectangular hollow section (RHS) with centroid at (0, 0).
    
    Orientation (Tekla convention):
        Top (+V)    = top wall
        Below (-V)  = bottom wall
        Front (+U)  = right wall
        Back (-U)   = left wall
    
    Args:
        b: Width (mm) - horizontal
        h: Height (mm) - vertical
        t: Wall thickness (mm)
    
    Returns:
        Shapely Polygon with centroid at origin (doubly symmetric)
    """
    hb = b / 2.0
    hh = h / 2.0
    
    # Outer rectangle
    outer = [(-hb, -hh), (hb, -hh), (hb, hh), (-hb, hh)]
    
    # Inner rectangle
    inner_hb = hb - t
    inner_hh = hh - t
    
    if inner_hb <= 0 or inner_hh <= 0:
        # Solid rectangle if wall too thick
        return Polygon(outer)
    
    inner = [(-inner_hb, -inner_hh), (inner_hb, -inner_hh),
             (inner_hb, inner_hh), (-inner_hb, inner_hh)]
    
    outer_poly = Polygon(outer)
    inner_poly = Polygon(inner)
    result = outer_poly.difference(inner_poly)
    
    return result


def circ_tube(
    d: float,
    t: float,
    nseg: int = 64,
) -> BaseGeometry:
    """
    Create circular hollow section (CHS) with centroid at (0, 0).
    
    Orientation (Tekla convention):
        Circular section is symmetric, but conventionally:
        Top (+V)    = top of circle
        Front (+U)  = right side of circle
    
    Args:
        d:    Outer diameter (mm)
        t:    Wall thickness (mm)
        nseg: Number of segments for circle approximation
    
    Returns:
        Shapely Polygon with centroid at origin
    """
    r_outer = d / 2.0
    r_inner = r_outer - t
    
    outer = Point(0, 0).buffer(r_outer, resolution=nseg)
    
    if r_inner <= 0:
        return outer
    
    inner = Point(0, 0).buffer(r_inner, resolution=nseg)
    result = outer.difference(inner)
    
    return result


# ============================================================
# BASIC PRIMITIVES
# ============================================================

def rectangle(
    b: float,
    h: float,
    x0: float = 0,
    y0: float = 0
) -> Polygon:
    """
    Create rectangle with bottom-left corner at (x0, y0).
    
    Args:
        b:  Width (along X)
        h:  Height (along Y)
        x0: X coordinate of bottom-left corner
        y0: Y coordinate of bottom-left corner
    
    Returns:
        Shapely Polygon
    """
    coords = [(x0, y0), (x0 + b, y0), (x0 + b, y0 + h), (x0, y0 + h)]
    return Polygon(coords)


def plate(
    width: float,
    height: float,
    *,
    origin: str = "centroid"
) -> Polygon:
    """
    Create plate geometry.
    
    Args:
        width:  Plate width
        height: Plate height
        origin: "centroid" (default), "top", or "bottom"
    
    Returns:
        Shapely Polygon
    """
    origin_key = str(origin).lower()
    if origin_key in ("centroid", "center", "centre", "mid", "middle"):
        x0 = -width / 2.0
        y0 = -height / 2.0
    elif origin_key in ("top", "upper"):
        x0 = -width / 2.0
        y0 = -height
    elif origin_key in ("bottom", "lower"):
        x0 = -width / 2.0
        y0 = 0.0
    else:
        raise ValueError(f"Unsupported origin: {origin}")

    return rectangle(width, height, x0=x0, y0=y0)


def circle(
    diam: float,
    x0: float = 0,
    y0: float = 0,
    nseg: int = 64
) -> Polygon:
    """
    Create circle (approximated as polygon).
    
    Args:
        diam: Diameter
        x0:   X coordinate of center
        y0:   Y coordinate of center
        nseg: Number of segments
    
    Returns:
        Shapely Polygon
    """
    p = Point(x0, y0).buffer(diam / 2.0, resolution=nseg)
    return p


def grid_circles(
    diam: float,
    e1: float,
    p1: float,
    n1: int,
    e2: float,
    d1: float,
    n2: int
) -> Optional[BaseGeometry]:
    """
    Create grid of circular holes (for bolt patterns).
    
    Args:
        diam: Hole diameter
        e1:   Edge distance to first row (Y direction)
        p1:   Pitch between rows (Y direction)
        n1:   Number of rows - 1 (0 = single row)
        e2:   Edge distance to first column (X direction)
        d1:   Pitch between columns (X direction)
        n2:   Number of columns - 1 (0 = single column)
    
    Returns:
        Union of all holes as Polygon, or None if empty
    """
    holes = []
    for i in range(n1 + 1):
        yc = -(e1 + i * p1)
        for j in range(n2 + 1):
            xc = e2 + j * d1
            holes.append(circle(diam, x0=xc, y0=yc))
    
    if not holes:
        return None
    return unary_union(holes)


def series_holes(
    plate_width: float,
    hole_height: float,
    e1: float,
    p1: float,
    n1: int
) -> Optional[BaseGeometry]:
    """
    Create series of rectangular slots (for block shear patterns).
    
    Args:
        plate_width: Width of each slot
        hole_height: Height of each slot
        e1: Edge distance to first slot
        p1: Pitch between slots
        n1: Number of slots - 1
    
    Returns:
        Union of all slots as Polygon, or None if empty
    """
    holes = []
    for i in range(n1 + 1):
        y_top = -(e1 + i * p1)
        hole = rectangle(plate_width, hole_height, x0=-plate_width / 2.0, y0=y_top - hole_height)
        holes.append(hole)
    
    if not holes:
        return None
    return unary_union(holes)


# ============================================================
# SECTION PROPERTIES CALCULATOR
# ============================================================
# These functions compute properties using Green's theorem.
# DO NOT MODIFY - proven correct against standard tables.
# ============================================================

def _poly_int(poly: Polygon) -> Tuple[float, float, float, float, float, float]:
    """
    Compute integral properties of single polygon using Green's theorem.
    
    Returns:
        (A, Cx, Cy, Ix_c, Iy_c, Ixy_c)
        Where Ix_c, Iy_c, Ixy_c are about the polygon's own centroid.
    """
    if poly.is_empty or abs(poly.area) < 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    poly = orient(poly, 1.0)  # CCW exterior; holes become CW

    def _ring_int(
        coords: Sequence[Sequence[float]]
    ) -> Tuple[float, float, float, float, float, float]:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        x = arr[:, 0]
        y = arr[:, 1]
        x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]

        a = x1 * y2 - x2 * y1  # signed area elements
        A = 0.5 * np.sum(a)
        Cx_num = np.sum((x1 + x2) * a)
        Cy_num = np.sum((y1 + y2) * a)
        Ix_0 = np.sum((y1**2 + y1 * y2 + y2**2) * a) / 12.0
        Iy_0 = np.sum((x1**2 + x1 * x2 + x2**2) * a) / 12.0
        Ixy_0 = np.sum((x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1) * a) / 24.0
        return A, Cx_num, Cy_num, Ix_0, Iy_0, Ixy_0

    A_total = 0.0
    Cx_num = 0.0
    Cy_num = 0.0
    Ix_0 = 0.0
    Iy_0 = 0.0
    Ixy_0 = 0.0

    A_i, Cx_i, Cy_i, Ix_i, Iy_i, Ixy_i = _ring_int(list(poly.exterior.coords))
    A_total += A_i
    Cx_num += Cx_i
    Cy_num += Cy_i
    Ix_0 += Ix_i
    Iy_0 += Iy_i
    Ixy_0 += Ixy_i

    for ring in poly.interiors:
        A_i, Cx_i, Cy_i, Ix_i, Iy_i, Ixy_i = _ring_int(list(ring.coords))
        A_total += A_i
        Cx_num += Cx_i
        Cy_num += Cy_i
        Ix_0 += Ix_i
        Iy_0 += Iy_i
        Ixy_0 += Ixy_i

    if abs(A_total) < 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if A_total < 0:
        A_total = -A_total
        Cx_num = -Cx_num
        Cy_num = -Cy_num
        Ix_0 = -Ix_0
        Iy_0 = -Iy_0
        Ixy_0 = -Ixy_0

    Cx0 = Cx_num / (6.0 * A_total)
    Cy0 = Cy_num / (6.0 * A_total)

    # Transfer to centroid (parallel axis theorem)
    Ix_c = Ix_0 - A_total * Cy0**2
    Iy_c = Iy_0 - A_total * Cx0**2
    Ixy_c = Ixy_0 - A_total * Cx0 * Cy0

    return A_total, Cx0, Cy0, Ix_c, Iy_c, Ixy_c


def _get_area_first_moments(g: BaseGeometry) -> Tuple[float, float, float]:
    """
    Compute area and first moments (A*Cx, A*Cy) for compound geometry.
    Used for static moment calculations.
    """
    polys = _collect_polygons(g)
    if not polys:
        return 0.0, 0.0, 0.0
    
    total_A = 0.0
    moment_Ax0 = 0.0
    moment_Ay0 = 0.0
    
    for p in polys:
        Ai, Cxi, Cyi, _, _, _ = _poly_int(p)
        if Ai > 1e-9:
            total_A += Ai
            moment_Ax0 += Ai * Cxi
            moment_Ay0 += Ai * Cyi
    
    return total_A, moment_Ax0, moment_Ay0


def props(g: BaseGeometry) -> SectionProps:
    """
    Compute geometric properties of cross-section.
    
    Args:
        g: Shapely geometry (Polygon or MultiPolygon)
    
    Returns:
        Dictionary with:
            A:        Area (mm^2)
            Cx, Cy:   Centroid coordinates (mm)
            Ix, Iy:   Second moments of area about centroidal axes (mm^4)
            Ixy:      Product of inertia (mm^4)
            Sx_max:   First moment of area (half-section) about X axis (mm^3)
            Sy_max:   First moment of area (half-section) about Y axis (mm^3)
            Wx_plus:  Section modulus, top fiber (mm^3)
            Wx_minus: Section modulus, bottom fiber (mm^3)
            Wy_plus:  Section modulus, right fiber (mm^3)
            Wy_minus: Section modulus, left fiber (mm^3)
    
    Raises:
        ValueError: If geometry is empty or has zero area
        TypeError: If geometry type is not supported
    """
    if g is None or g.is_empty:
        raise ValueError("Cannot calculate properties for empty geometry.")
    
    polys = _collect_polygons(g)
    if not polys:
        raise ValueError("Geometry contains no Polygon objects.")
    
    # Stage 1: Global centroid and area
    total_A = 0.0
    moment_Ax = 0.0
    moment_Ay = 0.0
    poly_props_list = []
    
    for p in polys:
        Ai, Cxi, Cyi, Ixi_c, Iyi_c, Ixyi_c = _poly_int(p)
        if Ai > 1e-9:
            total_A += Ai
            moment_Ax += Ai * Cxi
            moment_Ay += Ai * Cyi
            poly_props_list.append({
                'A': Ai, 'Cx': Cxi, 'Cy': Cyi,
                'Ix_c': Ixi_c, 'Iy_c': Iyi_c, 'Ixy_c': Ixyi_c
            })
    
    if total_A < 1e-9:
        raise ValueError("Total area is close to zero.")
    
    global_Cx = moment_Ax / total_A
    global_Cy = moment_Ay / total_A
    
    # Stage 2: Moments of inertia about global centroid
    total_Ix = 0.0
    total_Iy = 0.0
    total_Ixy = 0.0
    
    for props_i in poly_props_list:
        dist_x = props_i['Cx'] - global_Cx
        dist_y = props_i['Cy'] - global_Cy
        # Parallel axis theorem
        total_Ix += props_i['Ix_c'] + props_i['A'] * dist_y**2
        total_Iy += props_i['Iy_c'] + props_i['A'] * dist_x**2
        total_Ixy += props_i['Ixy_c'] + props_i['A'] * dist_x * dist_y
    
    rx = math.sqrt(total_Ix / total_A) if total_A > 1e-9 else 0.0
    ry = math.sqrt(total_Iy / total_A) if total_A > 1e-9 else 0.0
    Ip = total_Ix + total_Iy

    # Stage 3: Static moments (first moment of half-section)
    Sx_max = 0.0
    Sy_max = 0.0
    
    try:
        minx_g, miny_g, maxx_g, maxy_g = g.bounds
        pad = max(abs(maxx_g - minx_g), abs(maxy_g - miny_g)) * 0.1 + 1.0
        
        # Sx_max: area above centroidal X axis
        clipper_top = box(minx_g - pad, global_Cy, maxx_g + pad, maxy_g + pad)
        g_top = g.intersection(clipper_top)
        if not g_top.is_empty:
            A_top, _, moment_Ay0_top = _get_area_first_moments(g_top)
            if A_top > 1e-9:
                Sx_max = abs(moment_Ay0_top - A_top * global_Cy)
        
        # Sy_max: area right of centroidal Y axis
        clipper_right = box(global_Cx, miny_g - pad, maxx_g + pad, maxy_g + pad)
        g_right = g.intersection(clipper_right)
        if not g_right.is_empty:
            A_right, moment_Ax0_right, _ = _get_area_first_moments(g_right)
            if A_right > 1e-9:
                Sy_max = abs(moment_Ax0_right - A_right * global_Cx)
    except Exception as e:
        print(f"Warning: Could not calculate Sx_max/Sy_max. Error: {e}")
        Sx_max = float('nan')
        Sy_max = float('nan')
    
    # Stage 4: Section moduli
    all_xs, all_ys = [], []
    work_geom = g if isinstance(g, Polygon) else unary_union(g)
    for p in _collect_polygons(work_geom):
        xy_ext = np.asarray(p.exterior.coords)
        all_xs.extend(xy_ext[:, 0])
        all_ys.extend(xy_ext[:, 1])
        for interior in p.interiors:
            xy_int = np.asarray(interior.coords)
            all_xs.extend(xy_int[:, 0])
            all_ys.extend(xy_int[:, 1])
    
    if not all_xs:
        Wx_plus = Wx_minus = Wy_plus = Wy_minus = float('nan')
    else:
        xs_arr = np.asarray(all_xs)
        ys_arr = np.asarray(all_ys)
        
        y_dist_plus = abs(ys_arr.max() - global_Cy)
        y_dist_minus = abs(ys_arr.min() - global_Cy)
        x_dist_plus = abs(xs_arr.max() - global_Cx)
        x_dist_minus = abs(xs_arr.min() - global_Cx)
        
        Wx_plus = total_Ix / y_dist_plus if y_dist_plus > 1e-9 else float('inf')
        Wx_minus = total_Ix / y_dist_minus if y_dist_minus > 1e-9 else float('inf')
        Wy_plus = total_Iy / x_dist_plus if x_dist_plus > 1e-9 else float('inf')
        Wy_minus = total_Iy / x_dist_minus if x_dist_minus > 1e-9 else float('inf')
    
    return cast(SectionProps, dict(
        A=total_A,
        Cx=global_Cx,
        Cy=global_Cy,
        Ix=total_Ix,
        Iy=total_Iy,
        Ixy=total_Ixy,
        Ip=Ip,
        Sx_max=Sx_max,
        Sy_max=Sy_max,
        Wx_plus=Wx_plus,
        Wx_minus=Wx_minus,
        Wy_plus=Wy_plus,
        Wy_minus=Wy_minus,
        rx=rx,
        ry=ry
    ))


def props_at_point(
    g: BaseGeometry,
    x0: float = 0.0,
    y0: float = 0.0
) -> SectionPropsAtPoint:
    """
    Compute section properties about arbitrary point (x0, y0).

    Args:
        g:  Shapely geometry (Polygon or MultiPolygon)
        x0: X coordinate of reference point (mm)
        y0: Y coordinate of reference point (mm)

    Returns:
        Dictionary with:
            A:     Area (mm^2)
            Cx, Cy: Centroid coordinates (mm)
            Ix_0:  Second moment about horizontal axis through (x0, y0)
            Iy_0:  Second moment about vertical axis through (x0, y0)
            Ixy_0: Product of inertia about (x0, y0)
            Ip_0:  Polar second moment about (x0, y0)
            rx_0:  Radius of gyration about X axis through (x0, y0)
            ry_0:  Radius of gyration about Y axis through (x0, y0)
    """
    p = props(g)
    dx = p["Cx"] - x0
    dy = p["Cy"] - y0

    Ix_0 = p["Ix"] + p["A"] * dy**2
    Iy_0 = p["Iy"] + p["A"] * dx**2
    Ixy_0 = p["Ixy"] + p["A"] * dx * dy
    Ip_0 = Ix_0 + Iy_0
    if p["A"] > 1e-9:
        rx_0 = math.sqrt(Ix_0 / p["A"])
        ry_0 = math.sqrt(Iy_0 / p["A"])
    else:
        rx_0 = 0.0
        ry_0 = 0.0

    return cast(SectionPropsAtPoint, dict(
        A=p["A"],
        Cx=p["Cx"],
        Cy=p["Cy"],
        Ix_0=Ix_0,
        Iy_0=Iy_0,
        Ixy_0=Ixy_0,
        Ip_0=Ip_0,
        rx_0=rx_0,
        ry_0=ry_0
    ))


# ============================================================
# BOLT/WELD UTILITIES
# ============================================================

def create_bolt_group_geometry(
    row_dist: float,
    num_rows: int,
    bolt_pitch_in_row: float,
    num_bolts_per_row: int,
    num_bolts_last_row: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Generate bolt coordinates for rectangular bolt group.
    
    Origin (0, 0) = bottom-left bolt.
    Rows extend in +Y, columns extend in +X.
    
    Args:
        row_dist:           Distance between rows (Y direction)
        num_rows:           Number of rows (>= 1)
        bolt_pitch_in_row:  Distance between bolts in row (X direction)
        num_bolts_per_row:  Number of bolts per row
        num_bolts_last_row: Number of bolts in top row (if different)
    
    Returns:
        List of (x, y) tuples for each bolt
    """
    if num_rows < 1:
        raise ValueError("num_rows must be >= 1")
    if num_bolts_per_row < 1:
        raise ValueError("num_bolts_per_row must be >= 1")
    if num_rows > 1 and row_dist <= 0:
        raise ValueError("row_dist must be > 0 for multiple rows")
    if num_bolts_per_row > 1 and bolt_pitch_in_row <= 0:
        raise ValueError("bolt_pitch_in_row must be > 0 for multiple bolts")
    
    if num_bolts_last_row is None:
        num_bolts_last_row = num_bolts_per_row
    
    coords = []
    y = 0.0
    
    for i_row in range(num_rows):
        is_last = (i_row == num_rows - 1)
        n_bolts = num_bolts_last_row if is_last else num_bolts_per_row
        
        x = 0.0
        for _ in range(n_bolts):
            coords.append((x, y))
            x += bolt_pitch_in_row
        
        y += row_dist
    
    return coords


def calculate_bolt_group_properties(
    bolt_coords: List[Tuple[float, float]],
    bolt_area: float = 1.0
) -> Dict[str, Any]:
    """
    Compute properties of bolt group.
    
    Args:
        bolt_coords: List of (x, y) bolt positions
        bolt_area:   Area of single bolt (used as weight)
    
    Returns:
        Dictionary with:
            N:     Number of bolts
            Cx:    Centroid X coordinate
            Cy:    Centroid Y coordinate
            Ix:    Moment of inertia about centroidal X axis
            Iy:    Moment of inertia about centroidal Y axis
            Ixy:   Product of inertia
            Ip:    Polar moment of inertia (Ix + Iy)
            coords: Copy of input coordinates
    """
    if not bolt_coords:
        return {'N': 0, 'Cx': 0, 'Cy': 0, 'Ix': 0, 'Iy': 0, 'Ixy': 0, 'Ip': 0, 'coords': []}
    
    if bolt_area <= 0:
        raise ValueError("bolt_area must be > 0")
    
    coords = np.array(bolt_coords)
    N = len(coords)
    
    Cx = np.mean(coords[:, 0])
    Cy = np.mean(coords[:, 1])
    
    dx = coords[:, 0] - Cx
    dy = coords[:, 1] - Cy
    
    Ix = bolt_area * np.sum(dy**2)
    Iy = bolt_area * np.sum(dx**2)
    Ixy = bolt_area * np.sum(dx * dy)
    Ip = Ix + Iy
    
    return dict(
        N=N,
        Cx=Cx,
        Cy=Cy,
        Ix=Ix,
        Iy=Iy,
        Ixy=Ixy,
        Ip=Ip,
        coords=bolt_coords
    )


def calculate_weld_polar_inertia(
    length_a: float,
    length_b: float,
    t_w: float,
    a_side: bool = True,
    b_side: bool = True
) -> float:
    """
    Compute polar moment of inertia for rectangular weld group.
    
    Treats welds as lines (throat area = length * t_w).
    
    Args:
        length_a: Length of horizontal welds (top/bottom)
        length_b: Length of vertical welds (left/right)
        t_w:      Effective throat thickness (typically 0.7 * kf)
        a_side:   Include horizontal welds
        b_side:   Include vertical welds
    
    Returns:
        Polar moment of inertia Ip (mm^4)
    """
    Ix = 0.0
    Iy = 0.0
    
    # Horizontal welds (top and bottom)
    if a_side and length_a > 0 and t_w > 0:
        A_a = length_a * t_w
        Ix += 2 * A_a * (length_b / 2)**2
        Iy += 2 * (t_w * length_a**3 / 12)
    
    # Vertical welds (left and right)
    if b_side and length_b > 0 and t_w > 0:
        A_b = length_b * t_w
        Ix += 2 * (t_w * length_b**3 / 12)
        Iy += 2 * A_b * (length_a / 2)**2
    
    return Ix + Iy


# ============================================================
# OUTPUT UTILITIES
# ============================================================

def pretty(d: Mapping[str, Any], n: int = 1) -> str:
    """
    Format props() result for printing.
    
    Args:
        d: Dictionary from props()
        n: Number of decimal places
    
    Returns:
        Formatted string
    """
    float_format = f',.{n}f'
    lines = []
    
    key_order = ['A', 'Cx', 'Cy', 'Ix', 'Iy', 'Ip', 'Ixy', 'rx', 'ry', 'Sx_max', 'Sy_max',
                 'Wx_plus', 'Wx_minus', 'Wy_plus', 'Wy_minus']

    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating))
    
    for k in key_order:
        if k in d and _is_number(d[k]):
            lines.append(f"{k:<9}= {format(float(d[k]), float_format)}")
    
    for k, v in d.items():
        if k not in key_order and _is_number(v):
            lines.append(f"{k:<9}= {format(float(v), float_format)}")
    
    return '\n'.join(lines)


def plot_section(
    g: BaseGeometry,
    ax=None,
    *,
    face: str = "lightblue",
    edge: str = "k",
    linewidth: float = 1.0,
    show_axes: bool = False
):
    """
    Plot cross-section geometry using matplotlib.
    
    Args:
        g:         Shapely geometry to plot
        ax:        Matplotlib axes (created if None)
        face:      Fill color (default: 'lightblue')
        edge:      Edge color (default: 'k')
        linewidth: Edge line width (default: 1.0)
        show_axes: Show coordinate axes and grid
    
    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt
    
    create_new = ax is None
    if create_new:
        _, ax = plt.subplots()
    
    if g is None or g.is_empty:
        print("Warning: plot_section called with empty geometry")
        if create_new:
            ax.axis('off')
        return ax
    
    polys = _collect_polygons(g)
    
    for p in polys:
        ax.fill(*p.exterior.xy, color=face, edgecolor=edge, linewidth=linewidth)
        for ring in p.interiors:
            ax.fill(*ring.xy, color="white", edgecolor=edge, linewidth=linewidth)
    
    ax.set_aspect('equal')
    
    if create_new:
        if show_axes:
            ax.grid(True, linestyle=':', linewidth=0.4)
            ax.axhline(y=0, color='gray', linewidth=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.5)
            ax.set_xlabel('U (mm)')
            ax.set_ylabel('V (mm)')
        else:
            ax.axis('off')
    
    return ax


# ============================================================
# CONVERTERS
# ============================================================

# Canonical profile type mapping.
# Keys are normalized (lowercase, spaces/hyphens -> underscores) strings from:
# - Tekla snapshot: refs[*].profile_type (PROFILE_I / PROFILE_P / PROFILE_PD / ...)
# - Legacy DB cache: entry["Type"] (Tube / I-beam / ...)
# Values are internal builder names used by profile_from_dict().
_PROFILE_TYPE_MAP: dict[str, str] = {
    # Tekla snapshot profile_type values (authoritative)
    "profile_i": "i_beam",
    "profile_u": "channel",
    "profile_t": "t_beam",
    "profile_fpl": "angle",
    "profile_p": "rect_tube",
    "profile_pd": "circ_tube",

    # DB Type field values (legacy compatibility)
    "i_beam": "i_beam",
    "ibeam": "i_beam",
    "channel": "channel",
    "u_section": "channel",
    "t_beam": "t_beam",
    "t_section": "t_beam",
    "angle": "angle",
    "l_section": "angle",
    "tube": "rect_tube",
    "rect_tube": "rect_tube",
    "rhs": "rect_tube",
    "shs": "rect_tube",
    "round_tube": "circ_tube",
    "roundtube": "circ_tube",
    "circ_tube": "circ_tube",
    "chs": "circ_tube",
}


def profile_from_dict(
    data: Dict[str, Any],
    *,
    nseg: int = 8
) -> BaseGeometry:
    """
    Dict -> Shapely section geometry (centroid ~ (0, 0)).

    Supports:
      - DB/cache profile dictionaries (usually key/section + dimensions)
      - Tekla snapshot references (profile_type + profile_params[])
      - manual dictionaries (Type/profile_type + H/B + thicknesses)

    Important:
      - Tekla snapshot may contain "type": "Beam" (this is not the section type).
        For section reconstruction use "profile_type" (PROFILE_I / PROFILE_P / PROFILE_PD / ...),
        or project-level "Type" ("Tube", "Round_tube", ...).
    """
    if not isinstance(data, dict):
        raise TypeError("profile_from_dict expects dict")

    # ---- flatten Tekla profile_params[] to params dict ----
    params: Dict[str, Any] = {}
    pp = data.get("profile_params")
    if isinstance(pp, list):
        for it in pp:
            if not isinstance(it, dict):
                continue
            k = it.get("property")
            if k is None:
                continue
            k = str(k).strip().upper()
            if not k:
                continue
            v = it.get("value")
            if v is None:
                continue
            params[k] = v

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            v = data.get(k)
            if v is None:
                v = params.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return float(default)

    # ---- resolve profile type ----
    raw_type = data.get("profile_type") or data.get("Type") or ""
    raw_type = str(raw_type).strip().lower().replace("-", "_").replace(" ", "_")

    p = _PROFILE_TYPE_MAP.get(raw_type)

    if not p:
        key = data.get("key") or data.get("profile") or data.get("section") or "?"
        raise ValueError(
            f"Unknown profile type: {raw_type!r} (key={key!r}). "
            f"Expected one of: {sorted(set(_PROFILE_TYPE_MAP.keys()))}"
        )

    # ---- common dims ----
    h = _get(["H", "h", "HEIGHT", "d"])
    b = _get(["B", "b", "WIDTH"], h)

    # thicknesses
    tw = _get(["tw", "WEB_THICKNESS", "s"], 0.0)
    tf = _get(["tf", "FLANGE_THICKNESS", "FLANGE_THICKNESS_1"], 0.0)
    t_plate = _get(["t", "PLATE_THICKNESS"], 0.0)

    # radii
    r1 = _get(["r1", "ROUNDING_RADIUS_1", "ROUNDING_RADIUS", "r"], _get(["r2", "ROUNDING_RADIUS_2"], 0.0))

    # round tube diameter
    d_circle = _get(["D", "DIAMETER", "diameter"], 0.0)

    # ---- build by type ----
    if p == "i_beam":
        if (tw <= 0 or tf <= 0) and t_plate > 0:
            tw = tw if tw > 0 else t_plate
            tf = tf if tf > 0 else t_plate
        return i_beam(b=b, h=h, tw=tw, tf=tf, r1=r1, nseg=nseg)

    if p == "channel":
        if tw <= 0 and tf > 0:
            tw = tf
        if tf <= 0 and tw > 0:
            tf = tw
        return channel(h=h, b=b, tw=tw, tf=tf, r1=r1, nseg=nseg)

    if p == "angle":
        t = t_plate if t_plate > 0 else (tf if tf > 0 else tw)
        b_leg = b if b > 0 else h
        return angle(h=h, b=b_leg, t=t, r1=r1, nseg=nseg)

    if p == "t_beam":
        if tw <= 0 and t_plate > 0:
            tw = t_plate
        if tf <= 0 and t_plate > 0:
            tf = t_plate
        return t_beam(b=b, h=h, tw=tw, tf=tf, r1=r1, nseg=nseg)

    if p == "rect_tube":
        t = t_plate if t_plate > 0 else (tw if tw > 0 else tf)
        return rect_tube(b=b, h=h, t=t)

    if p == "circ_tube":
        d = d_circle if d_circle > 0 else h
        t = t_plate if t_plate > 0 else (tw if tw > 0 else tf)
        return circ_tube(d=d, t=t, nseg=max(nseg * 4, 32))

    # This should never happen if _PROFILE_TYPE_MAP values are correct.
    raise ValueError(f"No geometry builder for profile type: {p!r}")


def apply_tekla_position(
    g: BaseGeometry,
    *,
    plane: str = "MIDDLE",
    plane_offset: float = 0.0,
    rotation: Any = "TOP",
    rotation_offset: float = 0.0,
    depth: str = "MIDDLE",
    depth_offset: float = 0.0
) -> BaseGeometry:
    """
    Apply Tekla-style rotation and offsets to geometry.

    Input geometry is in Tekla coords:
    centroid at (0,0), +U = Front, +V = Top.
    """
    if g is None or g.is_empty:
        return g

    def _norm_plane(val: Any) -> str:
        key = str(val or "MIDDLE").strip().upper()
        if key in ("MID", "CENTER", "CENTRE"):
            key = "MIDDLE"
        # Tekla UI plane: LEFT/RIGHT/MIDDLE
        if key == "LEFT":
            key = "BACK"
        elif key == "RIGHT":
            key = "FRONT"
        return key

    def _norm_depth(val: Any) -> str:
        key = str(val or "MIDDLE").strip().upper()
        if key in ("MID", "CENTER", "CENTRE"):
            key = "MIDDLE"
        # Tekla UI depth: FRONT/BEHIND/MIDDLE
        if key == "FRONT":
            key = "TOP"
        elif key == "BEHIND":
            key = "BELOW"
        elif key == "BOTTOM":
            key = "BELOW"
        return key

    rot_map = {
        "TOP": 0.0,
        "FRONT": -90.0,
        "BELOW": 180.0,
        "BACK": 90.0,
    }

    if isinstance(rotation, str):
        rot_key = rotation.strip().upper()
        if rot_key in rot_map:
            angle = rot_map[rot_key] + float(rotation_offset)
        else:
            # allow numeric strings
            try:
                angle = float(rotation) + float(rotation_offset)
            except Exception:
                raise ValueError(f"Unsupported rotation: {rotation}")
    else:
        angle = float(rotation) + float(rotation_offset)

    g_rot = rotate(g, angle, origin=(0.0, 0.0), use_radians=False)

    minx, miny, maxx, maxy = g_rot.bounds
    plane_key = _norm_plane(plane)
    depth_key = _norm_depth(depth)

    if plane_key == "MIDDLE":
        base_u = 0.0
    elif plane_key == "FRONT":
        base_u = -maxx
    elif plane_key == "BACK":
        base_u = -minx
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    if depth_key == "MIDDLE":
        base_v = 0.0
    elif depth_key == "TOP":
        base_v = -maxy
    elif depth_key == "BELOW":
        base_v = -miny
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    xoff = base_u + float(plane_offset)
    yoff = base_v + float(depth_offset)

    return translate(g_rot, xoff=xoff, yoff=yoff)


def apply_tekla_offsets_only(
    g: BaseGeometry,
    *,
    plane: str = "MIDDLE",
    plane_offset: float = 0.0,
    depth: str = "MIDDLE",
    depth_offset: float = 0.0
) -> BaseGeometry:
    """
    Apply only plane/depth offsets (no rotation).
    Wrapper for the 'part_axes_joint' canon.
    """
    return apply_tekla_position(
        g,
        plane=plane,
        plane_offset=plane_offset,
        rotation="TOP",
        rotation_offset=0.0,
        depth=depth,
        depth_offset=depth_offset,
    )
