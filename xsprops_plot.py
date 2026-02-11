# xsprops_plot.py
# flake8: noqa E501
"""
2D section plotting helpers for xsprops.

Provides Matplotlib wrappers for drawing cross-sections and standard annotations:
- title (annotate_title)
- bounding box (annotate_bbox)
- centroid (annotate_centroid)
- dimension lines (annotate_dims)
- arrow callouts (annotate_arrow_text)

Coordinate system: +Y up, origin in the same coordinates as the input geometry.

Dependencies: matplotlib, numpy, shapely, xsprops (optional).
"""

# Note: numerical properties (A, I, centroid) are computed in xsprops.props().
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import Sequence, Tuple, Dict, Any, Optional, Mapping

from shapely.geometry.base import BaseGeometry
try:
    from xsprops import plot_section as _plot_section  # type: ignore
except Exception:
    _plot_section = None


DEFAULT_ARROW_KWARGS: Dict[str, Any] = dict(
    arrowstyle="<|-|>",
    color="black",
    mutation_scale=15,
    shrinkA=0,
    shrinkB=0
)
DEFAULT_TEXT_KWARGS: Dict[str, Any] = dict(
    fontsize=9,
    ha='center',
    va='center',
    color="black",
    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8)
)
DEFAULT_CENTROID_MARKER_KWARGS: Dict[str, Any] = dict(
    marker='+',
    color='red',
    s=50,
    zorder=10,
    linewidths=1.0
)
DEFAULT_SIMPLE_ARROW_KWARGS: Dict[str, Any] = dict(
    arrowstyle="->",
    color="black",
    connectionstyle="arc3"
)

def init_plot(
    figsize: Tuple[float, float] = (6, 6),
    show_axes: bool = False,
    show_grid: bool = False,
    grid_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Figure, Axes]:
    """
    Create and return a Matplotlib Figure/Axes for section plotting.

    Args:
        figsize: Figure size in inches (width, height).
        show_axes: Show axes labels and spines.
        show_grid: Show grid lines.
        grid_kwargs: Extra kwargs passed to ax.grid.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', adjustable='box')

    if show_axes:
        ax.axis("on")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    else:
        ax.axis("off")

    if show_grid:
        default_grid_style: Dict[str, Any] = dict(
            linestyle=":", linewidth=0.6, color="gray", alpha=0.7
        )
        final_grid_kwargs: Dict[str, Any] = default_grid_style.copy()
        if grid_kwargs is not None:
            final_grid_kwargs.update(grid_kwargs)
        ax.grid(True, **final_grid_kwargs)

    return fig, ax


def plot_section_with_style(
    section: BaseGeometry,
    *,
    face: Optional[str] = None,
    edge: Optional[str] = None,
    show_axes: bool = False,
    show_grid: bool = False,
    figsize: Tuple[float, float] = (6, 6),
    padding: float = 0.1
) -> Tuple[Figure, Axes]:
    """
    Create a figure, draw the section, and set view limits.

    Args:
        section: Shapely geometry (Polygon or MultiPolygon).
        face: Fill color.
        edge: Edge color.
        show_axes: Show axes.
        show_grid: Show grid.
        figsize: Figure size in inches.
        padding: Relative padding for axis limits.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = init_plot(figsize=figsize, show_axes=show_axes, show_grid=show_grid)

    face_color = face if face is not None else "lightblue"
    edge_color = edge if edge is not None else "k"

    if _plot_section is not None:
        _plot_section(section, ax=ax, face=face_color, edge=edge_color)
    else:
        raise RuntimeError(
            "plot_section is not available. Import xsprops before calling plot_section_with_style()."
        )

    if not section.is_empty:
        minx, miny, maxx, maxy = section.bounds
        width = maxx - minx
        height = maxy - miny

        pad_x = width * padding
        pad_y = height * padding
        if width <= 1e-9:
            pad_x = 1.0 * padding if padding > 0 else 1.0
        if height <= 1e-9:
            pad_y = 1.0 * padding if padding > 0 else 1.0

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    return fig, ax


def annotate_title(
    ax: Axes,
    template: str,
    params: Optional[Mapping[str, Any]] = None,
    **text_kwargs
):
    """
    Add a formatted title to the axes.

    Args:
        ax: Matplotlib axes.
        template: Format string for the title.
        params: Values used to format the template.
        **text_kwargs: Passed to ax.set_title().
    """
    params = params or {}
    try:
        title = template.format(**params)
    except KeyError as e:
        print(f"Warning: Key not found for title template. Error: {e}")
        title = template
    ax.set_title(title, **text_kwargs)


def plot_centroid_marker(ax: Axes, cx: float, cy: float, **marker_kwargs) -> None:
    """Plot centroid marker only."""
    kw = DEFAULT_CENTROID_MARKER_KWARGS.copy()
    kw.update(marker_kwargs)
    ax.scatter([cx], [cy], **kw)


def annotate_centroid(
    ax: Axes,
    props_data: Mapping[str, Any],
    *,
    show_marker: bool = True,
    show_text: bool = True,
    text_fmt: str = '({Cx:.1f}, {Cy:.1f})',
    marker_kwargs: Optional[Dict[str, Any]] = None,
    text_kwargs: Optional[Dict[str, Any]] = None,
    text_offset: Tuple[float, float] = (5, 5)
):
    """
    Annotate the centroid with a marker and/or text.

    Args:
        ax: Matplotlib axes.
        props_data: Dictionary returned by xsprops.props(). Must include Cx, Cy.
        show_marker: Draw centroid marker.
        show_text: Draw centroid label.
        text_fmt: Format string for the centroid label.
        marker_kwargs: Extra kwargs for the marker.
        text_kwargs: Extra kwargs for the label text.
        text_offset: Offset for the label in points.
    """
    if 'Cx' not in props_data or 'Cy' not in props_data:
        raise ValueError("props_data must contain 'Cx' and 'Cy' for centroid annotation.")
    cx, cy = props_data['Cx'], props_data['Cy']

    if show_marker:
        plot_centroid_marker(ax, cx, cy, **(marker_kwargs or {}))

    if show_text:
        try:
            label = text_fmt.format(Cx=cx, Cy=cy)
        except KeyError as e:
            print(f"Warning: Key not found for centroid text format. Error: {e}")
            label = f"({cx:.1f}, {cy:.1f})"

        merged_text_kwargs: Dict[str, Any] = DEFAULT_TEXT_KWARGS.copy()
        merged_text_kwargs.update({'ha': 'left', 'va': 'bottom', 'bbox': None})
        if text_kwargs:
            merged_text_kwargs.update(text_kwargs)

        ax.annotate(
            label,
            xy=(cx, cy),
            xytext=text_offset,
            textcoords='offset points',
            **merged_text_kwargs
        )


def annotate_bbox(
    ax: Axes,
    section: BaseGeometry,
    *,
    pad: float = 5.0,
    text_pad: float = 2.0,
    arrow_kwargs: Optional[Dict[str, Any]] = None,
    text_kwargs: Optional[Dict[str, Any]] = None,
    fmt: str = "{:.1f}"
):
    """
    Annotate overall width and height of a section.

    Args:
        ax: Matplotlib axes.
        section: Shapely geometry.
        pad: Offset of dimension lines from the geometry.
        text_pad: Offset of text from dimension lines.
        arrow_kwargs: Arrow style overrides.
        text_kwargs: Text style overrides.
        fmt: Number format for dimensions.
    """
    if section.is_empty:
        print("Warning: annotate_bbox called with empty geometry.")
        return

    minx, miny, maxx, maxy = section.bounds
    width = maxx - minx
    height = maxy - miny

    merged_arrow_kwargs: Dict[str, Any] = DEFAULT_ARROW_KWARGS.copy()
    if arrow_kwargs:
        merged_arrow_kwargs.update(arrow_kwargs)
    merged_text_kwargs: Dict[str, Any] = DEFAULT_TEXT_KWARGS.copy()
    if text_kwargs:
        merged_text_kwargs.update(text_kwargs)

    arrow_y_h = miny - pad
    text_x_h = (minx + maxx) / 2
    text_y_h = arrow_y_h - text_pad
    arrow_x_v = maxx + pad
    text_x_v = arrow_x_v + text_pad
    text_y_v = (miny + maxy) / 2

    ax.annotate("", xy=(minx, arrow_y_h), xytext=(maxx, arrow_y_h), arrowprops=merged_arrow_kwargs)
    h_text_kwargs = merged_text_kwargs.copy()
    h_text_kwargs.update({'va': 'top'})
    ax.text(text_x_h, text_y_h, fmt.format(width), **h_text_kwargs)

    ax.annotate("", xy=(arrow_x_v, miny), xytext=(arrow_x_v, maxy), arrowprops=merged_arrow_kwargs)
    v_text_kwargs = merged_text_kwargs.copy()
    v_text_kwargs.update({'ha': 'left', 'rotation': 90})
    ax.text(text_x_v, text_y_v, fmt.format(height), **v_text_kwargs)

    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    required_min_x = min(current_xlim[0], minx - pad)
    required_max_x = max(current_xlim[1], text_x_v + text_pad * 2)
    required_min_y = min(current_ylim[0], text_y_h - text_pad * 2)
    required_max_y = max(current_ylim[1], maxy + pad)

    ax.set_xlim(min(current_xlim[0], required_min_x), max(current_xlim[1], required_max_x))
    ax.set_ylim(min(current_ylim[0], required_min_y), max(current_ylim[1], required_max_y))


def plot_profile_with_props(
    section: BaseGeometry,
    props_data: Optional[Mapping[str, Any]] = None,
    *,
    title: Optional[str] = None,
    show_centroid: bool = True,
    show_bbox: bool = True,
    show_props_text: bool = True,
    figsize: Tuple[float, float] = (8, 8),
    face: str = "lightblue",
    edge: str = "k"
) -> Tuple[Figure, Axes]:
    """
    Plot section with properties annotation.

    Convenience function combining plot + annotations.

    Args:
        section: Shapely geometry
        props_data: Dict from props() or None to compute automatically
        title: Plot title
        show_centroid: Mark centroid with + marker
        show_bbox: Show bounding box dimensions
        show_props_text: Show properties text box
        figsize: Figure size
        face, edge: Colors

    Returns:
        (Figure, Axes) tuple
    """
    if props_data is None:
        from xsprops import props

        props_data = props(section)

    fig, ax = plot_section_with_style(
        section,
        face=face,
        edge=edge,
        figsize=figsize,
        show_axes=False
    )

    if title:
        ax.set_title(title, fontsize=12, weight="bold")

    if show_centroid:
        annotate_centroid(ax, props_data, show_marker=True, show_text=True)

    if show_bbox:
        annotate_bbox(ax, section)

    if show_props_text:
        props_text = (
            f"A = {props_data['A']:,.0f} mm^2\n"
            f"Ix = {props_data['Ix']:,.0f} mm^4\n"
            f"Iy = {props_data['Iy']:,.0f} mm^4\n"
            f"Wx = {props_data.get('Wx_plus', 0):,.0f} mm^3\n"
            f"Wy = {props_data.get('Wy_plus', 0):,.0f} mm^3"
        )
        if "rx" in props_data:
            props_text += f"\nrx = {props_data['rx']:.1f} mm"
        if "ry" in props_data:
            props_text += f"\nry = {props_data['ry']:.1f} mm"

        ax.text(
            0.02,
            0.98,
            props_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        )

    return fig, ax


def save_plot(fig: Figure, filename: str, dpi: int = 150) -> None:
    """Save figure to file with sensible defaults."""
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")


def annotate_dims(
    ax: Axes,
    dim_lines: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    labels: Sequence[str],
    *,
    offset: float = 5.0,
    text_offset: float = 2.0,
    orientation: str = 'auto',
    arrow_kwargs: Optional[Dict[str, Any]] = None,
    text_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Draw dimension lines between point pairs.

    Args:
        ax: Matplotlib axes.
        dim_lines: Sequence of point pairs.
        labels: Labels for each dimension line.
        offset: Offset of dimension lines from the measured line.
        text_offset: Offset of text from the dimension line.
        orientation: 'horizontal', 'vertical', or 'auto'.
        arrow_kwargs: Arrow style overrides.
        text_kwargs: Text style overrides.
    """
    if len(dim_lines) != len(labels):
        raise ValueError("Number of dimension lines must match number of labels.")

    merged_arrow_kwargs: Dict[str, Any] = DEFAULT_ARROW_KWARGS.copy()
    if arrow_kwargs:
        merged_arrow_kwargs.update(arrow_kwargs)
    merged_text_kwargs: Dict[str, Any] = DEFAULT_TEXT_KWARGS.copy()
    if text_kwargs:
        merged_text_kwargs.update(text_kwargs)

    for (p1, p2), label_spec in zip(dim_lines, labels):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        dist = float(np.hypot(dx, dy))

        current_orientation = orientation
        if orientation == 'auto':
            current_orientation = 'horizontal' if abs(dx) >= abs(dy) else 'vertical'

        if current_orientation == 'horizontal':
            sign_y = np.sign(offset) if offset != 0 else 1
            mid_y_orig = (y1 + y2) / 2
            off_y = mid_y_orig + offset
            off_p1 = (x1, off_y)
            off_p2 = (x2, off_y)
            text_x = (x1 + x2) / 2
            text_y = off_y + text_offset * sign_y
            rot = 0
            ha, va = 'center', 'bottom' if sign_y > 0 else 'top'
        elif current_orientation == 'vertical':
            sign_x = np.sign(offset) if offset != 0 else 1
            mid_x_orig = (x1 + x2) / 2
            off_x = mid_x_orig + offset
            off_p1 = (off_x, y1)
            off_p2 = (off_x, y2)
            text_x = off_x + text_offset * sign_x
            text_y = (y1 + y2) / 2
            rot = 90
            ha, va = 'left' if sign_x > 0 else 'right', 'center'
        else:
            raise NotImplementedError(f"Orientation '{current_orientation}' not implemented for annotate_dims")

        ax.annotate("", xy=off_p1, xytext=off_p2, arrowprops=merged_arrow_kwargs)

        final_label = label_spec
        if isinstance(label_spec, str) and '{dist' in label_spec:
            try:
                final_label = label_spec.format(dist=dist)
            except Exception as e:
                print(
                    f"Warning: Could not format label '{label_spec}' with {{dist=...}}. Error: {e}. Using raw label."
                )
                final_label = label_spec

        t_kwargs: Dict[str, Any] = merged_text_kwargs.copy()
        t_kwargs.update({'ha': ha, 'va': va, 'rotation': rot})
        ax.text(text_x, text_y, final_label, **t_kwargs)


def annotate_arrow_text(
    ax: Axes,
    text: Optional[str] = None,
    text_template: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    *,
    xy: Tuple[float, float],
    xytext: Tuple[float, float],
    arrow_kwargs: Optional[Dict[str, Any]] = None,
    **text_kwargs
):
    """
    Draw a text label with an arrow.

    Args:
        ax: Matplotlib axes.
        text: Label text.
        xy: Arrow tip location.
        xytext: Text location.
        arrow_kwargs: Arrow style overrides.
        text_kwargs: Text style overrides.
    """
    final_formatted_text = ""

    if text is not None:
        final_formatted_text = text
    elif text_template is not None:
        params = params or {}
        try:
            final_formatted_text = text_template.format(**params)
        except KeyError as e:
            print(f"Warning: Key not found for text_template in annotate_arrow_text. Error: {e}")
            final_formatted_text = text_template
        except Exception as e:
            print(f"Warning: Error formatting text_template in annotate_arrow_text. Error: {e}")
            final_formatted_text = text_template
    else:
        print("Warning: Both 'text' and 'text_template' are None in annotate_arrow_text.")
        final_formatted_text = ""

    final_arrowprops: Dict[str, Any] = DEFAULT_SIMPLE_ARROW_KWARGS.copy()
    if arrow_kwargs:
        final_arrowprops.update(arrow_kwargs)

    final_text_kwargs: Dict[str, Any] = DEFAULT_TEXT_KWARGS.copy()
    final_text_kwargs['bbox'] = None
    final_text_kwargs.update({'ha': 'center', 'va': 'center'})
    if text_kwargs:
        final_text_kwargs.update(text_kwargs)

    ax.annotate(
        final_formatted_text,
        xy=xy,
        xytext=xytext,
        xycoords='data',
        textcoords='data',
        arrowprops=final_arrowprops,
        **final_text_kwargs
    )
