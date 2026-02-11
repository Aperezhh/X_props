# X_props

`X_props` is a lightweight Python module for creating 2D steel cross-sections and calculating section properties.

## Features

- Parametric steel profiles: I/H, channel, angle, T, rectangular tube, circular tube
- Primitive geometry helpers: rectangle, plate, circle, hole patterns
- Section properties: `A`, `Cx`, `Cy`, `Ix`, `Iy`, `Ixy`, `Ip`, radii of gyration, section moduli
- Properties at an arbitrary point (`props_at_point`)
- Tekla-oriented placement helpers (`apply_tekla_position`, `apply_tekla_offsets_only`)
- Plotting support in `xsprops_plot.py`
- Bolt and weld utilities

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from xsprops import i_beam, props, pretty

section = i_beam(b=200, h=300, tw=8, tf=12, r1=10)
p = props(section)
print(pretty(p))
```

## Plot Example

```python
from xsprops import plate, props
from xsprops_plot import plot_profile_with_props

section = plate(200, 300)
p = props(section)
plot_profile_with_props(section, p, title="Plate 200x300")
```

## Examples For Engineers

- Main practical examples are in the notebook: `module_xprops.ipynb`
- Notebook figures are included in the repository root:
  - `Example 9_6.PNG`
  - `Example 9_8.PNG`
  - `Example 10_5.PNG`
- To run the notebook:

```bash
pip install jupyter
jupyter lab
```

## Coordinate Convention

- Centroid is at `(0, 0)`
- `+U` is the front direction
- `+V` is the top direction

## Requirements

- Python 3.10+
- `numpy`
- `shapely`
- `matplotlib` (needed for plotting)

## License

This project is licensed under the MIT License. See `LICENSE`.
