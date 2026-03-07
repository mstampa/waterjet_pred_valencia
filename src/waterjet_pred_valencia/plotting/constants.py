"""Shared style constants for simulation plotting."""

from typing import Dict, Tuple

from bokeh.palettes import Blues8, Colorblind5

from ..parameters import num_drop_classes

PHASE_COLORS: Dict[str, str] = {
    "core": Colorblind5[2],
    "air": Colorblind5[1],
    "stream": Colorblind5[3],
}
SPRAY_COLORS: Tuple[str, ...] = tuple(Blues8[i] for i in range(num_drop_classes))
SURROUNDINGS_COLOR = "#4a4a4a"

DEFAULT_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
DESIRED_MAJOR_TICKS = 10
