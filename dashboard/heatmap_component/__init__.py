import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_heatmap_component = components.declare_component("heatmap_click", path=_COMPONENT_DIR)


def heatmap_click(fig, height=830, key=None):
    """Render a Plotly treemap with click-to-ticker (no drill-down).
    Returns the clicked ticker string, or None.
    """
    import json
    fig_json = fig.to_json()
    return _heatmap_component(fig_json=fig_json, height=height, key=key, default=None)
