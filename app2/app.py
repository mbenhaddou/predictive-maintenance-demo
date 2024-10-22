import colorsys

import streamlit as st
import util
from util import ThemeColor


preset_colors: list[tuple[str, ThemeColor]] = [
    ("Default light", ThemeColor(
            primaryColor="#ff4b4b",
            backgroundColor="#ffffff",
            secondaryBackgroundColor="#f0f2f6",
            textColor="#31333F",
        )),
    ("Default dark", ThemeColor(
            primaryColor="#ff4b4b",
            backgroundColor="#0e1117",
            secondaryBackgroundColor="#262730",
            textColor="#fafafa",
    ))
]

theme_from_initial_config = util.get_config_theme_color()
if theme_from_initial_config:
    preset_colors.append(("From the config", theme_from_initial_config))

default_color = preset_colors[0][1]


def sync_rgb_to_hls(key: str):
    # HLS states are necessary for the HLS sliders.
    rgb = util.parse_hex(st.session_state[key])
    hls = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
    st.session_state[f"{key}H"] = round(hls[0] * 360)
    st.session_state[f"{key}L"] = round(hls[1] * 100)
    st.session_state[f"{key}S"] = round(hls[2] * 100)



def set_color(key: str, color: str):
    st.session_state[key] = color
    sync_rgb_to_hls(key)





st.title("Streamlit color theme editor")


def on_preset_color_selected():
    _, color = preset_colors[st.session_state.preset_color]
    set_color('primaryColor', color.primaryColor)
    set_color('backgroundColor', color.backgroundColor)
    set_color('secondaryBackgroundColor', color.secondaryBackgroundColor)
    set_color('textColor', color.textColor)


st.selectbox("Preset colors", key="preset_color", options=range(len(preset_colors)), format_func=lambda idx: preset_colors[idx][0], on_change=on_preset_color_selected)

if st.button("🎨 Generate a random color scheme 🎲"):
    primary_color, text_color, basic_background, secondary_background = util.generate_color_scheme()
    set_color('primaryColor', primary_color)
    set_color('backgroundColor', basic_background)
    set_color('secondaryBackgroundColor', secondary_background)
    set_color('textColor', text_color)




if st.checkbox("Apply theme to this page"):
    st.info("Select 'Custom Theme' in the settings dialog to see the effect")

    def reconcile_theme_config():
        keys = ['primaryColor', 'backgroundColor', 'secondaryBackgroundColor', 'textColor']
        has_changed = False
        for key in keys:
            if st._config.get_option(f'theme.{key}') != st.session_state[key]:
                st._config.set_option(f'theme.{key}', st.session_state[key])
                has_changed = True
        if has_changed:
            st.rerun()

    reconcile_theme_config()
