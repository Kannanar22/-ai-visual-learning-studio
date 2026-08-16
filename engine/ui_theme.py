"""
Shared "premium" visual theme for the Streamlit app: a bit of custom CSS
(gradient headers, card-styled containers, nicer buttons/inputs) plus a
small contrast-color helper so text placed on top of arbitrary node/box
colors is always readable, regardless of light/dark mode.
"""
import streamlit as st

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* App background: soft premium gradient */
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.08), transparent),
                radial-gradient(1000px 500px at 100% 0%, rgba(236,72,153,0.06), transparent),
                linear-gradient(180deg, #f7f8fc 0%, #f3f4fa 100%);
}

/* Page titles */
h1 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #4338CA, #7C3AED 45%, #DB2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    padding-bottom: 2px;
}
h2, h3, h4 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; color: #1E1B3A !important; }

/* Force readable body text regardless of the viewer's OS/browser dark-mode
   setting — the app's background is always light, so text must always be
   dark, even if Streamlit's auto theme detection picks "dark". */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stApp div[data-testid="stMarkdownContainer"],
.stApp div[data-testid="stMarkdownContainer"] * ,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stTable"] * ,
.stApp [data-testid="stDataFrame"] * {
    color: #1E1B3A;
}
.stApp { background-color: #F7F8FC; }
/* Keep the deliberately-styled sidebar text white even though the rule above targets .stApp * */
[data-testid="stSidebar"] * { color: #EDEBFF !important; }

/* Captions */
[data-testid="stCaptionContainer"] { color: #6B7280 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B3A 0%, #2A2560 100%);
}
[data-testid="stSidebar"] * { color: #EDEBFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,0.35);
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(79,70,229,0.28);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(79,70,229,0.38);
    color: white;
    border-color: rgba(99,102,241,0.6);
}

/* Cards for expanders / containers */
[data-testid="stExpander"] {
    border-radius: 14px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    background: rgba(255,255,255,0.75) !important;
    box-shadow: 0 2px 10px rgba(30,27,58,0.05);
}

/* Text areas / inputs */
.stTextArea textarea, .stTextInput input {
    border-radius: 10px !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
}

/* Tables */
[data-testid="stTable"] { border-radius: 12px; overflow: hidden; }

/* Divider glow under subheaders */
.premium-divider {
    height: 3px; border: none; border-radius: 3px; margin: 6px 0 18px 0;
    background: linear-gradient(90deg, #4F46E5, #DB2777, transparent);
}

/* Small pill badge, used for section eyebrows */
.premium-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    background: rgba(79,70,229,0.10); color: #4F46E5; font-weight: 600;
    font-size: 12px; letter-spacing: 0.03em; margin-bottom: 6px;
}
</style>
"""


def inject_premium_css() -> None:
    """Call once near the top of a page to apply the shared premium theme."""
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


def premium_divider() -> None:
    st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (26, 26, 46)  # fallback: dark navy


def contrast_color(hex_bg: str, light: str = "#FFFFFF", dark: str = "#111111") -> str:
    """
    Given a background hex color, return whichever of `light`/`dark` gives
    better readability, using perceived (WCAG-ish) luminance.
    This is what powers the "Auto-contrast text" option, so labels never
    disappear against a node/box of a similar color.
    """
    r, g, b = _hex_to_rgb(hex_bg)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return dark if luminance > 0.6 else light


def resolve_text_color(bg_hex: str, manual_color: str, auto_contrast: bool) -> str:
    """Pick manual color, or auto-computed contrast color, for text on `bg_hex`."""
    if auto_contrast:
        return contrast_color(bg_hex)
    return manual_color
