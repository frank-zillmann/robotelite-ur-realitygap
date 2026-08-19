"""Universal Robots brand palette + a shared matplotlib style for case 2 plots.

Colors are sourced from universal-robots.com's own site CSS (heading/link/icon
colors, not guessed), so plots read as on-brand in front of UR:

    NAVY        #002b39   heading/icon color, most-used dark tone on the site
    BLUE        #3e769c   primary link/heading blue
    MID_BLUE    #56a0d3
    LIGHT_BLUE  #76b4de
    DARK_BLUE   #134669
    GRAY        #899ba0   neutral/secondary text
    GRID        #d7dddf   the site's own hairline/border gray

Call ``apply()`` once (before creating any figure) to make every subsequent
plot use this palette by default -- color cycle, text/axis color, spines,
grid. Individual plots can still pick specific colors from the constants
above for meaning (e.g. two-tone diverging bars).

    import ur_style
    ur_style.apply()
"""
import matplotlib as mpl

NAVY = "#002b39"
BLUE = "#3e769c"
MID_BLUE = "#56a0d3"
LIGHT_BLUE = "#76b4de"
DARK_BLUE = "#134669"
GRAY = "#899ba0"
GRID = "#d7dddf"

# Default series color cycle, most-to-least prominent.
PALETTE = [BLUE, LIGHT_BLUE, MID_BLUE, DARK_BLUE, GRAY]


def apply():
    """Set matplotlib rcParams to the UR palette. Safe to call more than once."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "font.family": "sans-serif",
        "text.color": NAVY,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.labelcolor": NAVY,
        "axes.edgecolor": GRAY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })
