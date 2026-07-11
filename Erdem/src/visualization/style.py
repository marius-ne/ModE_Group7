from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.misc.constants import RESULTS_DIR


def reset_plot_style():
    """
    Resets the Matplotlib plot settings to the default values.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)  # Reset to default Matplotlib parameters


def get_figsize(
        width_cm=16,
        aspect: str | float | tuple[int, int] = "golden",
        nrows: int = 1,
        ncols: int = 1
) -> tuple[float, float]:
    """
    Computes a suitable matplotlib figsize based on a given width in centimeters and aspect ratio.
    :param width_cm: Width in centimeters (default is 16 cm, which is typical for a single column in A4 format)
    :param aspect: Aspect ratio for the width relative to the height (default is "golden", which uses the golden ratio).
    :param nrows: Number of subplot rows in the figure (Vertical subplots increase the figure height).
    :param ncols: Number of subplot columns in the figure (Horizontal subplots keep the same figure height).
    :return: Tuple of figure size (width_in, height_in) in inches.
    """
    width_in = width_cm / 2.54  # Convert centimeters to inches

    if aspect == "golden":
        aspect_ratio = (1 + 5**0.5) / 2

    elif aspect == "16:9":
        aspect_ratio = 16 / 9

    elif aspect == "4:3":
        aspect_ratio = 4 / 3

    elif isinstance(aspect, (float, int)):
        aspect_ratio = float(aspect)

    elif isinstance(aspect, (tuple, list)) and len(aspect) == 2:
        aspect_ratio = aspect[0] / aspect[1]

    else:
        raise ValueError("Unsupported aspect ratio.")

    single_height = width_in / aspect_ratio

    figure_height = single_height * nrows

    return width_in, figure_height


def apply_style(
    *,
    width_cm: float = 16,
    aspect: str | float | tuple[int, int] = "golden",
    nrows: int = 1,
    ncols: int = 1,
    science: bool = True,
    grid: bool = False,
    latex: bool = False,
) -> None:
    """
    Applies a consistent style to Matplotlib plots.
    :param width_cm: Width of the figure in centimeters (default is 16 cm, which is typical for a single column in A4 format)
    :param aspect: Aspect ratio for the width relative to the height (default is "golden", which uses the golden ratio).
    :param nrows: Number of subplot rows in the figure (Vertical subplots increase the figure height).
    :param ncols: Number of subplot columns in the figure (Horizontal subplots keep the same figure height).
    :param science: If True, imports "scienceplots" package and activates science style to the plots.
    :param grid: If True, enables "scienceplots" grid style on the plots.
    :param latex: If True, enables LaTeX rendering for text in plots (requires LaTeX installation).
    """
    # Reset to default settings before applying new style
    reset_plot_style()

    # --- SciencePlots ---------------------
    if science:

        import scienceplots

        styles = ["science"]
        if grid:
            styles.append("grid")
        if not latex:
            styles.append("no-latex")

        plt.style.use(styles)

    # --- Matplotlib settings ------------------
    mpl.rcParams.update({

        # Figure
        "figure.figsize": get_figsize(
                width_cm=width_cm,
                aspect=aspect,
                nrows=nrows,
                ncols=ncols,
            ),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",


        # Fonts
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],

        "font.size": 10,

        "axes.labelsize": 10,
        "axes.titlesize": 10,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "legend.fontsize": 9,


        # Grid
        "axes.grid": grid,
        "axes.axisbelow": grid,

        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,


        # Lines
        "lines.linewidth": 2,
        "lines.markersize": 6,


        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",

        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        "xtick.major.size": 4,
        "ytick.major.size": 4,

        "xtick.minor.size": 2,
        "ytick.minor.size": 2,


        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        })


    # LaTeX
    if latex:
        mpl.rcParams.update({

            "text.usetex": True,

            "text.latex.preamble": r"""
                \usepackage{lmodern}
                \usepackage{amsmath}
                \usepackage{amssymb}
                \usepackage{siunitx}
            """,

        })


def safe_figure(
        figure_obj,
        save_path: Path | str,
        filename: str,
        file_type: str = "pdf",
        **safefig_kwargs
):
    """
    Save the current figure to a file in the result's directory.
    :param figure_obj: The Matplotlib figure object to save (can be None if using plt.savefig directly)
    :param save_path: Path to save the figure (relative to GRAPHICS_DIRECTORY/)
    :param filename: Name of the file (without extension)
    :param file_type: Type of the file (e.g., "png", "pdf")
    :return: None
    """
    print("\nSaving plot...")
    figure_filename = RESULTS_DIR / save_path / f"{filename}.{file_type}"
    figure_filename.parent.mkdir(parents=True, exist_ok=True)

    figure_obj.savefig(figure_filename, dpi=300, bbox_inches="tight", **safefig_kwargs)
    print(f"- Plot saved to: {figure_filename}")

