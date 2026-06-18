import matplotlib as mpl
import matplotlib.pyplot as plt


def reset_plot_settings():
    """
    Resets the Matplotlib plot settings to the default values.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)  # Reset to default Matplotlib parameters


def get_figsize(
        width_cm: int = 16,
        ratio: str | int | float | tuple | list = "golden"
) -> tuple[float, float]: # Default für ratio vielleicht anpassen, z.B. 1.618 (goldener Schnitt) oder 16/9 (Breitbild)
    """
    Calculates the figure size in inches based on a given width in centimeters and an aspect ratio.
    :param width_cm: Width in centimeters (default is 16 cm, which is typical for a single column in A4 format)
    :param ratio: Aspect ratio for the width relative to the height (default is "golden", which uses the golden ratio).
    :return: Tuple of figure size (width_in, height_in) in inches.
    """
    width_in = width_cm / 2.54  # Convert centimeters to inches

    if ratio == "golden":
        phi = (1 + 5 ** 0.5) / 2
        height_in = width_in / phi
    elif isinstance(ratio, (int, float)):
        height_in = width_in / ratio
    elif isinstance(ratio, (tuple, list)) and len(ratio) == 2:
        width_in = width_cm / 2.54
        height_in = (width_cm * ratio[1] / ratio[0]) / 2.54
    else:
        raise ValueError("Invalid ratio type. Use 'golden', a float, or a tuple/list of two numbers!")

    return width_in, height_in


