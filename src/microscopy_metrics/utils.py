def umToPx(x, axisPhysicalSize):
    """Converts a value from micrometers (µm) to pixels based on the physical size of a pixel.
    Args:
        x (float): The value in micrometers to convert
        axisPhysicalSize (float): Physical size of a pixel (µm/px)
    Returns:
        float: The pixel value corresponding to x
    """
    if axisPhysicalSize == 0.0:
        return 0.0
    xConv = x / axisPhysicalSize
    return xConv


def pxToUm(x, axisPhysicalSize):
    """Converts a value from pixels to micrometers (µm) based on the physical size of a pixel.
    Args:
        x (float): The value in pixels to convert
        axisPhysicalSize (float): Physical size of a pixel (µm/px)

    Returns:
        float: The micrometer value corresponding to x
    """
    xConv = x * axisPhysicalSize
    return xConv
