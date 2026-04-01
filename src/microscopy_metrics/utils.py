def umToPx(x, axisPhysicalSize):
    """
    Args:
        x (float): The value in um to convert
        axisPhysicalSize (float):Physical size of a pixel (µm/px)

    Returns:
        float: The number of pixels corresponding to x
    """
    if axisPhysicalSize == 0.0:
        return 0.0
    xConv = x / axisPhysicalSize
    return xConv


def pxToUm(x, axisPhysicalSize):
    """
    Args:
        x (float):  The value in pixels to convert
        axisPhysicalSize (float): Physical size of a pixel (µm/px)

    Returns:
        float: The µm value corresponding to x
    """
    xConv = x * axisPhysicalSize
    return xConv




