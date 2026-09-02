import math


def rectangle_properties(width, height):
    """
    Calculate geometric properties of a solid rectangle.

    Parameters
    ----------
    width : float
        Rectangle width, b.

    height : float
        Rectangle height, h.

    Returns
    -------
    dict
        Geometric properties.
    """

    area = width * height

    centroid_x = width / 2
    centroid_y = height / 2

    Ix = width * height**3 / 12
    Iy = height * width**3 / 12

    return {
        "area": area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "Ix": Ix,
        "Iy": Iy,
    }


def circle_properties(radius):
    """
    Calculate geometric properties of a solid circle.

    Parameters
    ----------
    radius : float
        Circle radius.

    Returns
    -------
    dict
        Geometric properties.
    """

    area = math.pi * radius**2

    centroid_x = radius
    centroid_y = radius

    Ix = math.pi * radius**4 / 4
    Iy = Ix

    return {
        "area": area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "Ix": Ix,
        "Iy": Iy,
    }


def i_section_properties(
    depth,
    flange_width,
    flange_thickness,
    web_thickness,
):
    """
    Calculate geometric properties of a symmetric I / W section.

    Parameters
    ----------
    depth : float
        Total section depth, d.

    flange_width : float
        Flange width, bf.

    flange_thickness : float
        Flange thickness, tf.

    web_thickness : float
        Web thickness, tw.

    Returns
    -------
    dict
        Geometric properties.
    """

    # ---------------------------------------------------------
    # 1. Basic dimensions
    # ---------------------------------------------------------

    web_height = depth - 2 * flange_thickness

    if web_height <= 0:
        raise ValueError(
            "Depth must be greater than twice the flange thickness."
        )

    # ---------------------------------------------------------
    # 2. Areas
    # ---------------------------------------------------------

    flange_area = flange_width * flange_thickness
    web_area = web_thickness * web_height

    total_area = (
        2 * flange_area
        + web_area
    )

    # ---------------------------------------------------------
    # 3. Centroid
    # Symmetric section
    # ---------------------------------------------------------

    centroid_x = flange_width / 2
    centroid_y = depth / 2

    # ---------------------------------------------------------
    # 4. Local inertias
    # ---------------------------------------------------------

    Ix_flange_local = (
        flange_width
        * flange_thickness**3
        / 12
    )

    Iy_flange_local = (
        flange_thickness
        * flange_width**3
        / 12
    )

    Ix_web_local = (
        web_thickness
        * web_height**3
        / 12
    )

    Iy_web_local = (
        web_height
        * web_thickness**3
        / 12
    )

    # ---------------------------------------------------------
    # 5. Parallel-axis distances
    # ---------------------------------------------------------

    flange_centroid_distance = (
        depth / 2
        - flange_thickness / 2
    )

    # ---------------------------------------------------------
    # 6. Total Ix
    # ---------------------------------------------------------

    Ix = (
        2
        * (
            Ix_flange_local
            + flange_area
            * flange_centroid_distance**2
        )
        + Ix_web_local
    )

    # ---------------------------------------------------------
    # 7. Total Iy
    # No horizontal offset because section is symmetric
    # ---------------------------------------------------------

    Iy = (
        2 * Iy_flange_local
        + Iy_web_local
    )

    return {
        "area": total_area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "Ix": Ix,
        "Iy": Iy,
        "web_height": web_height,
        "flange_area": flange_area,
        "web_area": web_area,
        "Ix_flange_local": Ix_flange_local,
        "Iy_flange_local": Iy_flange_local,
        "Ix_web_local": Ix_web_local,
        "Iy_web_local": Iy_web_local,
        "flange_centroid_distance": flange_centroid_distance,
    }