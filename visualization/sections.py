import numpy as np
import plotly.graph_objects as go

SECTION_LINE_COLOR = "#2C3E50"
CENTROID_AXIS_COLOR = "#7F8C8D"
CENTROID_COLOR = "#3498DB"
TEXT_COLOR = "#1a1a1a"

DIMENSION_COLOR = "#E67E22"
DIMENSION_TEXT_COLOR = "#8B5A00"

def _format_section_figure(fig, title=None):

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
        color=TEXT_COLOR,
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        color=TEXT_COLOR,
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=TEXT_COLOR),
        ),
        showlegend=False,
        height=500,
        margin=dict(
            l=40,
            r=40,
            t=80,
            b=40,
        ),
        font=dict(
            color=TEXT_COLOR,
            size=12,
        ),
        plot_bgcolor="rgba(255,255,255,0.5)",
        paper_bgcolor="white",
    )

    return fig


def draw_rectangle(
    width,
    height,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a rectangular section.
    """

    fig = go.Figure()

    # Rectangle
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=width,
        y1=height,
        line=dict(
            width=2,
            color=SECTION_LINE_COLOR,
        ),
    )

    # Centroid
    if centroid_x is not None and centroid_y is not None:

        fig.add_trace(
            go.Scatter(
                x=[centroid_x],
                y=[centroid_y],
                mode="markers+text",
                text=["C"],
                textposition="top right",
                marker=dict(
                    size=10,
                    color=CENTROID_COLOR,
                ),
                textfont=dict(
                    color=TEXT_COLOR,
                ),
                hovertemplate=(
                    "Centroid"
                    "<br>x = %{x:.4f}"
                    "<br>y = %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        # Centroidal x-axis
        fig.add_shape(
            type="line",
            x0=0,
            y0=centroid_y,
            x1=width,
            y1=centroid_y,
            line=dict(
                dash="dash",
                width=1,
                color=CENTROID_AXIS_COLOR,
            ),
        )

        # Centroidal y-axis
        fig.add_shape(
            type="line",
            x0=centroid_x,
            y0=0,
            x1=centroid_x,
            y1=height,
            line=dict(
                dash="dash",
                width=1,
                color=CENTROID_AXIS_COLOR,
            ),
        )

    padding = max(width, height) * 0.25

    fig.update_xaxes(
        range=[
            -padding,
            width + padding,
        ]
    )

    fig.update_yaxes(
        range=[
            -padding,
            height + padding,
        ]
    )

    dimension_offset = max(width, height) * 0.12
    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=width,
        y=0,
        offset=-dimension_offset,
        text=f"b = {width:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=height,
        x=width,
        offset=dimension_offset,
        text=f"h = {height:.2f}",
    )


    return _format_section_figure(
        fig,
        title="Rectangle Section",
    )


def draw_circle(
    radius,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a circular section.
    """

    fig = go.Figure()

    diameter = 2 * radius

    fig.add_shape(
        type="circle",
        x0=0,
        y0=0,
        x1=diameter,
        y1=diameter,
        line=dict(
            width=2,
            color=SECTION_LINE_COLOR,
        ),
    )

    if centroid_x is not None and centroid_y is not None:

        fig.add_trace(
            go.Scatter(
                x=[centroid_x],
                y=[centroid_y],
                mode="markers+text",
                text=["C"],
                textposition="top right",
                marker=dict(
                    size=10,
                    color=CENTROID_COLOR,
                ),
                textfont=dict(
                    color=TEXT_COLOR,
                ),
                hovertemplate=(
                    "Centroid"
                    "<br>x = %{x:.4f}"
                    "<br>y = %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_shape(
            type="line",
            x0=0,
            y0=centroid_y,
            x1=diameter,
            y1=centroid_y,
            line=dict(
                width=2,
                color=CENTROID_AXIS_COLOR,
            ),
        )

        fig.add_shape(
            type="line",
            x0=centroid_x,
            y0=0,
            x1=centroid_x,
            y1=diameter,
            line=dict(
                dash="dash",
                width=1,
                color=CENTROID_AXIS_COLOR,
            ),
        )

    padding = diameter * 0.15

    fig.update_xaxes(
        range=[
            -padding,
            diameter + padding,
        ]
    )

    fig.update_yaxes(
        range=[
            -padding,
            diameter + padding,
        ]
    )

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=diameter,
        y=0,
        offset=-diameter * 0.18,
        text=f"Ø = {diameter:.2f}",
    )

    return _format_section_figure(
        fig,
        title="Circular Section",
    )


def draw_i_section(
    depth,
    flange_width,
    flange_thickness,
    web_thickness,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a symmetric I / W section.
    """

    fig = go.Figure()

    web_height = (
        depth
        - 2 * flange_thickness
    )

    web_x0 = (
        flange_width
        - web_thickness
    ) / 2

    web_x1 = (
        flange_width
        + web_thickness
    ) / 2

    # Bottom flange
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=flange_width,
        y1=flange_thickness,
        line=dict(width=2,color=SECTION_LINE_COLOR,),
    )

    # Web
    fig.add_shape(
        type="rect",
        x0=web_x0,
        y0=flange_thickness,
        x1=web_x1,
        y1=flange_thickness + web_height,
        line=dict(width=2,color=SECTION_LINE_COLOR,),
    )

    # Top flange
    fig.add_shape(
        type="rect",
        x0=0,
        y0=depth - flange_thickness,
        x1=flange_width,
        y1=depth,
        line=dict(width=2,color=SECTION_LINE_COLOR,),
    )

    # Centroid
    if centroid_x is not None and centroid_y is not None:

        fig.add_trace(
            go.Scatter(
                x=[centroid_x],
                y=[centroid_y],
                mode="markers+text",
                text=["C"],
                textposition="top right",
                marker=dict(
                    size=10,
                    color=CENTROID_COLOR,
                ),
                textfont=dict(
                    color=TEXT_COLOR,
                ),
                hovertemplate=(
                    "Centroid"
                    "<br>x = %{x:.4f}"
                    "<br>y = %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_shape(
            type="line",
            x0=0,
            y0=centroid_y,
            x1=flange_width,
            y1=centroid_y,
            line=dict(
                dash="dash",
                width=1,
                color=CENTROID_AXIS_COLOR,
            ),
        )

        fig.add_shape(
            type="line",
            x0=centroid_x,
            y0=0,
            x1=centroid_x,
            y1=depth,
            line=dict(
                dash="dash",
                width=1,
                color=CENTROID_AXIS_COLOR,
            ),
        )

    padding = max(
        flange_width,
        depth,
    ) * 0.15

    fig.update_xaxes(
        range=[
            -padding,
            flange_width + padding,
        ]
    )

    fig.update_yaxes(
        range=[
            -padding,
            depth + padding,
        ]
    )
    dimension_offset = max(flange_width, depth) * 0.12
    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=flange_width,
        y=depth,
        offset=dimension_offset,
        text=f"bf = {flange_width:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=depth,
        x=flange_width,
        offset=dimension_offset,
        text=f"d = {depth:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=depth - flange_thickness,
        y1=depth,
        x=0,
        offset=-dimension_offset,
        text=f"tf = {flange_thickness:.2f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=web_x0,
        x1=web_x1,
        y=centroid_y,
        offset=dimension_offset * 0.6,
        text=f"tw = {web_thickness:.2f}",
    )

    return _format_section_figure(
        fig,
        title="I / W Section",
    )


def add_horizontal_dimension(
    fig,
    x0,
    x1,
    y,
    offset,
    text,
):
    """
    Add a horizontal dimension line.
    """

    dimension_y = y + offset

    # Extension lines
    fig.add_shape(
        type="line",
        x0=x0,
        y0=y,
        x1=x0,
        y1=dimension_y,
        line=dict(
            color=DIMENSION_COLOR,
            width=1,
        ),
    )

    fig.add_shape(
        type="line",
        x0=x1,
        y0=y,
        x1=x1,
        y1=dimension_y,
        line=dict(
            color=DIMENSION_COLOR,
            width=1,
        ),
    )

    # Dimension line
    fig.add_shape(
        type="line",
        x0=x0,
        y0=dimension_y,
        x1=x1,
        y1=dimension_y,
        line=dict(
            color=DIMENSION_COLOR,
            width=1,
        ),
    )

    # Arrows + text
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=dimension_y,
        text=text,
        showarrow=False,
        font=dict(
            color=DIMENSION_TEXT_COLOR,
            size=13,
        ),
        yshift=12,
    )

    fig.add_annotation(
        x=x0,
        y=dimension_y,
        ax=20,
        ay=0,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=DIMENSION_COLOR,
    )

    fig.add_annotation(
        x=x1,
        y=dimension_y,
        ax=-20,
        ay=0,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=DIMENSION_COLOR,
    )

def add_vertical_dimension(
    fig,
    y0,
    y1,
    x,
    offset,
    text,
):
    """
    Add a vertical dimension line.
    """

    dimension_x = x + offset

    # Extension lines
    fig.add_shape(
        type="line",
        x0=x,
        y0=y0,
        x1=dimension_x,
        y1=y0,
        line=dict(
            color=DIMENSION_COLOR,
            width=1,
        ),
    )

    fig.add_shape(
        type="line",
        x0=x,
        y0=y1,
        x1=dimension_x,
        y1=y1,
        line=dict(
            color=DIMENSION_COLOR,
            width=1,
        ),
    )

    # Dimension line
    fig.add_shape(
        type="line",
        x0=dimension_x,
        y0=y0,
        x1=dimension_x,
        y1=y1,
        line=dict(
            color=DIMENSION_COLOR,
            width=1,
        ),
    )

    # Text
    fig.add_annotation(
        x=dimension_x,
        y=(y0 + y1) / 2,
        text=text,
        showarrow=False,
        textangle=-90,
        font=dict(
            color=DIMENSION_TEXT_COLOR,
            size=13,
        ),
        xshift=14,
    )

    # Arrows
    fig.add_annotation(
        x=dimension_x,
        y=y0,
        ax=0,
        ay=-20,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=DIMENSION_COLOR,
    )

    fig.add_annotation(
        x=dimension_x,
        y=y1,
        ax=0,
        ay=20,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=DIMENSION_COLOR,
    )


# =============================================================
# AISC Database Shapes
# =============================================================


def _add_centroid_marker(fig, centroid_x, centroid_y, x_range, y_range):
    """
    Add a centroid marker with crosshair axes to a figure.
    """

    fig.add_trace(
        go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode="markers+text",
            text=["C"],
            textposition="top right",
            marker=dict(
                size=10,
                color=CENTROID_COLOR,
            ),
            textfont=dict(
                color=TEXT_COLOR,
            ),
            hovertemplate=(
                "Centroid"
                "<br>x = %{x:.4f}"
                "<br>y = %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=centroid_y,
        x1=x_range[1],
        y1=centroid_y,
        line=dict(
            dash="dash",
            width=1,
            color=CENTROID_AXIS_COLOR,
        ),
    )

    fig.add_shape(
        type="line",
        x0=centroid_x,
        y0=y_range[0],
        x1=centroid_x,
        y1=y_range[1],
        line=dict(
            dash="dash",
            width=1,
            color=CENTROID_AXIS_COLOR,
        ),
    )


def draw_channel(
    depth,
    flange_width,
    flange_thickness,
    web_thickness,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a channel (C / MC) section.

    Web at x=0, flanges extending in +x.
    """

    fig = go.Figure()

    # Bottom flange
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=flange_width,
        y1=flange_thickness,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    # Top flange
    fig.add_shape(
        type="rect",
        x0=0,
        y0=depth - flange_thickness,
        x1=flange_width,
        y1=depth,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    # Web
    fig.add_shape(
        type="rect",
        x0=0,
        y0=flange_thickness,
        x1=web_thickness,
        y1=depth - flange_thickness,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    padding = max(flange_width, depth) * 0.25

    fig.update_xaxes(range=[-padding, flange_width + padding])
    fig.update_yaxes(range=[-padding, depth + padding])

    if centroid_x is not None and centroid_y is not None:
        _add_centroid_marker(
            fig,
            centroid_x,
            centroid_y,
            x_range=[0, flange_width],
            y_range=[0, depth],
        )

    dimension_offset = max(flange_width, depth) * 0.12

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=flange_width,
        y=0,
        offset=-dimension_offset,
        text=f"bf = {flange_width:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=depth,
        x=flange_width,
        offset=dimension_offset * 1.8,
        text=f"d = {depth:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=depth - flange_thickness,
        y1=depth,
        x=0,
        offset=-dimension_offset,
        text=f"tf = {flange_thickness:.3f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=web_thickness,
        y=depth / 2,
        offset=dimension_offset * 0.6,
        text=f"tw = {web_thickness:.3f}",
    )

    return _format_section_figure(fig, title="Channel Section")


def draw_tee(
    depth,
    flange_width,
    flange_thickness,
    web_thickness,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a structural tee (WT / MT / ST) section.

    Flange at top, stem hanging down.
    """

    fig = go.Figure()

    stem_x0 = (flange_width - web_thickness) / 2
    stem_x1 = (flange_width + web_thickness) / 2

    # Flange
    fig.add_shape(
        type="rect",
        x0=0,
        y0=depth - flange_thickness,
        x1=flange_width,
        y1=depth,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    # Stem
    fig.add_shape(
        type="rect",
        x0=stem_x0,
        y0=0,
        x1=stem_x1,
        y1=depth - flange_thickness,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    padding = max(flange_width, depth) * 0.25

    fig.update_xaxes(range=[-padding, flange_width + padding])
    fig.update_yaxes(range=[-padding, depth + padding])

    if centroid_x is not None and centroid_y is not None:
        _add_centroid_marker(
            fig,
            centroid_x,
            centroid_y,
            x_range=[0, flange_width],
            y_range=[0, depth],
        )

    dimension_offset = max(flange_width, depth) * 0.12

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=flange_width,
        y=depth,
        offset=dimension_offset,
        text=f"bf = {flange_width:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=depth,
        x=flange_width,
        offset=dimension_offset * 1.8,
        text=f"d = {depth:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=depth - flange_thickness,
        y1=depth,
        x=0,
        offset=-dimension_offset,
        text=f"tf = {flange_thickness:.3f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=stem_x0,
        x1=stem_x1,
        y=0,
        offset=-dimension_offset,
        text=f"tw = {web_thickness:.3f}",
    )

    return _format_section_figure(fig, title="Structural Tee Section")


def draw_angle(
    depth,
    width,
    thickness,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a single angle (L) section.

    Vertical leg along the y-axis, horizontal leg along the x-axis,
    right-angle corner at the origin.
    """

    fig = go.Figure()

    # Vertical leg
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=thickness,
        y1=depth,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    # Horizontal leg
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=width,
        y1=thickness,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    padding = max(depth, width) * 0.25

    fig.update_xaxes(range=[-padding, width + padding])
    fig.update_yaxes(range=[-padding, depth + padding])

    if centroid_x is not None and centroid_y is not None:
        _add_centroid_marker(
            fig,
            centroid_x,
            centroid_y,
            x_range=[0, width],
            y_range=[0, depth],
        )

    dimension_offset = max(depth, width) * 0.12

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=width,
        y=0,
        offset=-dimension_offset,
        text=f"b = {width:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=depth,
        x=0,
        offset=-dimension_offset * 1.6,
        text=f"d = {depth:.2f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=thickness,
        y=depth,
        offset=dimension_offset,
        text=f"t = {thickness:.3f}",
    )

    return _format_section_figure(fig, title="Single Angle Section")


def draw_double_angle(
    depth,
    width,
    thickness,
    gap=0.375,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a double angle (2L) section, back-to-back.

    Vertical legs are separated by `gap`, horizontal legs point outward.
    Symmetric about x = 0.
    """

    fig = go.Figure()

    half_gap = gap / 2

    # Right angle
    fig.add_shape(
        type="rect",
        x0=half_gap,
        y0=0,
        x1=half_gap + thickness,
        y1=depth,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    fig.add_shape(
        type="rect",
        x0=half_gap,
        y0=0,
        x1=half_gap + width,
        y1=thickness,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    # Left angle (mirrored)
    fig.add_shape(
        type="rect",
        x0=-half_gap - thickness,
        y0=0,
        x1=-half_gap,
        y1=depth,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    fig.add_shape(
        type="rect",
        x0=-half_gap - width,
        y0=0,
        x1=-half_gap,
        y1=thickness,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    half_total_width = half_gap + width
    padding = max(half_total_width * 2, depth) * 0.2

    fig.update_xaxes(
        range=[-half_total_width - padding, half_total_width + padding]
    )
    fig.update_yaxes(range=[-padding, depth + padding])

    if centroid_x is not None and centroid_y is not None:
        _add_centroid_marker(
            fig,
            centroid_x,
            centroid_y,
            x_range=[-half_total_width, half_total_width],
            y_range=[0, depth],
        )

    dimension_offset = max(half_total_width * 2, depth) * 0.1

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=depth,
        x=half_total_width,
        offset=dimension_offset,
        text=f"d = {depth:.2f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=half_gap,
        x1=half_gap + width,
        y=0,
        offset=-dimension_offset,
        text=f"b = {width:.2f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=half_gap,
        x1=half_gap + thickness,
        y=depth,
        offset=dimension_offset,
        text=f"t = {thickness:.3f}",
    )

    if gap > 0:

        add_horizontal_dimension(
            fig=fig,
            x0=-half_gap,
            x1=half_gap,
            y=depth / 2,
            offset=0,
            text=f"gap = {gap:.3f}",
        )

    return _format_section_figure(fig, title="Double Angle Section")


def draw_hss_rect(
    height,
    width,
    thickness,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a rectangular / square HSS section (hollow tube).
    """

    fig = go.Figure()

    # Outer profile
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=width,
        y1=height,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    # Inner profile (wall hole)
    fig.add_shape(
        type="rect",
        x0=thickness,
        y0=thickness,
        x1=width - thickness,
        y1=height - thickness,
        line=dict(width=1, dash="dot", color=SECTION_LINE_COLOR),
    )

    padding = max(width, height) * 0.25

    fig.update_xaxes(range=[-padding, width + padding])
    fig.update_yaxes(range=[-padding, height + padding])

    if centroid_x is not None and centroid_y is not None:
        _add_centroid_marker(
            fig,
            centroid_x,
            centroid_y,
            x_range=[0, width],
            y_range=[0, height],
        )

    dimension_offset = max(width, height) * 0.12

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=width,
        y=0,
        offset=-dimension_offset,
        text=f"b = {width:.2f}",
    )

    add_vertical_dimension(
        fig=fig,
        y0=0,
        y1=height,
        x=width,
        offset=dimension_offset,
        text=f"h = {height:.2f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=thickness,
        y=height,
        offset=dimension_offset,
        text=f"t = {thickness:.3f}",
    )

    return _format_section_figure(fig, title="Rectangular HSS Section")


def draw_hss_round(
    outer_diameter,
    inner_diameter,
    centroid_x=None,
    centroid_y=None,
):
    """
    Draw a round HSS / pipe section (hollow tube).
    """

    fig = go.Figure()

    fig.add_shape(
        type="circle",
        x0=0,
        y0=0,
        x1=outer_diameter,
        y1=outer_diameter,
        line=dict(width=2, color=SECTION_LINE_COLOR),
    )

    inner_offset = (outer_diameter - inner_diameter) / 2

    fig.add_shape(
        type="circle",
        x0=inner_offset,
        y0=inner_offset,
        x1=inner_offset + inner_diameter,
        y1=inner_offset + inner_diameter,
        line=dict(width=1, dash="dot", color=SECTION_LINE_COLOR),
    )

    padding = outer_diameter * 0.15

    fig.update_xaxes(range=[-padding, outer_diameter + padding])
    fig.update_yaxes(range=[-padding, outer_diameter + padding])

    if centroid_x is not None and centroid_y is not None:
        _add_centroid_marker(
            fig,
            centroid_x,
            centroid_y,
            x_range=[0, outer_diameter],
            y_range=[0, outer_diameter],
        )

    add_horizontal_dimension(
        fig=fig,
        x0=0,
        x1=outer_diameter,
        y=0,
        offset=-outer_diameter * 0.18,
        text=f"OD = {outer_diameter:.2f}",
    )

    add_horizontal_dimension(
        fig=fig,
        x0=inner_offset,
        x1=inner_offset + inner_diameter,
        y=outer_diameter / 2,
        offset=0,
        text=f"ID = {inner_diameter:.3f}",
    )

    return _format_section_figure(fig, title="Round HSS Section")


PANEL_KEYS = {
    "w_m_s_hp": ["A", "wt./ft.", "Ix", "Sx", "rx", "Zx", "Iy", "Sy", "ry", "Zy", "J", "Cw"],
    "c_mc": ["A", "wt./ft.", "Ix", "Sx", "rx", "Zx", "Iy", "Sy", "ry", "x(bar)", "J", "Cw"],
    "wt_mt_st": ["A", "wt./ft.", "Ix", "Sx", "rx", "y(bar)", "Zx", "Iy", "Sy", "ry", "Zy", "J"],
    "single_angles": ["A", "wt./ft.", "Ix", "Sx", "rx", "Iy", "Sy", "ry", "Iz", "rz", "J"],
    "double_angles": ["A", "wt./ft.", "Ix", "Sx", "rx", "y(bar)", "ry(3/8)"],
    "hss_rect": ["A", "wt./ft.", "Ix", "Sx", "rx", "Zx", "Iy", "Sy", "ry", "Zy", "J", "C"],
    "hss_round": ["A", "wt./ft.", "Ix", "Sx", "rx", "Zx", "J", "C"],
}


def _format_property_value(value):

    if isinstance(value, bool) or value is None:
        return str(value)

    if isinstance(value, (int, float)):
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:.3f}"

    return str(value)


def _add_properties_panel(fig, properties, keys):
    """
    Add a text panel outside the plot area (right side) listing the
    given properties, and widen the right margin to make room for it.
    """

    lines = []

    for key in keys:

        value = properties.get(key)

        if value is None:
            continue

        lines.append(f"{key} = {_format_property_value(value)}")

    if not lines:
        return

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text="<br>".join(lines),
        font=dict(size=12, color=TEXT_COLOR),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=SECTION_LINE_COLOR,
        borderwidth=1,
        borderpad=10,
    )

    fig.update_layout(margin=dict(r=230))


def draw_aisc_shape(family_key, properties, gap=0.375, shape_name=None):
    """
    Draw an AISC database shape from its family key and property dict
    (as loaded from data/aisc_sections/*.json).
    """

    if family_key == "w_m_s_hp":

        fig = draw_i_section(
            depth=properties["d"],
            flange_width=properties["bf"],
            flange_thickness=properties["tf"],
            web_thickness=properties["tw"],
            centroid_x=properties["bf"] / 2,
            centroid_y=properties["d"] / 2,
        )

    elif family_key == "c_mc":

        fig = draw_channel(
            depth=properties["d"],
            flange_width=properties["bf"],
            flange_thickness=properties["tf"],
            web_thickness=properties["tw"],
            centroid_x=properties["x(bar)"],
            centroid_y=properties["d"] / 2,
        )

    elif family_key == "wt_mt_st":

        fig = draw_tee(
            depth=properties["d"],
            flange_width=properties["bf"],
            flange_thickness=properties["tf"],
            web_thickness=properties["tw"],
            centroid_x=properties["bf"] / 2,
            centroid_y=properties["d"] - properties["y(bar)"],
        )

    elif family_key == "single_angles":

        fig = draw_angle(
            depth=properties["d"],
            width=properties["b"],
            thickness=properties["t"],
            centroid_x=properties["x(bar)"],
            centroid_y=properties["y(bar)"],
        )

    elif family_key == "double_angles":

        fig = draw_double_angle(
            depth=properties["d"],
            width=properties["b"],
            thickness=properties["t"],
            gap=gap,
            centroid_x=0,
            centroid_y=properties["y(bar)"],
        )

    elif family_key == "hss_rect":

        fig = draw_hss_rect(
            height=properties["h"],
            width=properties["b"],
            thickness=properties["t(des)"],
            centroid_x=properties["b"] / 2,
            centroid_y=properties["h"] / 2,
        )

    elif family_key == "hss_round":

        fig = draw_hss_round(
            outer_diameter=properties["OD"],
            inner_diameter=properties["ID"],
            centroid_x=properties["OD"] / 2,
            centroid_y=properties["OD"] / 2,
        )

    else:
        raise ValueError(f"Unknown AISC shape family: {family_key}")

    if shape_name:
        fig.update_layout(
            title=dict(
                text=shape_name,
                font=dict(size=18, color=TEXT_COLOR),
            ),
        )

    _add_properties_panel(fig, properties, PANEL_KEYS.get(family_key, []))

    return fig