import numpy as np
import plotly.graph_objects as go

SECTION_LINE_COLOR = "#EAEAEA"
CENTROID_AXIS_COLOR = "#9CA3AF"
CENTROID_COLOR = "#4FC3F7"
TEXT_COLOR = "#EAEAEA"

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
        title=title,
        showlegend=False,
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
        font=dict(
            color=TEXT_COLOR,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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

    padding = max(width, height) * 0.15

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
                marker=dict(size=10),
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
                color=SECTION_LINE_COLOR,
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
                color=SECTION_LINE_COLOR,
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
                marker=dict(size=10),
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
                color=SECTION_LINE_COLOR,
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
                color=SECTION_LINE_COLOR,
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

    return _format_section_figure(
        fig,
        title="I / W Section",
    )