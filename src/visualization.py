import cv2
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import base64


# plotly.py helper functions
def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title_text=None):
    img_width, img_height = im.size
    fig = go.Figure()

    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[img_width, img_width],
            y=[img_height, img_height],
            showlegend=False,
            mode="markers",
            marker_opacity=0,
            hoverinfo="none",
            legendgroup="Image",
        )
    )

    fig.update_layout(height=500, width=400)

    fig.add_layout_image(
        dict(
            source=pil_to_b64(im),
            sizing=None,  # "stretch",
            opacity=1,
            layer="below",
            x=0,
            y=0,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
        )
    )

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width]
    )

    fig.update_yaxes(
        showgrid=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        range=[img_height, 0],
    )

    title = {
        'text': title_text,
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'}

    fig.update_layout(title=title, showlegend=showlegend)

    return fig


def draw_objects(image, title_text):
    print("starting to draw images")

    img = Image.fromarray(image)
    fig = pil_to_fig(img, showlegend=False, title_text=title_text)

    return fig
