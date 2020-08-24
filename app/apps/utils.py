import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte,filters
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage

import dash_html_components as html
import dash_bootstrap_components as dbc

# heatmap visualization
def vis_heatmap(pred, rgb, d):
    pred = filters.gaussian(pred)
    pred[0, 0] = 1.0
    cmap_h = plt.cm.hot_r
    h = cmap_h(pred)

    cmap_d = plt.cm.gray
    d_img = cmap_d(d)

    ret = 0.5*d_img+0.5*h
    mask = (rgb != 0).all(-1)
    ret[mask,-1] = 1.0
    ret[~mask,-1] = 0.0
    return ret

# color bar for heatmap
def get_colorbar():
    cmap_h = plt.cm.hot_r
    bar = np.zeros((5, 224))
    for i in range(224):
        bar[:, i] = i/224
    bar = array_to_data_url(img_as_ubyte(cmap_h(bar)))
    return bar
bar = get_colorbar()

# show uploaded rgb/d image
def parse_rgbd(content):
    pil_img = image_string_to_PILImage(content)
    img = np.array(pil_img)
    if img.shape[-1] != 4:
        return dbc.Row(dbc.Col(dbc.Alert("Please upload an RGB-D image with 4 channels", color="warning"), width={'size': 6, 'offset': 3})), None

    rgb = img[..., :3]
    d = img[..., 3]
    mask = (rgb != 0).all(-1)
    d[mask] = (255-d[mask])
    
    rgb = array_to_data_url(rgb)
    d = array_to_data_url(d)

    return html.Div([html.Div(
        [
            html.Div(
                'RGB', 
                style={
                    'verticalAlign': 'middle', 'textAlign': 'center'
                }
            ),
            html.Img(
                src=rgb, 
                width=224, 
                height=224)
        ], 
        style={
            'display': 'inline-block'
        }),
        html.Div([
            html.Div(
                'Depth', 
                style={
                    'verticalAlign':'middle',
                    'textAlign': 'center'
            }),
            html.Img(
                src=d, 
                width=224, 
                height=224
            )], 
        style={
            'display': 'inline-block'
        })
    ], style={'textAlign': 'center'})