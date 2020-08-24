import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL

import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage

from .header import header
from .prediction import predict,ACTIVITY,VALID_ACTIVITY
from .utils import vis_heatmap,bar

from app import app,DEVICE

layout = html.Div([
    # title
    html.H1(
        'Activity Prediction App',
        style={'textAlign': 'center'}
    ),
    header,
    # activity dropdown
    dbc.Row([
        dbc.Col(
            dbc.Button(
                'Activity:',
                color='link',
                disabled=True
            ),
            width={'size': 1, 'offset': 3},
            style={'padding-right':0}
        ),
        dbc.Col(html.Div(
            dcc.Dropdown(
                id='action-dropdown',
                options=[{'label':act,'value':i} for i,act in enumerate(ACTIVITY) if VALID_ACTIVITY[i]>500],
                placeholder="Select an action",
                style={'textAlign': 'center'}
            )),
            width={'size': 4},
            style={'padding-left':0,'padding-right':0}
        ),
        dbc.Col(
            dbc.Button(
                'Submit',
                id='submit-button-state',
                color='primary'
            ),
            width={'size': 1},
            style={'padding-left':0}
        )
    ], style={'margin-top': '10px'}),
    html.Div(
        id='vis-map',
        style={'textAlign': 'center'}
    ),
])

@app.callback(
    Output('vis-map', 'children'),
    [Input('submit-button-state', 'n_clicks')],
    [State('obj-rgbd', 'children'),
    State('action-dropdown', 'value')]
)
def vis_map(n_clicks, img_url, act):
    if img_url is None or act is None:
        return dbc.Row(dbc.Col(dbc.Alert(
            "Please select an activity.", 
            color="info"), 
            width={'size': 6, 'offset': 3}))

    pil_img = image_string_to_PILImage(img_url)
    img = resize(np.array(pil_img),(224,224))
    rgb = img[...,:3]
    d = img[...,3]

    pred = predict(pil_img,act)
    vis = array_to_data_url(img_as_ubyte(vis_heatmap(pred,rgb,d)))
    return html.Div([
        html.Img(src=vis),
        html.Div([
            html.Div(
                'p(I,Act):0.0',
                style={'display':'inline-block'}
            ),
            html.Img(src=bar),
            html.Div(
                '1.0',
                style={'display':'inline-block','margin-right':'2em'}
            )
        ],style={'textAlign': 'center'})
    ])