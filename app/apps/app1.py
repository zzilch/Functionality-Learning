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
from .recognition import recognize,CLASSES
from .utils import vis_heatmap,bar

from app import app

layout = html.Div([
    # title
    html.H1(
        'Activity Recognition App',
        style={'textAlign': 'center'}
    ),
    header,
    # class buttons
    html.Div(
        id='class-buttons',
        style={'textAlign': 'center'}
    ),
    # prm visualization
    html.Div(
        id='vis-prm',
        style={'textAlign': 'center'}
    ),
    # prm objects
    html.Div(
        id='objs-prm',
        style={'textAlign': 'center'}
    )    
])

@app.callback(
    [Output('class-buttons', 'children'),
     Output('objs-prm', 'children')],
    [Input('upload-image', 'contents')]
)
def vis_class(img_url):
    if img_url is None:
        return None, None
    pil_img = image_string_to_PILImage(img_url)
    img = resize(np.array(pil_img), (224, 224))
    rgb = img[..., :3]
    d = img[..., 3]
    pred = recognize(pil_img)
    if pred is None:
        return dbc.Row(dbc.Col(dbc.Alert("Found no class instance", color="warning"), width={'size': 6, 'offset': 3})), None

    ys, ps, prms = pred
    prms = [array_to_data_url(img_as_ubyte(
        vis_heatmap(prm, rgb, d))) for prm in prms]
    return dbc.ButtonGroup([dbc.Button("Class:", color='link', disabled=True)]+[
        dbc.Button(
            f'{CLASSES[y]}',
            id={'type': 'pred-class', 'index': i},
            color="secondary")
        for i, y in enumerate(ys)], style={'margin-top': '10px'}
    ), [
        html.Div([
            html.Img(src=prms[i]),
            html.Div([
                html.Div(
                    'p(I,Act):0.0',
                    style={'display': 'inline-block'}
                ),
                html.Img(src=bar),
                html.Div(
                    '1.0',
                    style={'display': 'inline-block', 'margin-right': '2em'}
                )
            ], style={'textAlign': 'center'})
        ], id={'type': 'pred-prm', 'index': i},
            style={'display': 'none'})
        for i, y in enumerate(ys)
    ]


@app.callback(
    Output('vis-prm', 'children'),
    [Input({'type': 'pred-class', 'index': ALL}, 'n_clicks')],
    [State({'type': 'pred-prm', 'index': ALL}, 'children')]
)
def on_select(n_clicks, prms):
    if len(n_clicks) == 0 or not any(n_clicks):
        return
    triggered_index = eval(
        dash.callback_context.triggered[0]['prop_id'].split('.')[0])['index']
    return html.Div(prms[triggered_index], style={'textAlign': 'center'})