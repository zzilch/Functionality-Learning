import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from .utils import parse_rgbd

from app import app

header = html.Div([
    # upload
    dbc.Row(dbc.Col(dcc.Upload(html.Div(
        [
            'Drag and Drop or ',
            html.A('Select An Image',href="#",className="alert-link")
        ]),
        id='upload-image',
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
        },
        multiple=False),
        width={'size': 6, 'offset': 3}
    )),
    # rgbd visualization
    html.Div(id='vis-rgbd'),
    # rgbd object
    html.Div(
        id='obj-rgbd',
        style={'display': 'none'}
    )
])

@app.callback(
    Output('obj-rgbd', 'children'),
    [Input('upload-image', 'contents')]
)
def on_upload(content):
    return content

@app.callback(
    Output('vis-rgbd', 'children'),
    [Input('obj-rgbd', 'children')]
)
def vis_rgbd(content):
    if content is None:
        return dbc.Row(dbc.Col(dbc.Alert(
            "Please upload an RGB-D image with 4 channels", 
            color="info"), 
            width={'size': 6, 'offset': 3}))
    return parse_rgbd(content)