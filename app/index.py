import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

from app import app
from apps import app1,app2
# from layouts import layout1, layout2
# import callbacks

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Recognition", href="/recognition")),
        dbc.NavItem(dbc.NavLink("Prediction", href="/prediction")),
    ],
    brand="Functionality Learning",
    brand_href="/",
    color="primary",
    dark=True,
)

cards = dbc.Row(dbc.Col(dbc.CardDeck(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Recognition", className="card-title"),
                    html.P(
                        [
                            "Input: RGBD scene image with humans",
                            html.Br(),
                            "Output: Word classes and corresponding regions"
                        ],
                        className="card-text",
                    ),
                    dbc.Button(
                        "Click here", color="success", className="mt-auto",
                        href='/recognition'
                    ),
                ]
            )
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Prediction", className="card-title"),
                    html.P(
                        [
                            "Input: RGBD scene image without human, Activity label",
                            html.Br(),
                            "Output: Activity map",
                        ],
                        className="card-text",
                    ),
                    dbc.Button(
                        "Click here", color="success", className="mt-auto",
                        href='/prediction'
                    ),
                ]
            )
        )
    ]
),width={'size': 6, 'offset': 3}),style={'margin-top': '10px'})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/recognition':
        return app1.layout
    elif pathname == '/prediction':
        return app2.layout
    elif pathname == '/':
        return cards
    else:
        return dbc.Alert("404: Page Not Found", color="warning"),

if __name__ == '__main__':
    app.run_server(debug=False)