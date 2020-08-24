import dash
import torch
import dash_bootstrap_components as dbc

CLS_MODEL_PATH = '../models/cls_model.ckpt'
SEG_MODEL_PATH = '../models/seg_model.ckpt'
PRED_MODEL_PATH = '../models/pred_model.ckpt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
server = app.server

