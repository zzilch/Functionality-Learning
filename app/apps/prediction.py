import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from .model import Conditional_DeepLab_ResNet50
from app import PRED_MODEL_PATH,DEVICE

ACTIVITY = ['cook',
 'cook kitchen',
 'cook meal',
 'eat',
 'eat meal',
 'have conversation',
 'lie',
 'lie bed',
 'lie sofa',
 'listen music',
 'look',
 'look computer',
 'look door',
 'look wall',
 'look window',
 'play chess',
 'play game',
 'play music',
 'rest',
 'rest bed',
 'rest chair',
 'rest sofa',
 'sit',
 'sit bed',
 'sit chair',
 'sit desk',
 'sit music',
 'sit sofa',
 'sit table',
 'stand',
 'stand cabinet',
 'stand desk',
 'stand door',
 'stand kitchen',
 'stand sink',
 'stand table',
 'talk',
 'use computer',
 'use kitchen',
 'wait',
 'wash sink',
 'watch',
 'watch television',
 'work']

VALID_ACTIVITY = [
    196,
    141,
    686,
    49,
    178,
    766,
    1121,
    1682,
    1633,
    99,
    72,
    49,
    33,
    42,
    36,
    59,
    34,
    129,
    417,
    101,
    45,
    505,
    1107,
    77,
    2193,
    1142,
    50,
    2024,
    565,
    1616,
    106,
    60,
    82,
    525,
    197,
    98,
    222,
    1526,
    51,
    24,
    75,
    40,
    1138,
    216
]

transform = T.Compose([
    T.Resize(224,interpolation=Image.NEAREST),
    T.ToTensor()
])

pred_model = Conditional_DeepLab_ResNet50(4,44,pretrained=False,aux_loss=True)
pred_model.load_state_dict(torch.load(PRED_MODEL_PATH))
pred_model.eval().to(DEVICE)
print('Prediction model run on:', DEVICE)

def predict(img,act):
    with torch.no_grad():
        img = transform(img).unsqueeze(0).to(DEVICE)
        act = torch.tensor(act).long().unsqueeze(0).to(DEVICE)
        pred_model.to(DEVICE)
        output = pred_model(img,act)['out'].sigmoid().squeeze(0).cpu().numpy()
    return output




# The following are imported in app: 
#   >> predict, model, transform, CLASSES, DEVICE

# import os
# import matplotlib.pyplot as plt
# test_imgs = os.listdir('img')
# act = 2
# img = Image.open(f'img/{test_imgs[0]}')
# output = predict(img,act,model,transform)
# plt.imshow(output)
# plt.title(CLASSES[act])
