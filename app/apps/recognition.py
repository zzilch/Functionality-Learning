import numpy as np
import torch
from PIL import Image
from skimage import filters
import torchvision.transforms as T

from .model import ResNet50
from .prm import peak_response_mapping
from .model import DeepLab_ResNet50

from app import CLS_MODEL_PATH,SEG_MODEL_PATH,DEVICE


CLASSES = ['cook',
 'eat',
 'have',
 'lie',
 'listen',
 'look',
 'play',
 'rest',
 'sit',
 'stand',
 'talk',
 'use',
 'wait',
 'wash',
 'watch',
 'work',
 'bed',
 'cabinet',
 'chair',
 'chess',
 'computer',
 'conversation',
 'desk',
 'door',
 'game',
 'kitchen',
 'meal',
 'music',
 'sink',
 'sofa',
 'table',
 'television',
 'wall',
 'window']

CATEGORY = ['empty',
 'Wall',
 'Floor',
 'stairs',
 'door',
 'window',
 'chair',
 'wardrobe_cabinet',
 'table',
 'shelving',
 'indoor_lamp',
 'sofa',
 'kitchen_cabinet',
 'desk',
 'stand',
 'bed',
 'tv_stand',
 'toy',
 'ottoman',
 'dresser',
 'sink',
 'plant',
 'kitchenware',
 'kitchen_appliance',
 'workplace',
 'hanging_kitchen_cabinet',
 'mirror',
 'vase',
 'bench_chair',
 'fireplace',
 'kitchen_set',
 'household_appliance',
 'person',
 'curtain',
 'hanger',
 'shoes_cabinet',
 'pet',
 'partition',
 'music',
 'gym_equipment',
 'table_and_chair',
 'dressing_table',
 'clock',
 'candle',
 'television',
 'picture_frame',
 'recreation',
 'rug',
 'pillow',
 'trash_can',
 'fan',
 'column',
 'books',
 'arch',
 'switch',
 'heater',
 'computer',
 'air_conditioner',
 'whiteboard',
 'others']


transform = T.Compose([
    T.Resize(224,interpolation=Image.NEAREST),
    T.ToTensor()
])

cls_model = peak_response_mapping(ResNet50(4,34,pretrained=False),win_size=5)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH))
cls_model.inference()
cls_model.to(DEVICE)
print('Classification model run on:', DEVICE)

seg_model = DeepLab_ResNet50(4,60,pretrained=False,aux_loss=True)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH))
seg_model.eval()
seg_model.to(DEVICE)

def classify(img):
    img = transform(img).unsqueeze(0).to(DEVICE)
    cls_model.to(DEVICE)
    output = cls_model(img,class_threshold=6, peak_threshold=6)
    if output is not None:
        aggregation, class_response_maps, valid_peak_list, peak_response_maps = output
        y_peak = torch.unique(valid_peak_list[:,1])
        prob = aggregation[0,y_peak].sigmoid().cpu().numpy()
        prms = []
        for y in y_peak:
            peaks = [valid_peak_list[:,1]==y]
            prm = peak_response_maps[peaks].sum(0)
            prm = prm/prm.max()
            prms.append(prm.cpu().numpy())
        cls_model.zero_grad()
        return [y_peak,prob,prms]
    else:
        return None

def segment(img):
    with torch.no_grad():
        img = transform(img).unsqueeze(0).to(DEVICE)
        seg_model.to(DEVICE)
        output = seg_model(img)['out'].argmax(1).squeeze()
        output = output.cpu().numpy()
        return output

def recognize(img):
    cls_result = classify(img)
    if cls_result is None: return None
    seg_result = segment(img)
    excepts = [np.nan,1,2,4,5,10,47,33,53,3,59,21,36]
    for i in range(len(cls_result[-1])):
        prm = cls_result[-1][i]
        oid = np.unique(seg_result[prm>prm[prm>0].mean()])
        oid = [ o for o in oid if o not in excepts]
        omask = np.isin(seg_result,oid)
        prmo = filters.gaussian(prm*0.5+0.5*omask*prm)
        prmo = prmo/prmo.max()
        cls_result[-1][i] = prmo
    return cls_result


