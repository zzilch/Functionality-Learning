from skimage import io,transform,filters
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from multiprocessing import Pool,Manager

def get_prmo(pred):
    rid,wids = pred.split(' ')
    wids = wids.split(',')

    obj = transform.resize(io.imread(f'../../data/np/object/{rid}.objectId.np.encoded.png')[...,2],(224,224),mode='edge',anti_aliasing=False,anti_aliasing_sigma=None,preserve_range=True,order=0)
    obj_csv = pd.read_csv(f'../../data/np/object/{rid}.objectId.csv')

    excepts = [np.nan,'Wall','Floor','door','window','rug','curtain','arch','stairs','Ceiling','Ground','Box','plant','pet']
    save_dir = f'../../output/prmo/{rid}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for wid in wids:
        prm = io.imread(f'../../output/prm/{rid}/{rid}.{wid}.png')
        prm = prm/prm.max()
    
        oid = np.unique(obj[prm>prm[prm>0].mean()])
        oid = [ o for o in oid if obj_csv['category'][o] not in excepts]

        omask = np.isin(obj,oid)
        prmo = filters.gaussian(prm*0.5+0.5*omask*prm)
        prmo = prmo/prmo.max()

        save_path = f'{save_dir}/{rid}.{wid}.png'
        io.imsave(save_path,prmo)

    return 1

if __name__ == '__main__':
    preds = open('../../output/prm.txt').read().split('\n')
    print(len(preds))
    with Manager() as manager:
        with Pool() as pool:
            bar = tqdm(total=len(preds))
            [pool.apply_async(get_prmo,args=(pred,),callback=bar.update) for pred in preds]
            pool.close()
            pool.join()
            bar.close()
