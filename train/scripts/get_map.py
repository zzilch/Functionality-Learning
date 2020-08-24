from skimage import io,transform,filters
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from multiprocessing import Pool,Manager

def get_act(line):
    rid,pid,words = line.split(' ')
    words = words.split(',')

    if len(words)==1:
        act = io.imread(f'../../output/prmo/{rid}/{rid}.{words[0]}.png')/255.0
    else:
        act1 = io.imread(f'../../output/prmo/{rid}/{rid}.{words[0]}.png')/255.0
        act2 = io.imread(f'../../output/prmo/{rid}/{rid}.{words[1]}.png')/255.0
        act = np.sqrt(act1*act2)
        act = act/act.max()
    
    save_dir = f'../../output/act/{rid}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/{rid}.{pid}.png'
    io.imsave(save_path,act)
    
    return 1

if __name__ == '__main__':
    lines = open('../../output/maps.txt').read().split('\n')
    print(len(lines))
    with Manager() as manager:
        with Pool() as pool:
            bar = tqdm(total=len(lines))
            [pool.apply_async(get_act,args=(line,),callback=bar.update) for line in lines]
            pool.close()
            pool.join()
            bar.close()
