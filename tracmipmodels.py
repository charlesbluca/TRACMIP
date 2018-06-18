# load modules
import numpy as np

def get_modelnames():
    modnames = ['AM2', 'CALTECH', 'CAM3', 'CAM4', 'CNRM', 'ECHAM6.1', \
                'ECHAM6.3', 'CAM5Nor', 'IPSL', 'MetCTL', 'MetENT', 'MIROC5', \
                'MPAS', 'GISS']
    return modnames[0:nmod]

def get_modelnumbers():
    modnumbers = ['3', '5', '6', '8', '9', '10', \
                  '11', '1', '2', '3', '12', '7']                
    return modnumbers[0:nmod] 

def get_modelsubplots():
    modsubplots = [3, 15, 4, 5, 7, 8, 9, 6, 10, 11, \
                   12, 13, 14]
    return modsubplots[0:nmod]    

def get_modelcolors12():
    # from color brewer: http://colorbrewer2.org/
    modcolors12 = np.array([(166,206,227), (31,120,180), (178,223,138), (51,160,44), \
              (251,154,153), (227,26,28) , (253,191,111), (255,127,0), \
              (202,178,214), (106,61,154), (255,255,153), (177,89,40)])/255
    return modcolors12[0:nmod]
    
def get_modelcolors(nmod):
    # from http://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
    colors = np.array([(240,163,255),(0,117,220),(153,63,0),(76,0,92),(25,25,25),    \
                   (0,92,49),(43,206,72),(255,204,153),(128,128,128),(148,255,181), \
                   (143,124,0),(157,204,0),(194,0,136),(0,51,128),(255,164,5),      \
                   (255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),    \
                   (224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),  \
                   (255,80,5)])/255
    return colors[0:nmod]                   