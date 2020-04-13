from sklearn import tree
import pickle
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.cm as cmx
from matplotlib import colors as colors
import scipy.io as sio
import matplotlib as mpl
import numpy as np


def load_features(season):
    '''
    This function will load the input features and target. The input features have already been cleaned for NaNs or
    observations of significant wave heights/mean period that are less than 0.1 m/s.
    Inputs:

    season  :   String, options 'winter' or 'summer'
    '''



basic_cols=['#ff0000','k','#07076b']
my_cmap=colors.LinearSegmentedColormap.from_list('mycmap', basic_cols)

#Load the data. This data has already been cleaned - data gaps, either due to poor observations or gaps in wave forecasts
#have been removed. Choose either the winter dataset or the summer dataset.
#Visualize the time series before visualizing the feature space:
