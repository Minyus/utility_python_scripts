# -*- coding: utf-8 -*-

%%time 

import numpy as np
import pandas as pd
import os
import math
import itertools
import re
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import sys


pd.options.display.max_rows = 8
pd.options.display.max_columns = 100

path = r'C:\Users\U\Documents'
#os.chdir(path.replace(os.sep, '/'))
print(f'chdir: {os.getcwd()}')
d01_df = pd.read_csv('file.csv')
