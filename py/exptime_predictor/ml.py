import numpy as np
import os
import pandas as pd
import sqlite3
import time
from datetime import date, timedelta
from astropy.time import Time

from obiwan.common import save_png,dobash
from exptime_predictor.data import main as data_main
# to make this notebook's output stable across runs
np.random.seed(7)



if __name__ == '__main__':
  df_train = data_main()
  




