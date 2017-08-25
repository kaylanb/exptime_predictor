import numpy as np
import os
import pandas as pd
import sqlite3
import time
from datetime import date, timedelta

from obiwan.common import save_png,dobash

# to make this notebook's output stable across runs
np.random.seed(7)

# Where to save the figures
REPO_DIR= os.path.join(os.environ['HOME'],
                       'PhdStudent/Research/desi/ml_data/')
DB_DIR= os.path.join(REPO_DIR,'obsbot/obsdb')

class SqliteConnection(object):
  def pandas_fetchall(self,sqlite_fn, exe_cmd):
    print('Reading sqlite db: %s' % sqlite_fn)
    conn = sqlite3.connect(sqlite_fn)
    c= conn.cursor() 
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")   
    print('Has tables: ',c.fetchall())        
    print('executing query: %s' % exe_cmd)
    df = pd.read_sql_query(exe_cmd, conn)
    # cleanup
    c.close()
    conn.close()
    return df

  def fetchall(self,sqlite_fn, exe_cmd):
    print('Reading sqlite db: %s' % sqlite_fn)
    conn = sqlite3.connect(sqlite_fn)
    c= conn.cursor() 
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")   
    print('Has tables: ',c.fetchall())        
    print('executing query: %s' % exe_cmd)
    c.execute(exe_cmd)
    cols= [col[0] for col in c.description] 
    print('Table has cols',cols) 
    list_of_tuples= c.fetchall()
    # cleanup
    c.close()
    conn.close()
    return list_of_tuples

class Data(object):
  """Fetches and loads decam.sqlite3 data as Pandas df
  
  Args:
    repo_dir: path to obsbot repo, eg. $repo_dir/obsbot
  """

  def __init__(self, repo_dir):
    self.repo_dir= repo_dir
    self.Sql= SqliteConnection()

  def fetch_data(self):
    curr_path = os.getcwd()
    os.chdir( os.path.join(self.repo_dir, 'obsbot'))
    dobash('git pull origin master')
    os.chdir( curr_path)

  def load_data(self):
    """return Pandas DF of sqlite table"""
    fn = os.path.join(self.repo_dir,"obsbot/obsdb",
                      "decam.sqlite3")
    return self.Sql.pandas_fetchall(fn,
                                    "select * from obsdb_measuredccd")

def get_next_day(ymd):
  """given year month day string (ymd) return next calendar day
  
  Args:
      ymd: '20110531'
      
  Returns:
      ymd for next day: '20110601' for example above
  """
  t=time.strptime('20110531','%Y%m%d')
  newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(1)
  return newdate.strftime('%Y%m%d')

class Clean(object):
  def drop_nonreal_exposures(self,df):
    """remove flats, bias, other non-science exposures"""
    bands= df.loc[:,'band']
    df= df[(bands == 'g') |
           (bands == 'r') | 
           (bands == 'z') ]
    df= df[(df.loc[:,'obstype'] == 'object')]
    return df
   
  def add_night_obs(self,df):
    """assigns integer ymd to each instance

    Night observed begins in evening and continutes to morning of next
      day. Found that float hrminsec = 150000.0 cleanly separates morning
      from evening
    """
    t = Time(df.loc[:,'mjd_obs'], format='mjd')
    # t.iso is list of elements like '2016-02-25 05:11:31.769'
    day_str= np.array([a.split(' ')[0] for a in t.iso])
    hr_str= np.array([a.split(' ')[1] for a in t.iso])
    day= np.char.replace(day_str,'-','').astype(int) # integer ymd
    hr= np.char.replace(hr_str,':','').astype(float) # float hrminsec
    # 
    hr_thresh= 150000.
    nights= np.sort( list(set(day)))
    # store df's new night_obs colm
    night_arr= np.zeros(len(df)).astype(np.int32) - 1
    #
    for night in nights:
      # night starts at nighttime
      first_half= (day == night) & (hr > hr_thresh)
      # night continues to morning of next day
      str_next_day= get_next_day(str(night))
      sec_half= (day == int(str_next_day) ) & (hr < hr_thresh)
      keep= first_half | sec_half
      if np.where(keep)[0].size > 0:
          night_arr[keep]= night
    return df.assign(night_obs= night_arr)

  def thresh_exp_per_night(self,df, nexp=20):
    """keep data from night IFF has more than thresh number of good exposures"""
    keep= np.ones(len(df))
    for night in set(df.loc[:,'night_obs'].values):
      ind= np.where(df.loc[:,'night_obs'] == night)[0]
      if len(ind) < nexp:
        keep[ind]= False
    return df[keep]
     
  def drop_bad_transp(self,df, thresh=0.9):
    """drop low transp exposures for now"""
    df= df[(df.loc[:,'transparency'] > thresh)]
    return df

   

if __name__ == '__main__':
  d= Data(REPO_DIR)
  d.fetch_data()
  df = d.load_data()

  df= Clean().drop_nonreal_exposures()
  # FIX!! 20k occurences of night_obs == -1 (overwhelming majority)
  df= Clean().add_night_obs(df)
  # Don't do the following yet
  #df= Clean().thresh_exp_per_night(df, nexp=20)
  #df= Clean().drop_bad_transp(df, thresh=0.9)
  raise ValueError


