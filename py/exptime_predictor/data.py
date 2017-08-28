import numpy as np
import os
import pandas as pd
import sqlite3
import time
from datetime import date, timedelta
from astropy.time import Time

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

def change_day(ymd,new_day):
  """given year month day string (ymd) returns the previous or next calendar day
  
  Args:
      ymd: '20110531'
      new_day: ['yesterday','tomorrow']
      
  Returns:
      ymd for next day: '20110601' for example above
  """
  assert(new_day in ['yesterday','tomorrow'])
  t=time.strptime(ymd,'%Y%m%d')
  dt= timedelta(days=1)
  if new_day == 'yesterday':
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday) - dt
  elif new_day == 'tomorrow':
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday) + dt
  return newdate.strftime('%Y%m%d')


class Clean(object):
  def keep_science_exposures(self,df):
    """Only exposures named 'object' and band in grz are science exposures"""
    bands= df.loc[:,'band']
    isGRZ= (bands == 'g') | (bands == 'r') | (bands == 'z') 
    isObject= df.loc[:,'obstype'] == 'object'
    return df[ isGRZ & isObject ]
   
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

  def add_night_obs(self,df):
    """assigns integer ymd to each instance

    Night observed begins in evening and continutes to morning of next
      day. Found that float hrminsec = 150000.0 cleanly separates morning
      from evening

    Returns:
      df modified to have 'night_obs' and 'hr_obs' columns
    """
    t = Time(df.loc[:,'mjd_obs'], format='mjd')
    # t.iso is list of elements like '2016-02-25 05:11:31.769'
    nightobs_str= np.array([a.split(' ')[0] for a in t.iso])
    nightobs_str= np.char.replace(nightobs_str,'-','') # ymd
    hr_str= np.array([a.split(' ')[1] for a in t.iso])
    hr= np.char.replace(hr_str,':','').astype(float) # float hrminsec
    # All mornings need to be relabeled as previous calendar day
    hr_thresh= 150000.
    isMorning= hr < hr_thresh
    for ymd in list(set(nightobs_str)):
      # relabel the mornings 
      nightobs_str[ (nightobs_str == ymd) & (isMorning) ]= change_day(ymd,'yesterday')
    df= df.assign(night_obs= nightobs_str.astype(int))
    df= df.assign(hr_obs= hr)
    return df

  def drop_nights_wfew_exposures(self,df, nexp=20):
    """Return df with all nights with less than nexp dropped"""
    return df.groupby("night_obs").filter(lambda g: g.night_obs.size >= nexp)
    
  def drop_bad_transp(self,df, thresh=0.9):
    """drop low transp exposures"""
    return df[ df.loc[:,'transparency'] > thresh ]

   

if __name__ == '__main__':
  d= Data(REPO_DIR)
  d.fetch_data()
  df = d.load_data()

  df= Clean().keep_science_exposures(df)
  df= Clean().add_night_obs(df)
  df= Clean().drop_bad_transp(df, thresh=0.9)
  # TODO: remove duplicated expids b/c have > 1000 exposures on some nights
  # ALWAYS last step
  df= Clean().drop_nights_wfew_exposures(df, nexp=20)
  raise ValueError


