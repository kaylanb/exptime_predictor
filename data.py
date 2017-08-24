# Common imports
import numpy as np
import os
import pandas as pd
import sqlite3

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
  def __init__(self):
    self.repo_dir= REPO_DIR
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

  def apply_cuts(self): 
    # Remove whitespaces
    self.data.set('band', self.data.get('band').astype(np.string_))
    self.data.set('band', np.char.strip(self.data.band))
    print('Initially %d' % len(self.data))
    # obstype: {u'dark', u'dome flat', u'object', u'zero'}
    # band: {u'', u'VR', u'Y', u'g', u'i', u'r', u'solid', u'u', u'z'}
    keep= (self.data.obstype == 'object')
    if self.camera == 'decam':
      keep *= (np.any([self.data.band == 'g',
                       self.data.band == 'r',
                       self.data.band == 'z'],axis=0) )
    elif self.camera == 'mosaic':
      keep *= (self.data.band == 'zd')
    self.data.cut(keep)
    print('Object and band cuts %d' % len(self.data))
    # rename_bands:
    if self.camera == 'mosaic':
      assert( np.all(self.data.band == 'zd') )
      self.data.set('band', np.array(['z']*len(self.data)) )

if __name__ == '__main__':
  d= Data()
  d.fetch_data()
  df = d.load_data()
  raise ValueError


