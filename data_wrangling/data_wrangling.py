import pandas as pd 
import os,sys

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

df = pd.read_csv(cwd+ "/sentences.csv")
