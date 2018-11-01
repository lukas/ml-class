import pandas as pd
import numpy as np

sdf = pd.Series(np.sin(np.linspace(0,60,1000)))
sdf.to_csv("sin.csv", header=["sin"])

two_sdf = pd.Series(np.sin(np.linspace(0,20,5000))*np.sin(np.linspace(0,200,5000)))
two_sdf.to_csv("twosin.csv", header=["sin"])