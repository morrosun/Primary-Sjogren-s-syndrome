from sklearn.decomposition import PCA
import pandas as pd
df = pd.read_csv("~", header = 0)
data = df.iloc[0:len(df),0:(len(df.columns)-1)]
transfer = PCA(n_components=0.9)
data1 = transfer.fit_transform(data)
transfer2 = PCA(n_components=2)
data2 = transfer2.fit_transform(data)
data2.to_csv('~.csv')
