from pandas import read_csv
from sklearn.metrics import adjusted_rand_score
from package.copac import COPAC

df = read_csv('./datasets/MargemEquatorial/dataset.csv')

if 'Unnamed: 0' in df.columns:
    del df['Unnamed: 0']
if 'sorting' in df.columns:
    del df['sorting']
y = df['petrofacie'].values
del df['petrofacie']
df = df.reset_index(drop=True)

for i in range(10, 20, 1):
    eps = float(i)
    copac = COPAC(k=60, eps=eps, mu=1)
    copac.fit(df)

    ari = adjusted_rand_score(y, copac.labels_)
    print('eps=%s, ari=%s' % (eps, ari))
