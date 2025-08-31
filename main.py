import pandas as pd

from prepr import Preprocessor

df = pd.read_csv('./MLS/BikeData.csv')
p = Preprocessor(df)
p.encode_categorical_features()

print(p.num_features)        # числовые признаки
print(p.cat_features)        # категориальные признаки
print(p.means())             # средние значения по числовым колонкам
p.__str__()
print(p)                     # 7 строк с новой колонкой
p.heatmap()
print(p)