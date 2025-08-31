import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Preprocessor:
    # принимает на вход датафрйм
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # определяет числ и кат признаки
        self.num_features = [col for col in df.columns if self.df[col].dtype != 'object' and col != 'Hour']
        self.cat_features = [col for col in df.columns if col not in self.num_features]

    # считает среднее по числовым признакам
    def means(self):
        return self.df[self.num_features].mean()

    # добавляет новую колонку
    def add_hours_intervals_feature(self):
        self.df['hours_intervals'] = self.df['Hour'].apply(lambda x: 0 if 0 <= x <= 8 else 1)

    def set_features(self):
        num, cat = [], []
        for col in self.df.columns:
            if self.df[col].dtype != 'object':
                num.append(col)
            else:
                cat.append(col)

    def fill_nans_values(self):
        for col in self.num_features:
            self.df[col] = self.df[col].fillna(self.df[col].median())

    def del_corr_features(self, threshold=0.8):
        """
        удаляет признаки (столбцы) из таблицы, которые слишком сильно похожи на другие признаки,
        то есть сильно коррелируют (например, корреляция больше 0.8)"""

        # создаем таблицу корреляций
        corr_matrix = self.df.corr().abs()

        # делаем треугольную матрицу те оставляем верхнюю часть таблицы
        ones_matrix = np.ones(corr_matrix.shape)  # 1
        upper_triangle = np.triu(ones_matrix)  # 2 оставляем только верхнюю часть
        matrix_bool = upper_triangle.astype(bool)  # 3 превращает в булевую
        corr_matrix_triu = corr_matrix.where(matrix_bool)  # 4

        # corr_features = []
        # for col in corr_matrix_triu.columns:
        #     if any((corr_matrix_triu[col] > threshold) & corr_matrix_triu[col] != 1):
        #         corr_features.append(col)
        # self.df.drop(columns=corr_features, inplace=True)

        # перепишем через 2 фора
        to_drop = set()
        for i in corr_matrix_triu.columns:
            for j in corr_matrix_triu.columns:
                # df.loc[строка, колонка]
                if i != j and corr_matrix_triu.loc[i, j] > threshold:
                    self.df.drop(corr_matrix_triu.loc[j], inplace=True)
                    to_drop.add(j)
        self.df.drop(columns=to_drop, inplace=True)

    def drop_law_var_features(self, threshold):
        vars = self.df[self.num_features].var()
        drop_features = vars[vars < threshold].index.tolist()
        self.df.drop(columns=drop_features, inplace=True)

    # тепловая карта это визуализация матрицы чисел где каждое значение представлено цветом

    def encode_categorical_features(self):
        for col in self.cat_features:
            if len(self.df[col].unique()) <= 5:
                self.df = pd.get_dummies(self.df, columns=[col])
            else:
                # это просто target encoding, где Temperature — это целевая переменная (target)
                encode_col = self.df.groupby(col)['Temperature'].mean()
                # применяем к колонкам
                self.df[col] = self.df[col].apply(lambda x: encode_col.loc[x])

    def heatmap(self):
        corr_matrix = self.df.select_dtypes(include='number').corr()

        plt.figure(figsize=(10, 8))  # размер картинки
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")  # тепловая карта

        os.makedirs("../images", exist_ok=True)

        plt.savefig("images/correlation_heatmap.png")
        plt.close()
    def __str__(self):
        return str(self.df.head(7))
