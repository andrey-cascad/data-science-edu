import numpy as np
import pandas as pd
import plotly.express as px

import sklearn

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import openpyxl
import scipy
from numpy import exp, array, random, dot
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


# mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
#
# clf = RandomForestClassifier(random_state=0)
# clf.fit()
# clf.predict()

def variablesDict():
    return {
        'Unnamed: 0': 'index',
        'Соотношение матрица-наполнитель': 'matrix_filling_ratio',
        'Плотность, кг/м3': 'density',
        'модуль упругости, ГПа': 'elasticity_modal',
        'Количество отвердителя, м.%': 'hardener_quantity',
        'Содержание эпоксидных групп,%_2': 'epoxy_groups_percent',
        'Температура вспышки, С_2': 'ignition_temperature',
        'Поверхностная плотность, г/м2': 'surface_density',
        'Модуль упругости при растяжении, ГПа': 'extension_elasticity_modal',
        'Прочность при растяжении, МПа': 'extension_strength',
        'Потребление смолы, г/м2': 'resin_consumption',
        'Угол нашивки, град': 'sewing_angle',
        'Шаг нашивки': 'sewing_step',
        'Плотность нашивки': 'sewing_density'
    }


def merge_sources():
    bp_df = pd.read_excel('hw_data/X_bp.xlsx')
    nup_df = pd.read_excel('hw_data/X_nup.xlsx')

    print(bp_df.describe())
    print(nup_df.describe())

    return pd.merge(left=bp_df, right=nup_df, how='inner', on='Unnamed: 0').rename(columns=variablesDict())


def getTrainDataFrame(df: DataFrame, percent: float = 0.7):
    return df.head(int((len(df) * percent)) + 1)


def getTestDataFrame(df: DataFrame, percent: float = 0.3):
    return df.tail(int((len(df) * percent)))


def hystogrames(df: DataFrame):
    a = 5
    b = 5
    c = 1
    plt.figure(figsize=(30, 30))
    plt.suptitle('Гистограммы переменных', fontsize=30)

    for col in df.columns:
        plt.subplot(a, b, c)
        sns.histplot(data=df[col], kde=True, color="y")
        plt.ylabel(None)
        plt.title(col, size=20)
        plt.show()
        c += 1

def boxplots(df: DataFrame):
    a = 5
    b = 5
    c = 1

    plt.figure(figsize=(30, 30))
    plt.suptitle('Box plots', y=0.9, fontsize=30)
    for col in df.columns:
        plt.subplot(a, b, c)
        sns.boxplot(data=df, y=df[col], fliersize=15, linewidth=5, boxprops=dict(facecolor='y', color='g'),
                    medianprops=dict(color='blue'), whiskerprops=dict(color="g"), capprops=dict(color="red"),
                    flierprops=dict(color="g", markeredgecolor="blue"))
        plt.ylabel(None)
        plt.title(col, size=20)
        c += 1

def emissions(df: DataFrame):
    method_3s = 0
    method_iq = 0

    count_iq = []
    count_3s = []

    for column in df:
        d = df.loc[:, [column]]

        zscore = (df[column] - df[column].mean()) / df[column].std()
        d['3s'] = zscore.abs() > 3
        method_3s += d['3s'].sum()
        count_3s.append(d['3s'].sum())
        print(column, '3s', ': ', d['3s'].sum())

        q1 = np.quantile(df[column], 0.25)
        q3 = np.quantile(df[column], 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        d['iq'] = (df[column] <= lower) | (df[column] >= upper)
        method_iq += d['iq'].sum()
        count_iq.append(d['iq'].sum())
        print(column, ': ', d['iq'].sum())

        print('Метод 3-х сигм, выбросов:', method_3s)
        print('Метод межквартильных расстояний, выбросов:', method_iq)

df = merge_sources()

if df.duplicated().sum() == 0:
    print("No duplicates.")

df.drop(['index'], axis=1, inplace=True)

print(df.describe())

df.hist(figsize=(25, 25), color="y")
plt.show()

hystogrames(df)
boxplots(df)

variables = variablesDict()

for display_name, name in variablesDict().items():
    if name != 'index':
        print(display_name)
        print("Mean value: ", df[name].mean())
        print("Std value: ", df[name].std())
        print("25 percentile value: ", np.percentile(df[name], 25))
        print("50 percentile value: ", np.percentile(df[name], 50))
        print("75 percentile value: ", np.percentile(df[name], 75))
        print("Median value: ", df[name].median())
        print("-----------------------")

normalized_df = preprocessing.normalize(df)

df_length = len(df)

training_df = getTrainDataFrame(df)
test_df = getTestDataFrame(df)

print("Training dataset: ", len(training_df))
print("Test dataset: ", len(test_df))

emissions(df)

normalizer = Normalizer()
res = normalizer.fit_transform(df)
df_norm_n = pd.DataFrame(res, columns = df.columns)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
    df_norm_n.loc[:, df_norm_n.columns != 'extension_elasticity_modal'],
    df[['extension_elasticity_modal']], test_size = 0.3, random_state = 42)

svr2 = make_pipeline(StandardScaler(), SVR(kernel = 'rbf', C = 500.0, epsilon = 1.0))

svr2.fit(x_train_2, np.ravel(y_train_2))

y_pred_svr2 = svr2.predict(x_test_2)
mae_svr2 = mean_absolute_error(y_pred_svr2, y_test_2)
mse_svr_elast2 = mean_squared_error(y_test_2,y_pred_svr2)
print('Support Vector Regression Results Train:')
print("Test score: {:.2f}".format(svr2.score(x_train_2, y_train_2))) # Скор для тренировочной выборки
print('Support Vector Regression Results:')
print('SVR_MAE:', round(mean_absolute_error(y_test_2, y_pred_svr2)))
print('SVR_MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test_2, y_pred_svr2)))
print('SVR_MSE: {:.2f}'.format(mse_svr_elast2))
print("SVR_RMSE: {:.2f}".format (np.sqrt(mse_svr_elast2)))
print("Test score: {:.2f}".format(svr2.score(x_test_2, y_test_2))) # Скор для тестовой выборки

def random_forest():
    rfr2 = RandomForestRegressor(n_estimators=15, max_depth=7, random_state=33)
    rfr2.fit(x_train_2, y_train_2.values)
    y2_pred_forest = rfr2.predict(x_test_2)
    mae_rfr2 = mean_absolute_error(y2_pred_forest, y_test_2)
    mse_rfr_elast2 = mean_squared_error(y_test_2, y2_pred_forest)
    print('Random Forest Regressor Results Train:')
    print("Test score: {:.2f}".format(rfr2.score(x_train_2, y_train_2)))
    print('Random Forest Regressor Results:')
    print('RF_MAE: ', round(mean_absolute_error(y_test_2, y2_pred_forest)))
    print('RF_MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test_2, y2_pred_forest)))
    print('RF_MSE: {:.2f}'.format(mse_rfr_elast2))
    print("RF_RMSE: {:.2f}".format(np.sqrt(mse_rfr_elast2)))
    print("Test score: {:.2f}".format(rfr2.score(x_test_2, y_test_2)))


def multi_layer_perception():
    mlp2 = MLPRegressor(random_state = 1, max_iter = 500)
    mlp2.fit(x_train_2, y_train_2)
    y_pred_mlp2 = mlp2.predict(x_test_2)
    mae_mlp2 = mean_absolute_error(y_pred_mlp2, y_test_2)
    mse_mlp_elast2 = mean_squared_error(y_test_2,y_pred_mlp2)
    print('Multi-layer Perceptron regressor Results Train:')
    print("Test score: {:.2f}".format(mlp2.score(x_train_2, y_train_2)))
    print('Multi-layer Perceptron regressor Results:')
    print('SGD_MAE: ', round(mean_absolute_error(y_test_2, y_pred_mlp2)))
    print('SGD_MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test_2, y_pred_mlp2)))
    print('SGD_MSE: {:.2f}'.format(mse_mlp_elast2))
    print("SGD_RMSE: {:.2f}".format (np.sqrt(mse_mlp_elast2)))
    print("Test score: {:.2f}".format(mlp2.score(x_test_2, y_test_2)))


def lasso():
    clf2 = linear_model.Lasso(alpha=0.1)
    clf2.fit(x_train_2, y_train_2)
    y_pred_clf2 = clf2.predict(x_test_2)
    mae_clf2 = mean_absolute_error(y_pred_clf2, y_test_2)
    mse_clf_elast2 = mean_squared_error(y_test_2, y_pred_clf2)
    print('Lasso regressor Results Train:')
    print("Test score: {:.2f}".format(clf2.score(x_train_2, y_train_2)))
    print('Lasso regressor Results:')
    print('SGD_MAE: ', round(mean_absolute_error(y_test_2, y_pred_clf2)))
    print('SGD_MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test_2, y_pred_clf2)))
    print('SGD_MSE: {:.2f}'.format(mse_clf_elast2))
    print("SGD_RMSE: {:.2f}".format(np.sqrt(mse_clf_elast2)))
    print("Test score: {:.2f}".format(clf2.score(x_test_2, y_test_2)))


def stochastic_gradient_descent():
    sdg2 = SGDRegressor()
    sdg2.fit(x_train_2, y_train_2)
    y_pred_sdg2 = sdg2.predict(x_test_2)
    mae_sdg2 = mean_absolute_error(y_pred_sdg2, y_test_2)
    mse_sdg_elast2 = mean_squared_error(y_test_2, y_pred_sdg2)
    print('Stochastic Gradient Descent Regressor Results Train:')
    print("Test score: {:.2f}".format(sdg2.score(x_train_2, y_train_2)))
    print('Stochastic Gradient Descent Regressor Results:')
    print('SGD_MAE: ', round(mean_absolute_error(y_test_2, y_pred_sdg2)))
    print('SGD_MSE: {:.2f}'.format(mse_sdg_elast2))
    print("SGD_RMSE: {:.2f}".format(np.sqrt(mse_sdg_elast2)))
    print('SGD_MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test_2, y_pred_sdg2)))
    print("Test score: {:.2f}".format(sdg2.score(x_test_2, y_test_2)))

lasso()
random_forest()
multi_layer_perception()
stochastic_gradient_descent()
