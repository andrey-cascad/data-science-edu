from flask import Flask, render_template, request
import pandas as pd
import pickle

MODELS_PATH = '../models/'


def parameters():
    return {
        'matrix_filling_ratio': 'Соотношение матрица-наполнитель',
        'density': 'Плотность, кг/м3',
        'elasticity_modal': 'модуль упругости, ГПа',
        'hardener_quantity': 'Количество отвердителя, м.%',
        'epoxy_groups_percent': 'Содержание эпоксидных групп,%_2',
        'ignition_temperature': 'Температура вспышки, С_2',
        'surface_density': 'Поверхностная плотность, г/м2',
        'extension_elasticity_modal': 'Модуль упругости при растяжении, ГПа',
        'extension_strength': 'Прочность при растяжении, МПа',
        'resin_consumption': 'Потребление смолы, г/м2',
        'sewing_angle': 'Угол нашивки, град',
        'sewing_step': 'Шаг нашивки',
        'sewing_density': 'Плотность нашивки'
    }


def get_data_from_form(features, params):
    param_names = features.keys()
    data = dict.fromkeys(param_names, None)
    error = ''

    for param_name, param_value in params.items():
        if param_value.strip(' \t') != '':
            try:
                data[param_name] = float(param_value)
            except:
                error += f'{features[param_name]} - некорректное значение "{param_value}"\n'
    if 'matrix_filling_ratio' in data and data['matrix_filling_ratio'] is not None:
        if data['matrix_filling_ratio'] < 0 or data['matrix_filling_ratio'] > 6:
            error += f'{features["matrix_filling_ratio"]} - значение вне корректного диапазона\n'
    if 'density' in data and data['density'] is not None:
        if data['density'] < 1700 or data['density'] > 2300:
            error += f'{features["density"]} - значение вне корректного диапазона\n'
    if 'elasticity_modal' in data and data['elasticity_modal'] is not None:
        if data['elasticity_modal'] < 2 or data['elasticity_modal'] > 2000:
            error += f'{features["elasticity_modal"]} - значение вне корректного диапазона\n'
    if 'hardener_quantity' in data and data['hardener_quantity'] is not None:
        if data['hardener_quantity'] < 17 or data['hardener_quantity'] > 200:
            error += f'{features["hardener_quantity"]} - значение вне корректного диапазона\n'
    if 'epoxy_groups_percent' in data and data['epoxy_groups_percent'] is not None:
        if data['epoxy_groups_percent'] < 14 or data['epoxy_groups_percent'] > 34:
            error += f'{features["epoxy_groups_percent"]} - значение вне корректного диапазона\n'
    if 'ignition_temperature' in data and data['ignition_temperature'] is not None:
        if data['ignition_temperature'] < 100 or data['ignition_temperature'] > 414:
            error += f'{features["ignition_temperature"]} - значение вне корректного диапазона\n'
    if 'surface_density' in data and data['surface_density'] is not None:
        if data['surface_density'] < 0.6 or data['surface_density'] > 1400:
            error += f'{features["surface_density"]} - значение вне корректного диапазона\n'
    if 'extension_elasticity_modal' in data and data['extension_elasticity_modal'] is not None:
        if data['extension_elasticity_modal'] < 64 or data['extension_elasticity_modal'] > 83:
            error += f'{features["extension_elasticity_modal"]} - значение вне корректного диапазона\n'
    if 'extension_strength' in data and data['extension_strength'] is not None:
        if data['extension_strength'] < 1036 or data['extension_strength'] > 3849:
            error += f'{features["extension_strength"]} - значение вне корректного диапазона\n'
    if 'resin_consumption' in data and data['resin_consumption'] is not None:
        if data['resin_consumption'] < 33 or data['resin_consumption'] > 414:
            error += f'{features["resin_consumption"]} - значение вне корректного диапазона\n'
    if 'sewing_angle' in data and data['sewing_angle'] is not None:
        if data['sewing_angle'] != 0.0 and data['sewing_angle'] != 90.0:
            error += f'{features["sewing_angle"]} - значение вне корректного диапазона\n'
    if 'sewing_step' in data and data['sewing_step'] is not None:
        if data['sewing_step'] < 0 or data['sewing_step'] > 15:
            error += f'{features["sewing_step"]} - значение вне корректного диапазона\n'
    if 'sewing_density' in data and data['sewing_density'] is not None:
        if data['sewing_density'] < 0 or data['sewing_density'] > 104:
            error += f'{features["sewing_density"]} - значение вне корректного диапазона\n'
    if None in data.values():
        error += f'Некоторые значения отсутствуют!\n'
    data_clean = dict(zip(features.values(), data.values()))
    return data_clean, error


def load_pickle_obj(filename):
    file = open(MODELS_PATH + filename, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


app = Flask(__name__)


@app.route('/features/', methods=['post', 'get'])
def features_page():
    params = {'matrix_filling_ratio': '3', 'density': '2000', 'elasticity_modal': '1999', 'hardener_quantity': '95',
              'epoxy_groups_percent': '25', 'ignition_temperature': '255', 'surface_density': '720',
              'extension_elasticity_modal': '70', 'extension_strength': '2300', 'resin_consumption': '180',
              'sewing_angle': '0', 'sewing_step': '8', 'sewing_density': '52'}
    error = ''
    result = ''

    if request.method == 'POST':
        params = request.form.to_dict()
        data, error = get_data_from_form(parameters(), params)
        if error == '':
            x = pd.DataFrame(data, index=[0])
            result = str(x.to_html())

    return render_template('features.html', params=params, error=error, result=result)


@app.route('/model_1_2/', methods=['post', 'get'])
def model_1_2_page():
    params = dict(zip(parameters().keys(), ['4.02912621359223', '1880.0', '622.0', '111.86',
                                            '22.2678571428571', '284.615384615384', '470.0', '220.0', '90.0',
                                            '4.0', '60.0']))
    #
    error = ''
    x = pd.DataFrame()
    extension_elasticity_modal = ''
    extension_strength = ''

    if request.method == 'POST':
        params = request.form.to_dict()
        data, error = get_data_from_form(parameters(), params)
        if error == '':
            x = pd.DataFrame(data, index=[0])

            preprocessor1 = load_pickle_obj('preprocessor1')
            model1 = load_pickle_obj('model1_best')
            x1 = preprocessor1.transform(x)
            y1 = model1.predict(x1)
            extension_elasticity_modal = y1[0]

            preprocessor2 = load_pickle_obj('preprocessor2')
            model2 = load_pickle_obj('model2_best')
            x2 = preprocessor2.transform(x)
            y2 = model2.predict(x2)
            extension_strength = y2[0]

    return render_template('model_1_2.html', params=params, error=error, inputs=x.to_html(),
                           extension_elasticity_modal=extension_elasticity_modal, extension_strength=extension_strength)


@app.route('/model_3/', methods=['post', 'get'])
def model_3_page():
    params = dict(zip(parameters().keys(), ['1880.0', '622.0', '111.86', '22.2678571428571',
                                            '284.615384615384', '470.0', '73.3333333333333',
                                            '2455.55555555555', '220.0', '90.0', '4.0', '60.0']))
    #
    error = ''
    x = pd.DataFrame()
    matrix_filling_ratio = ''

    if request.method == 'POST':
        params = request.form.to_dict()
        data, error = get_data_from_form(parameters(), params)
        if error == '':
            x = pd.DataFrame(data, index=[0])

            preprocessor3 = load_pickle_obj('preprocessor3')
            model3 = load_pickle_obj('model3_1')
            x3 = preprocessor3.transform(x)
            y3 = model3.predict(x3)
            matrix_filling_ratio = y3[0]

    return render_template('model_3.html', params=params, error=error, inputs=x.to_html(),
                           matrix_filling_ratio=matrix_filling_ratio)


@app.route('/')
def main_page():
    return render_template('main.html')


@app.route('/url_map/')
def url_map():
    return str(app.url_map)


app.run()
