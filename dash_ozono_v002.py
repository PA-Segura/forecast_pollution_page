from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from pandas import Timedelta
from dash import Dash, dcc, html, Output, Input, callback_context, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from forecast_utils import getDataFrameFromDateRange, calculate_color, probability_2pass_threshold, moving_average_probabilities
import time
from urllib.parse import parse_qs, urlparse

debug_version = False
stations_dict = {
    'TLA': 'TLA - Tlalnepantla',
    'LLA': 'LLA - Los Laureles',
    'XAL': 'XAL - Xalostoc',
    'MER': 'MER - Merced',
    'SAG': 'SAG - San Agustín',
    'IZT': 'IZT - Iztacalco',
    'UIZ': 'UIZ - UAM Iztapalapa',
    'MGH': 'MGH - Miguel Hidalgo',
    'HGM': 'HGM - Hospital General de México',
    'FAC': 'FAC - FES Acatlán',
    'VIF': 'VIF - Villa de las Flores',
    'CAM': 'CAM - Camarones',
    'ATI': 'ATI - Atizapan',
    'CCA': 'CCA - Instituto de Ciencias de la Atmósfera y Cambio Climático',
    'TLI': 'TLI - Tultitlán',
    'PED': 'PED - Pedregal',
    'CUT': 'CUT - Cuautitlán',
    'MON': 'MON - Montecillo',
    'CUA': 'CUA - Cuajimalpa',
    'BJU': 'BJU - Benito Juárez',
    'NEZ': 'NEZ - Nezahualcóyotl',
    'UAX': 'UAX - UAM Xochimilco',
    'AJM': 'AJM - Ajusco Medio',
    'LPR': 'LPR - La Presa',
    'GAM': 'GAM - Gustavo A. Madero',
    'TAH': 'TAH - Tlahuac',
    'MPA': 'MPA - Milpa Alta',
    'AJU': 'AJU - Ajusco',
    'SFE': 'SFE - Santa Fe',
    'INN': 'INN - Investigaciones Nucleares',
}

# stations_dict = {
#     'UIZ': 'UIZ - UAM Iztapalapa',
#     'AJU': 'AJU - Ajusco',
#     'ATI': 'ATI - Atizapan',
#     'CUA': 'CUA - Cuajimalpa',
#     'SAG': 'SAG - San Agustín',
#     'CUT': 'CUT - Cuautitlán',
#     'PED': 'PED - Pedregal',
#     'TAH': 'TAH - Tlahuac',
#     'GAM': 'GAM - Gustavo A. Madero',
#     'IZT': 'IZT - Iztacalco',
#     'CCA': 'CCA - Instituto de Ciencias de la Atmósfera y Cambio Climático',
#     'HGM': 'HGM - Hospital General de México',
#     'LPR': 'LPR - La Presa',
#     'MGH': 'MGH - Miguel Hidalgo',
#     'CAM': 'CAM - Camarones',
#     'FAC': 'FAC - FES Acatlán',
#     'TLA': 'TLA - Tlalnepantla',
#     'MER': 'MER - Merced',
#     'XAL': 'XAL - Xalostoc',
#     'LLA': 'LLA - Los Laureles',
#     'TLI': 'TLI - Tultitlán',
#     'UAX': 'UAX - UAM Xochimilco',
#     'BJU': 'BJU - Benito Juárez',
#     'MPA': 'MPA - Milpa Alta',
#     'MON': 'MON - Montecillo',
#     'NEZ': 'NEZ - Nezahualcóyotl',
#     'INN': 'INN - Investigaciones Nucleares',
#     'AJM': 'AJM - Ajusco Medio',
#     'VIF': 'VIF - Villa de las Flores'
# }
# #    'SFE': 'SFE - Santa Fe',
#     'INN': 'INN - Investigaciones Nucleares',

dropdown_options = [{'label': name, 'value': code} for code, name in stations_dict.items()]

results_errd_df = pd.read_csv('./assets/results_errd_df_phours_station.csv')
print(results_errd_df)

intervals = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 250)]

# Función para determinar el índice del intervalo para un valor de predicción


def find_interval_index(value, intervals):
    for index, interval in enumerate(intervals):
        if value >= interval[0] and value < interval[1]:
            return index
    return None  # Retornar None si no se encuentra en ningún intervalo


def db_query_pasthours(id_est, start_time_str, end_time_str):
    # Crear la consulta SQL utilizando las variables start_time_str y end_time_str
    last_12_hours_query = f"""SELECT fecha, val, id_est FROM cont_otres
    WHERE fecha BETWEEN '{start_time_str}' AND '{end_time_str}'
    AND id_est = '{id_est}'
    ORDER BY fecha;"""

    # Obtener los datos con la consulta SQL
    last_12_hours_data = getDataFrameFromDateRange(last_12_hours_query)
    print(last_12_hours_query)
    return last_12_hours_data


def db_query_predhours(id_est, end_time_str):
    columnas_hp = ', '.join([f'hour_p{str(i).zfill(2)}' for i in range(1, 25)])
    # Consulta para df_pred_bd
    consulta_pred = f"""SELECT fecha, id_est, {columnas_hp} FROM forecast_otres 
                    WHERE fecha BETWEEN '{end_time_str}' AND '{end_time_str}'
                      AND id_tipo_pronostico = 6 AND id_est = '{id_est}'
                    ORDER BY fecha ;"""

    # Obtener los datos con la consulta SQL
    df_pred_bd = getDataFrameFromDateRange(consulta_pred)
    return df_pred_bd


def get_mu_sigma(hour, station, value, df, intervals):
    # Utilizar la función find_interval_index para encontrar el índice del intervalo
    interval_idx = find_interval_index(value, intervals)
    # Si se encuentra un intervalo válido
    if interval_idx is not None:
        # Convertir el intervalo a un string para buscar en el DataFrame
        interval_string = f"{intervals[interval_idx][0]}-{intervals[interval_idx][1]}"
        print(f'interval_string:{interval_string}')
        row = df[(df['hour'] == hour) & (df['station'] == station)
                 & (df['interval'] == interval_string)]
        if not row.empty:
            return row.iloc[0]['mu'], row.iloc[0]['sigma']
    return None, None  # Retornar None si no se encuentra un intervalo válido o no hay datos


def calculate_prediction_intervals(df_pred, hour_column_prefix, station, df_results, intervals):
    prediction_intervals_plus = []
    prediction_intervals_minus = []

    for i in range(1, 25):
        # Obtener el valor de predicción para la hora específica
        value_pred = df_pred[f'{hour_column_prefix}{str(i).zfill(2)}']

        # Obtener mu y sigma para el valor predicho, la estación y la hora
        mu, sigma = get_mu_sigma(i, station, value_pred, df_results, intervals)

        # Verificar que mu y sigma no sean None
        if sigma is not None and mu is not None:
            # Calcular los intervalos de predicción superior e inferior
            # Se usa signo (-) porque errores es (predicted-observed) Supon observed 150 ppb, y pronostico 120, errore es -30, entonces por eso el p_interval tiene signo (-)
            prediction_interval_plus = mu - (sigma * 1.96)
            prediction_interval_minus = mu + (sigma * 1.96)
        else:
            # En caso de que no se encuentre un intervalo válido, podrías querer añadir un valor por defecto o None
            prediction_interval_plus = 0  #np.nan
            prediction_interval_minus = 0  #np.nan

        prediction_intervals_plus.append(prediction_interval_plus)
        prediction_intervals_minus.append(prediction_interval_minus)
    print(prediction_intervals_plus, prediction_intervals_minus)
    # time.sleep(5)
    return prediction_intervals_plus, prediction_intervals_minus


def calculate_probabilities(df_pred_bd):
    """
    Calcula varias probabilidades basadas en diferentes criterios para el DataFrame proporcionado.
    """
    probabilities = []

    # Preparar datos de pronóstico
    df_pred_bd['fecha'] = pd.to_datetime(df_pred_bd['fecha'])
    timestamps_pred = [df_pred_bd['fecha'][0] + pd.Timedelta(hours=i) for i in range(1, 25)]
    values_pred = df_pred_bd.loc[0, 'hour_p01':'hour_p24']

    # Caso 1: Probabilidad de superar "Más de 50 ppb en 8hrs en siguientes 24 horas"
    mu, sigma = -0.43, 6.11  # Para mean_err_24h
    thresholdC2 = 50
    mean8probs = moving_average_probabilities(values_pred, 8, mu, sigma, thresholdC2)
    probabilities.append(max(mean8probs))

    # Caso 2: Probabilidad de superar "Umbral de 90 ppb en las siguientes 24 horas"
    forecast_level = values_pred.max()
    mu, sigma = 5.08, 18.03  # Para max_err_24h
    thresholdC3 = 90
    probabilities.append(probability_2pass_threshold(forecast_level, mu, sigma, thresholdC3))

    # Caso 3: Probabilidad de superar "Umbral de 120 ppb en las siguientes 24 horas"
    forecast_level = values_pred.max()
    mu, sigma = 5.08, 18.03  # Para max_err_24h
    thresholdC4 = 120
    probabilities.append(probability_2pass_threshold(forecast_level, mu, sigma, thresholdC4))

    # Caso 4: Probabilidad de superar Umbral de 150 ppb en las siguientes 24 horas
    forecast_level = values_pred.max()
    mu, sigma = 5.08, 18.03  # Para max_err_24h
    thresholdC1 = 150
    probabilities.append(probability_2pass_threshold(forecast_level, mu, sigma, thresholdC1))


    return probabilities


def db_query_last_predhour(id_est, time_now):
    columnas_hp = ', '.join([f'hour_p{str(i).zfill(2)}' for i in range(1, 25)])

    # Asegurarse de que time_now es un objeto datetime
    if not isinstance(time_now, datetime):
        # Convierte time_now a datetime si es necesario
        time_now = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S')

    # Calcular la hora de inicio restando 24 horas
    start_time = time_now - timedelta(hours=24)
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    # Consulta para encontrar el último pronóstico disponible en las últimas 5 horas
    consulta_pred = f"""SELECT fecha, id_est, {columnas_hp} FROM forecast_otres 
                        WHERE fecha BETWEEN '{start_time_str}' AND '{time_now.strftime('%Y-%m-%d %H:%M:%S')}'
                          AND id_tipo_pronostico = 6 AND id_est = '{id_est}'
                        ORDER BY fecha DESC
                        LIMIT 1;"""

    # Obtener los datos con la consulta SQL
    df_pred_bd = getDataFrameFromDateRange(consulta_pred)
    print('HERE -------')
    print(df_pred_bd)
    # time.sleep(10)
    # Obtener la fecha y hora del último pronóstico
    last_pred_datetime = df_pred_bd.iloc[0]['fecha'] if not df_pred_bd.empty else time_now

    return df_pred_bd, last_pred_datetime


def generate_layout(id_est='MER', fecha=None):
    if id_est == None:
        id_est = 'MER'

    if fecha:
        try:
            # Intenta convertir la cadena de fecha a un objeto datetime
            now = datetime.strptime(fecha, '%Y-%m-%dT%H')
            print(now)
            _, now = db_query_last_predhour(id_est, now) #datetime.now())
            print(now)
            # time.sleep(10)
            # now = datetime.strptime(fecha, '%Y-%m-%dT%H') #:%M')
        except ValueError:
            # Si hay un error en el formato de la fecha, utiliza la fecha y hora actual
            now = datetime.now()
            now = now.replace(minute=0, second=0, microsecond=0)
    else:
        # Si no se proporciona una fecha, obtenemos la fecha del último pronóstico disponible
        _, now = db_query_last_predhour(id_est, datetime.now())
        print(now)
        # time.sleep(5)
        # _, now = db_query_predhours(id_est, datetime.now())

    now_str = now.strftime('%Y-%m-%d %H:%M:%S')


    # Definir una función para obtener datos de la base de datos:
    # # id_est = 'MER'
    num_hours_past = 24

    # Retroceder 12 horas para obtener la hora de inicio
    start_time = now - timedelta(hours=num_hours_past)

    # Convertir los objetos datetime a cadenas de texto en el formato que necesitas para SQL
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = now.strftime('%Y-%m-%d %H:%M:%S')

    # Convertir la cadena de texto a un objeto datetime
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    # Retroceder 12 horas para obtener la hora de inicio
    start_time = end_time - timedelta(hours=num_hours_past)

    # Avanzar 1 hora para obtener la hora de inicio para los datos futuros
    future_start_time = end_time + timedelta(hours=1)

    # Avanzar 25 horas para obtener la hora de fin para los datos futuros (esto incluirá la hora inicial)
    future_end_time = end_time + timedelta(hours=24)

    # Convertir los objetos datetime a cadenas de texto en el formato que necesitas para SQL
    future_start_time_str = future_start_time.strftime('%Y-%m-%d %H:%M:%S')
    future_end_time_str = future_end_time.strftime('%Y-%m-%d %H:%M:%S')

    # Convertir el objeto datetime de la hora de inicio a una cadena de texto en el formato que necesitas para SQL
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    # %% #################################################
    # Data of ozone levels for the previous 24 hours

    last_12_hours_data = db_query_pasthours(id_est, start_time_str, end_time_str)

    # %% #################################################
    # Time series figure for past 12 hours

    time_series_fig = go.Figure()

    # Adding the main time series plot for past 12 hours
    style_traces = dict(
        family="Helvetica",  # Arial, sans-serif",  # Cambia a la fuente que prefieras
        size=20,  # Ajusta el tamaño de la fuente según tus necesidades
        color="black"
    )


    if not last_12_hours_data.empty:
        last_obs_time = last_12_hours_data.iloc[-1]['fecha']
        last_obs_value = last_12_hours_data.iloc[-1]['val']

        time_series_fig.add_trace(
            go.Scatter(x=last_12_hours_data['fecha'],
                       y=last_12_hours_data['val'],
                       mode='lines+markers',
                       name=f'<b>Pasadas {num_hours_past} horas</b>',
                       textfont=style_traces))

    else:
        print("El DataFrame con 24 horas previas está vacío, o hay un error")
        last_obs_time = None
        last_obs_value = None

    # %% #################################################
    # Data for the next 24 hours
    # Consulta de BD con predicciones
    df_pred_bd = db_query_predhours(id_est, end_time_str)

    # Asegurar que el DataFrame no esté vacío antes de continuar
    if df_pred_bd.empty:
        layout = html.Div(
    [

        # El resto del layout
        html.Div([
            html.Div([
                html.H3(
                    f'''Estación {stations_dict[id_est]} a las {now_str[:-3]} hrs. No hay datos en dataframe''',
                    style={
                        'text-align': 'center',
                        'font-family': 'Helvetica',  # 'Verdana',  # 'Palatino Linotype', #Cambia 'Arial' por la fuente que prefieras
                        'font-size': '24px'
                    }
                ),

                # ... Resto de los componentes del layout, dependiendo de si hay datos o no
            ]),
            # ... Más componentes del layout aquí
        ]),
        # ... Más componentes del layout aquí
    ]
)

        return layout

    if not df_pred_bd.empty:
        # Convertir la columna 'fecha' a tipo datetime para df_pred_bd
        df_pred_bd['fecha'] = pd.to_datetime(df_pred_bd['fecha'])

        # Crear una serie de timestamps para cada hora para df_pred_bd
        timestamps_pred = [
            df_pred_bd['fecha'][0] + pd.Timedelta(hours=i) for i in range(1, 25)
        ]

        # Obtener los valores para cada hora para df_pred_bd
        values_pred = df_pred_bd.loc[0, 'hour_p01':'hour_p24']

        # prediction_intervals_phour como 1.96 sigma de gaussian fit
        # Ejemplo de uso
        # prediction_intervals = calculate_prediction_intervals(
        #     values_pred, 'hour_p', id_est, results_errd_df, intervals)

        prediction_intervals_plus, prediction_intervals_minus = calculate_prediction_intervals(
            values_pred, 'hour_p', id_est, results_errd_df, intervals)

        # print(prediction_intervals, len(prediction_intervals))
        # prediction_intervals_phour = prediction_intervals
    else:
        print("El DataFrame para las próximas 24 horas está vacío.")

        timestamps_pred = [last_obs_time]

        # Obtener los valores para cada hora para df_pred_bd
        # Aquí puedes continuar con el resto del código para agregar estos datos al gráfico
        values_pred = [last_obs_value]

    # Adding standard deviation bands (Lower Bound) for next 24 hours
    # quitando los valores negativos que no son posibles en la vida real
    y_int_minus = (values_pred + prediction_intervals_minus)
    y_int_minus = [x if x >= 0 else 0 for x in y_int_minus]

    time_series_fig.add_trace(
        go.Scatter(x=timestamps_pred,
                   # y=values_pred - prediction_intervals_phour,
                   y= y_int_minus,
                   fill=None,
                   mode='lines',
                   # 'rgba(16, 182, 232,0.4)',# (68, 68, 68, 0.3)',
                   line_color='rgba(160, 223, 242,0.45)',
                   showlegend=False))



    # Add dashed vertical line to indicate the last observation time
    if last_obs_time:
        time_series_fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=last_obs_time,
                x1=last_obs_time,
                y0=10,
                y1=150,  # max(max(last_12_hours_data['val']), max(next_24_levels)),
                line=dict(color="blue", width=2, dash="dash"),
                line_color='rgba(0, 0, 250,0.6)'
            ))

        # Add dashed horizontal line to indicate the 150 ppb threshold
        time_series_fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=last_12_hours_data.iloc[0]['fecha'],  # Start time
                x1=timestamps_pred[-1],  # End time
                y0=150,  # Threshold level
                y1=150,  # Threshold level
                line=dict(color="red", width=2, dash="dash"),
                line_color='rgba(250, 0, 0,0.6)'
            ))

        # Add tick label
        time_series_fig.add_annotation(
            x=last_12_hours_data.iloc[0]['fecha'],
            y=(150 - 40), #+ num_hours_past,
            xref="x",
            yref="y",
            text="Umbral_150",
            showarrow=False,
            font=dict(family="Courier New, monospace", size=16, color="red"),
            textangle=90  # Rotar el texto 90 grados
        )

        # Add tick label
        time_series_fig.add_annotation(
            x=last_obs_time,
            y=0,
            xref="x",
            yref="y",
            text="Última Obs.",
            showarrow=False,
            font=dict(family="Courier New, monospace", size=16, color="blue"),
        )

    time_series_fig.update_layout(
        margin=dict(t=5, b=5, l=5, r=5),  # Reducir márgenes
        title='',
        # f'Niveles de Contaminantes: Pasadas {num_hours_past} horas y Pronóstico de 24 horas (Estación {id_est})',
        xaxis_title='Hora',
        yaxis_title='Nivel de Ozono (ppb)',  # (ppm x10^-3)

        # Estilo para el título del eje X
        xaxis=dict(
            title_font=dict(
                family="Helvetica, sans-serif",  # Tipo de fuente
                size=18,                         # Tamaño de la fuente
                color="#7f7f7f"                  # Color de la fuente
            ),
            tickfont=dict(
                family="Helvetica, sans-serif",  # Tipo de fuente para los ticks
                size=16,                         # Tamaño de la fuente para los ticks
                color="#7f7f7f"                  # Color de la fuente para los ticks
            )
        ),

        # Estilo para el título del eje Y
        yaxis=dict(
            title_font=dict(
                family="Helvetica, sans-serif",  # Tipo de fuente
                size=18,                         # Tamaño de la fuente
                color="#7f7f7f"                  # Color de la fuente
            ),
            tickfont=dict(
                family="Helvetica, sans-serif",  # Tipo de fuente para los ticks
                size=16,                         # Tamaño de la fuente para los ticks
                color="#7f7f7f"                  # Color de la fuente para los ticks
            )
        )
    )

    # # Updating the layout to include the station label
    y_int_plus = (values_pred + prediction_intervals_plus)
    y_int_plus = [x if x >= 0 else 0 for x in y_int_plus]


    # Adding prediction interval bands (Upper Bound) for next 24 hours
    time_series_fig.add_trace(
        go.Scatter(x=timestamps_pred,
                   # y=values_pred + prediction_intervals_phour,
                   y=y_int_plus, #values_pred + prediction_intervals_plus,
                   fill='tonexty',
                   # (16, 182, 232,0.4)', #rgba(68, 68, 68, 0.3)',
                   fillcolor='rgba(160, 223, 242,0.45)',
                   mode='lines',
                   # 'rgba(16, 182, 232,0.4)',# (68, 68, 68, 0.3)',
                   line_color='rgba(160, 223, 242,0.45)',
                   showlegend=False))

    # Adding the main time series plot for next 24 hours
    time_series_fig.add_trace(
        go.Scatter(x=timestamps_pred,
                   y=values_pred,
                   mode='lines+markers',
                   name='<b>Pronóstico</b>',  # ))
                   textfont=style_traces))

    time_series_fig.update_yaxes(range=[0, None])

    time_series_fig.update_layout(legend=dict(
        x=0.5,  # Posición en el eje x
        y=1.3,  # Posición en el eje y
        xanchor='center',  # Anclaje en el eje x
        yanchor='top',  # Anclaje en el eje y
        bgcolor='rgba(255,255,255,0)'  # Fondo transparente
    ))

    # # Ajustar el gráfico
    time_series_fig.update_layout(
        legend=dict(
            font=dict(
                family="Helvetica",
                size=16,
                color="black"
            )
        )
    )
    # Cambiar el fondo del gráfico de serie de tiempo a
    time_series_fig.update_layout(
        plot_bgcolor='rgba(173, 216, 230, 0.3)',
        # paper_bgcolor='rgba(240, 240, 240, 1)'  # Color gris claro para el fondo de la figura
    )



    if debug_version:
        time_series_fig.add_annotation(
            x=0.5,  # Posición en el eje x, en coordenadas relativas al gráfico (0-1)
            y=0.5,  # Posición en el eje y, en coordenadas relativas al gráfico (0-1)
            xref="paper",  # Coordenadas relativas al tamaño del gráfico
            yref="paper",  # Coordenadas relativas al tamaño del gráfico
            text="Datos Sintéticos/Depuración",  # El texto que quieres mostrar
            showarrow=False,  # No mostrar flecha apuntando al texto
            font=dict(family="Courier New, monospace", size=40, color="red"),
            bgcolor="rgba(0, 255, 0, 0.5)"  # Fondo verde fosforescente con alfa de 0.5
        )

    ############################################################
    # Generate Dial figures
    # Probabilidades para los diales, dado en orden consistente con los labels, ver función.

    probabilities = calculate_probabilities(df_pred_bd)

    dial_figs = []
    labels = [
        # "Media > 50 ppb/8hrs",#<br>en siguientes 24 hrs",
        "Media de más de 50 ppb en 8hrs",
        "Umbral de 90 ppb",# <br>siguientes 24 hrs",
        "Umbral de 120 ppb",# <br>siguientes 24 hrs",
        "Umbral de 150 ppb"# <br>siguientes 24 hrs"

    ]

    font_size = 15  # Tamaño de la fuente para el letrero en los diales

    for i, prob in enumerate(probabilities):
        color_for_bar = calculate_color(prob)
        dial_fig = go.Figure(
            go.Indicator(mode="gauge+number",
                         value=prob * 100,
                         gauge={
                             'axis': {
                                 'range': [None, 100]
                             },
                             'shape': 'angular',
                             'bar': {
                                 'color': color_for_bar
                             }
                         }))
        dial_fig.update_traces(number={'suffix': '%'})

        style_tit_dials = dict(
            family='Helvetica',  # Cambia la fuente a Arial o la que desees
            size=18,  # Cambia el tamaño de fuente a tu preferencia
            color='black'  # Cambia el color del título si es necesario
            # weight='bold'  # Pone el título en negrita
        )
        dial_fig.update_layout(
            # margin=dict(t=0, b=0, l=35, r=35),  # Ajusta márgenes para reducir espacio vertical
            margin=dict(t=60, b=25, l=25, r=25),  # Reducir márgenes
            height=230,  # Reducir altura del contenedor del dial
            title_text=f'<b>{labels[i]}</b>',
            title_x=0.5,  # Centrar horizontalmente
            # title_y=0.85,  # Centrar verticalmente (ajusta según tus preferencias)
            title_y=.95,  # Centrar verticalmente (ajusta según tus preferencias)
            title_xanchor='center',
            title_yanchor='top',
            title_font=style_tit_dials
        )

        # Agregar letrero sólo si debug_version es True
        if debug_version:
            dial_fig.add_annotation(
                x=0.5,  # Posición en el eje x, en coordenadas relativas al gráfico (0-1)
                y=0.5,  # Posición en el eje y, en coordenadas relativas al gráfico (0-1)
                xref="paper",  # Coordenadas relativas al tamaño del gráfico
                yref="paper",  # Coordenadas relativas al tamaño del gráfico
                text="Datos Sintéticos/Depuración",  # El texto que quieres mostrar
                showarrow=False,  # No mostrar flecha apuntando al texto
                font=dict(family="Courier New, monospace",
                          size=font_size,
                          color="red"),
                bgcolor="rgba(0, 255, 0, 0.5)"  # Fondo verde fosforescente con alfa de 0.5
            )
        dial_figs.append(dial_fig)

    # Generate dial figures in a 2x2 grid
    dial_grid = html.Div([
        html.Div([
            dcc.Graph(id=f'dial_{i}',
                      figure=dial_figs[i],
                      config={'displayModeBar': False})
        ],
            style={
            'width': '50%',
                     'height': '90%',
                     'display': 'inline-block'
        }) for i in range(4)
    ],
        style={'width': '100%'})

    layout = html.Div(
        [

            #     # El resto del layout
            html.Div([
                html.Div([
                    html.H3(f'''Estación {stations_dict[id_est]} a las {now_str[:-3]} hrs.''',
                            style={
                                'text-align': 'center',
                                'font-family': 'Helvetica',  # 'Verdana',  # 'Palatino Linotype', #Cambia 'Arial' por la fuente que prefieras
                                'font-size': '24px'
                            }),
                    dcc.Graph(id='time-series',
                              figure=time_series_fig,
                              config={'displayModeBar': False}),
                ],
                    style={
                    'width': '50%',
                             'display': 'inline-block',
                             'vertical-align': 'center'
                },
                    className='column'),

                html.Div([
                    html.H3('Probabilidad P_hist de superar umbrales en siguientes 24 hrs.',
                            style={
                                'text-align': 'center',
                                'font-family':  'Helvetica',  # 'Verdana',  # 'Palatino Linotype'
                                'font-size': '24px'
                            }),
                    dial_grid
                ],
                    style={
                    'width': '50%',
                             'display': 'inline-block',
                             'vertical-align': 'top'
                },
                    className='column'),
html.Footer([
    # html.P('Autores:', style={'font-weight': 'bold', 'font-family': 'Arial'}),
    html.Div([
        html.Span('Autores: ', style={'font-weight': 'bold', 'font-family': 'Arial','display': 'inline'}),
        html.Span('Olmo Zavala Romero - ', style={'display': 'inline'}),
        html.A('osz09@fsu.edu', href='mailto:osz09@fsu.edu', style={'display': 'inline'}),
        html.Span(', ', style={'display': 'inline'}),

        html.Span('Pedro A. Segura Chávez - ', style={'display': 'inline'}),
        html.A('psegura@atmosfera.unam.mx', href='mailto:psegura@atmosfera.unam.mx', style={'display': 'inline'}),
        html.Span(', ', style={'display': 'inline'}),

        html.Span('Pablo Camacho-Gonzalez - ', style={'display': 'inline'}),
        html.A('pablopcg1@ciencias.unam.mx', href='mailto:pablopcg1@ciencias.unam.mx', style={'display': 'inline'}),
        html.Span(', ', style={'display': 'inline'}),

        html.Span('Jorge Zavala-Hidalgo - ', style={'display': 'inline'}),
        html.A('jzavala@atmosfera.unam.mx', href='mailto:jzavala@atmosfera.unam.mx', style={'display': 'inline'}),
        html.Span(', ', style={'display': 'inline'}),

        html.Span('Pavel Oropeza Alfaro - ', style={'display': 'inline'}),
        html.A('poropeza@atmosfera.unam.mx', href='mailto:poropeza@atmosfera.unam.mx', style={'display': 'inline'}),
        html.Span(', ', style={'display': 'inline'}),

        html.Span('Rosario Romero-Centeno - ', style={'display': 'inline'}),
        html.A('rosario@atmosfera.unam.mx', href='mailto:rosario@atmosfera.unam.mx', style={'display': 'inline'}),
        html.Span(', ', style={'display': 'inline'}),

        html.Span('Octavio Gomez-Ramos - ', style={'display': 'inline'}),
        html.A('octavio@atmosfera.unam.mx', href='mailto:octavio@atmosfera.unam.mx', style={'display': 'inline'}),
        
        # Si hay más autores, continúa con el mismo patrón aquí
        # ...
    ], style={'text-align': 'left', 'padding': '7px', 'font-family': 'Arial'}),
 ]),

                # CSS in-line for responsive design
                html.Div([
                    html.Script(type='text/javascript',
                                children=[
                                    '''
                                    const styleTag = document.createElement("style");
                                    styleTag.innerHTML = `
                                    @media (max-width: 800px) {
                                    .column {
                                    flex-direction: column !important;
                                    width: 100% !important;
                                    }
                                    }
                                    `;
                                    document.head.appendChild(styleTag);
                                    '''
                                ])
                ])
            ])
        ],
        style={'background-color': '#ffffff'})
    return layout


# Configurar la aplicación Dash
title_dash = 'Pronóstico de Calidad del Aire Mediante Redes Neuronales: Nivel de Ozono de la RAMA'

if debug_version:
    title_dash = 'Ver_Desarrollo/Depuración:' + title_dash

app = Dash(__name__, title=title_dash, suppress_callback_exceptions=True)
server = app.server

# Establecer el layout inicial de la aplicación para escuchar cambios en la URL
# que incluye el Dropdown
app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),

        # logos
        html.Div([
            html.Img(src='assets/logo-mobile-icaycc.png',
                         style={
                             'width': '20%',
                             'display': 'inline-block',
                             'margin-left': '10px',
                             'margin-right': '10px'
                         }),
        ],
            style={
            'text-align': 'left',
                'margin-top': '20px'
        }),

        # Añadir el cintillo en la parte superior
        html.Div(
            [
                html.H1(title_dash,
                        style={
                            'font-family': 'Helvetica',
                            'color': 'white'
                        })
            ],
            style={
                'background-color': '#00505C',
                'padding': '5px',
                'text-align': 'center'
            }),

        # Div tag: Para poner el dropdown menu
        html.Div([
            html.Div([],
                     style={
                'width': '65%',
                     'display': 'inline-block',
                     'vertical-align': 'top'
                     },
                     className='column'),
            html.Div(
                [
                    # Div tag: Para poner el dropdown menu
                    html.Div(children=[
                        html.Label('Cambiar estación RAMA:'),
                        dcc.Dropdown(
                            id='my-dropdown',  # Asegúrate de asignar un ID único
                            options=dropdown_options,
                            value='MER'  # Valor predeterminado
                        )
                    ],
                        style={
                        'text-align': 'left',
                            'margin-top': '20px',
                            'font-family': 'Helvetica'
                    }),
                ],
                style={
                    'width': '30%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'text-align': 'right'
                },
                className='column'),
        ]),
        html.Div(id='page-content')  # Contenedor para el contenido generado dinámicamente
    ])

##############################
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'search'), Input('my-dropdown', 'value')],
    [State('page-content', 'children'), State('url', 'search')]
)
def update_content(url_search, dropdown_value, current_layout, state_url_search):
    ctx = callback_context

    # Si no hay Input, no actualizar nada
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Parsea la URL para obtener los parámetros
    parsed_url = urlparse(url_search)
    query_params = parse_qs(parsed_url.query)

    # Usar valores predeterminados si no se proporciona ninguna fecha o id_est
    default_id_est = 'MER'
    default_fecha = datetime.now().replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H')

    # Intenta obtener 'id_est' y 'fecha' de los parámetros de la URL o usa los valores predeterminados
    id_est_from_url = query_params.get('id_est', [default_id_est])[0]
    fecha_from_url = query_params.get('fecha', [default_fecha])[0]

    # Determina qué input disparó el callback
    if trigger_id == 'url':
        # Si el cambio proviene de la URL
        id_est = id_est_from_url
        fecha = fecha_from_url
    elif trigger_id == 'my-dropdown' and dropdown_value:
        # Si el cambio proviene del dropdown, se usa el valor del dropdown y se conserva la fecha actual
        id_est = dropdown_value
        fecha = fecha_from_url  # Aquí se mantiene la fecha que ya estaba en la URL
    else:
        # Si no hay cambios o el valor del dropdown no es válido, se previene la actualización
        raise PreventUpdate

    # Llamar a la función generate_layout con los parámetros obtenidos
    return generate_layout(id_est, fecha)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=debug_version, host='0.0.0.0', port=8888) # port=8050)
