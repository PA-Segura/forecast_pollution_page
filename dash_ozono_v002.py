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

debug_version = True

stations_dict = {
    'UIZ': 'UIZ - UAM Iztapalapa',
    'AJU': 'AJU - Ajusco',
    'ATI': 'ATI - Atizapan',
    'CUA': 'CUA - Cuajimalpa',
    'SFE': 'SFE - Santa Fe',
    'SAG': 'SAG - San Agustín',
    'CUT': 'CUT - Cuautitlán',
    'PED': 'PED - Pedregal',
    'TAH': 'TAH - Tlahuac',
    'GAM': 'GAM - Gustavo A. Madero',
    'IZT': 'IZT - Iztacalco',
    'CCA': 'CCA - Instituto de Ciencias de la Atmósfera y Cambio Climático',
    'HGM': 'HGM - Hospital General de México',
    'LPR': 'LPR - La Presa',
    'MGH': 'MGH - Miguel Hidalgo',
    'CAM': 'CAM - Camarones',
    'FAC': 'FAC - FES Acatlán',
    'TLA': 'TLA - Tlalnepantla',
    'MER': 'MER - Merced',
    'XAL': 'XAL - Xalostoc',
    'LLA': 'LLA - Los Laureles',
    'TLI': 'TLI - Tultitlán',
    'UAX': 'UAX - UAM Xochimilco',
    'BJU': 'BJU - Benito Juárez',
    'MPA': 'MPA - Milpa Alta',
    'MON': 'MON - Montecillo',
    'NEZ': 'NEZ - Nezahualcóyotl',
    'INN': 'INN - Investigaciones Nucleares',
    'AJM': 'AJM - Ajusco Medio',
    'VIF': 'VIF - Villa de las Flores'
}

dropdown_options = [{'label': name, 'value': code} for code, name in stations_dict.items()]

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
                    ORDER BY fecha;"""

    # Obtener los datos con la consulta SQL
    df_pred_bd = getDataFrameFromDateRange(consulta_pred)
    return df_pred_bd


def calculate_probabilities(df_pred_bd):
    """
    Calcula varias probabilidades basadas en diferentes criterios para el DataFrame proporcionado.
    """
    probabilities = []

    # Preparar datos de pronóstico
    df_pred_bd['fecha'] = pd.to_datetime(df_pred_bd['fecha'])
    timestamps_pred = [df_pred_bd['fecha'][0] + pd.Timedelta(hours=i) for i in range(1, 25)]
    values_pred = df_pred_bd.loc[0, 'hour_p01':'hour_p24']

    # Caso 1: Probabilidad de superar Umbral de 150 ppb en las siguientes 24 horas
    forecast_level = values_pred.max()
    mu, sigma = 5.08, 18.03  # Para max_err_24h
    thresholdC1 = 150
    probabilities.append(probability_2pass_threshold(forecast_level, mu, sigma, thresholdC1))

    # Caso 2: Probabilidad de superar "Media > 50 ppb/8hrs en siguientes 24 horas"
    mu, sigma = -0.43, 6.11  # Para mean_err_24h
    thresholdC2 = 50
    mean8probs = moving_average_probabilities(values_pred, 8, mu, sigma, thresholdC2)
    probabilities.append(max(mean8probs))

    # Caso 3: Probabilidad de superar "Umbral de 90 ppb en las siguientes 24 horas"
    thresholdC3 = 90
    probabilities.append(probability_2pass_threshold(forecast_level, mu, sigma, thresholdC3))

    # Caso 4: Probabilidad de superar "Umbral de 120 ppb en las siguientes 24 horas"
    thresholdC4 = 120
    probabilities.append(probability_2pass_threshold(forecast_level, mu, sigma, thresholdC4))

    return probabilities



# # Definir una función para generar el layout de la aplicación
def generate_layout(id_est='MER'):
    # Aquí va tu código para generar el layout dinámicamente basado en id_est
    # Por ejemplo, consultar la base de datos, generar figuras, etc.

    #Obtener la fecha y hora actual
    now = datetime.now()

    # Redondear a la hora en punto más cercana
    now = now.replace(minute=0, second=0, microsecond=0)
    now = now - timedelta(hours=205 * 24 )  # para que sea año 2019 en tests.
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    # Definir una función para obtener datos de la base de datos:
    # Consulta para obtener datos observados de la estación UAX para el periodo especificado en 2019

    # id_est = 'MER'
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

    # # Initialize the Dash app
    # title_dash = 'Pronóstico de Calidad del Aire Mediante Redes Neuronales: Nivel de Ozono de la RAMA'
    
    # if debug_version:
    #     title_dash = 'Ver_Desarrollo/Depuración:' + title_dash
        
    # app = Dash(__name__, title=title_dash)
    # server = app.server  # <-------- add this!!!
        
    # Time series figure for past 12 hours
    time_series_fig = go.Figure()

    # Adding the main time series plot for past 12 hours
    time_series_fig.add_trace(
        go.Scatter(x=last_12_hours_data['fecha'],
                   y=last_12_hours_data['val'],
                   mode='lines+markers',
                   name=f'Pasadas {num_hours_past} horas'))

    if not last_12_hours_data.empty:
        last_obs_time = last_12_hours_data.iloc[-1]['fecha']
        last_obs_value = last_12_hours_data.iloc[-1]['val']

    else:
        print("El DataFrame con 24 horas previas está vacío, o hay un error")

    # %% #################################################
    # Data for the next 24 hours
    # prediction_intervals_phour, tomando RMSE para prediction intervals
    prediction_intervals_phour = [
        10.416753, 10.987964, 11.639589, 12.107008, 12.472200, 12.658631,
        12.769756, 12.810751, 12.783473, 12.766654, 12.772853, 12.825606,
        12.849491, 12.884915, 12.912820, 12.828738, 12.778177, 12.848434,
        12.861159, 12.875029, 12.891028, 12.869136, 12.911643, 13.101732
    ]
    
    # Consulta de BD con predicciones
    df_pred_bd = db_query_predhours(id_est, end_time_str)

    # Asegurarte de que el DataFrame no esté vacío antes de continuar
    if not df_pred_bd.empty:
        # Convertir la columna 'fecha' a tipo datetime para df_pred_bd
        df_pred_bd['fecha'] = pd.to_datetime(df_pred_bd['fecha'])
        
        # Crear una serie de timestamps para cada hora para df_pred_bd
        timestamps_pred = [
            df_pred_bd['fecha'][0] + pd.Timedelta(hours=i) for i in range(1, 25)
        ]
        
        # Obtener los valores para cada hora para df_pred_bd
        values_pred = df_pred_bd.loc[0, 'hour_p01':'hour_p24']
        
    else:
        print("El DataFrame para las próximas 24 horas está vacío.")
        
        timestamps_pred = [last_obs_time]
        # Obtener los valores para cada hora para df_pred_bd
        # Aquí puedes continuar con el resto del código para agregar estos datos al gráfico
        values_pred = [last_obs_value]
        
    # Adding standard deviation bands (Lower Bound) for next 24 hours
    time_series_fig.add_trace(
        go.Scatter(x=timestamps_pred,
                   y=values_pred - prediction_intervals_phour,
                   fill=None,
                   mode='lines',
                   line_color='rgba(68, 68, 68, 0.3)',
                   showlegend=False))
    
    # Add dashed vertical line to indicate the last observation time
    time_series_fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=last_obs_time,
            x1=last_obs_time,
            y0=10,
            y1=150,  # max(max(last_12_hours_data['val']), max(next_24_levels)),
            line=dict(color="blue", width=2, dash="dash")))
    
    # Add dashed horizontal line to indicate the 150 ppb threshold
    time_series_fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=last_12_hours_data.iloc[0]['fecha'],  # Start time
            x1=timestamps_pred[-1],  # End time
            y0=150,  # Threshold level
            y1=150,  # Threshold level
            line=dict(color="red", width=2, dash="dash")))
    
    # Add tick label
    time_series_fig.add_annotation(
        x=last_12_hours_data.iloc[0]['fecha'],
        y=150 + num_hours_past,
        xref="x",
        yref="y",
        text="Umbral_150",
        showarrow=False,
        font=dict(family="Courier New, monospace", size=16, color="red"),
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
    
    # Updating the layout to include the station label
    time_series_fig.update_layout(
        title='',
        # f'Niveles de Contaminantes: Pasadas {num_hours_past} horas y Pronóstico de 24 horas (Estación {id_est})',
        xaxis_title='Hora',
        yaxis_title='Nivel de Ozono (ppb)'  # (ppm x10^-3)
    )
    
    # Adding standard deviation bands (Upper Bound) for next 24 hours
    time_series_fig.add_trace(
        go.Scatter(x=timestamps_pred,
                   y=values_pred + prediction_intervals_phour,
                   fill='tonexty',
                   fillcolor='rgba(68, 68, 68, 0.3)',
                   mode='lines',
                   line_color='rgba(68, 68, 68, 0.3)',
                   showlegend=False))
    
    # Adding the main time series plot for next 24 hours
    time_series_fig.add_trace(
        go.Scatter(x=timestamps_pred,
                   y=values_pred,
                   mode='lines+markers',
                   name='Pronóstico'))
    
    time_series_fig.update_layout(legend=dict(
        x=0.5,  # Posición en el eje x
        y=1.16,  # Posición en el eje y
        xanchor='center',  # Anclaje en el eje x
        yanchor='top',  # Anclaje en el eje y
        bgcolor='rgba(255,255,255,0)'  # Fondo transparente
    ))

    # # Ajustar el tamaño del gráfico
    # time_series_fig.update_layout(
    #     autosize=False,
    #     width=1000,  # Ancho del gráfico
    #     height=500,  # Altura del gráfico
    # )

    # # # Opcionalmente, puedes ajustar también el margen
    # time_series_fig.update_layout(margin=dict(l=50, r=50, b=100, t=100, pad=4))

    if debug_version:
        time_series_fig.add_annotation(
            x=0.5,  # Posición en el eje x, en coordenadas relativas al gráfico (0-1)
            y=0.5,  # Posición en el eje y, en coordenadas relativas al gráfico (0-1)
            xref="paper",  # Coordenadas relativas al tamaño del gráfico
            yref="paper",  # Coordenadas relativas al tamaño del gráfico
            text="Datos Sintéticos/Depuración",  # El texto que quieres mostrar
            showarrow=False,  # No mostrar flecha apuntando al texto
            font=dict(family="Courier New, monospace", size=40, color="red"),
            bgcolor=
            "rgba(0, 255, 0, 0.5)"  # Fondo verde fosforescente con alfa de 0.5
        )
        
        
        
    ############################################################
    # Generate Dial figures
    # Probabilidades para los diales, dado en orden consistente con los labels, ver función.
    
    probabilities = calculate_probabilities(df_pred_bd)
    
    dial_figs = []
    labels = [
        "Umbral de 150 ppb siguientes 24 hrs",
        "Media > 50 ppb/8hrs en siguientes 24 hrs",
        "Umbral de 90 ppb siguientes 24 hrs",
        "Umbral de 120 ppb siguientes 24 hrs"
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
        dial_fig.update_layout(title=labels[i])
        
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
                bgcolor=
                "rgba(0, 255, 0, 0.5)"  # Fondo verde fosforescente con alfa de 0.5
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
                     'height': '100%',
                     'display': 'inline-block'
                 }) for i in range(4)
    ],
                         style={'width': '100%'})
    
    layout = html.Div(
        [
            #     # El resto de tu layout
            html.Div([
                html.Div([
                    html.H3(f'''Estación {stations_dict[id_est]} a las {now_str[:-3]} hrs.''',
                            style={
                                'text-align': 'center',
                                'font-family': 'Helvetica'
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
                
                # html.Div([
                html.Div([
                    html.H3('Probabilidad de superar umbrales',
                            style={
                                'text-align': 'center',
                                'font-family': 'Helvetica'
                            }), dial_grid
                ],
                         style={
                             'width': '50%',
                             'display': 'inline-block',
                             'vertical-align': 'top'
                         },
                         className='column'),
                html.Footer([
                    html.P(
                        [
                            'Autores: Olmo Zavala Romero, osz09@fsu.edu, Pedro A. Segura Chávez, psegura@atmosfera.unam.mx, Pablo Camacho-Gonzalez,pablopcg1@ciencias.unam.mx, Jorge Zavala-Hidalgo, jzavala@atmosfera.unam.mx, Pavel Oropeza Alfaro, poropeza@atmosfera.unam.mx , Agustin R. Garcia, agustin@atmosfera.unam.mx'
                        ],
                        style={
                            'text-align': 'left',
                            'padding': '7px',
                            # 'background-color': '#f1f1f1',
                            'font-family': 'Helvetica'
                        }),
                    html.P(
                        [
                            'Referencia:  ',
                            html.A(f'doi:XXXXXXX',
                                   href='https://doi.org/XXXXXXX',
                                   target='_blank')
                        ],
                        style={
                            'text-align': 'left',
                            'padding': '20px',
                            # 'background-color': '#f1f1f1',
                            'font-family': 'Helvetica'
                        })
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
# debug_version = True
title_dash = 'Pronóstico de Calidad del Aire Mediante Redes Neuronales: Nivel de Ozono de la RAMA'

if debug_version:
    title_dash = 'Ver_Desarrollo/Depuración:' + title_dash

app = Dash(__name__, title=title_dash,suppress_callback_exceptions=True)
server = app.server

# Establecer el layout inicial de la aplicación para escuchar cambios en la URL
# que incluye el Dropdown
app.layout =  html.Div(
        [
            dcc.Location(id='url', refresh=False),
            # html.Div(id='page-content'),
            
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
                    'padding': '10px',
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
                    html.Label('Estación seleccionada:'),
                    dcc.Dropdown(
                        id='my-dropdown',  # Asegúrate de asignar un ID único
                        options=dropdown_options,
                        value='MER'  # Valor predeterminado
                    )
                ],
                         style={
                             'text-align': 'left',
                             'margin-top': '20px'
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


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'search'), Input('my-dropdown', 'value')],
    [State('url', 'search')]
)

def update_content(url_search, dropdown_value, state_url_search):
    ctx = callback_context

    # Si el callback no fue disparado por ningún Input, usa un valor predeterminado
    if not ctx.triggered:
        return generate_layout('UAX')  # Valor por defecto

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Determina qué input disparó el callback
    if trigger_id == 'url':
        # Si el cambio proviene de la URL
        id_est = url_search.split('=')[-1] if url_search else 'UAX'
    else:
        # Si el cambio proviene del dropdown
        id_est = dropdown_value if dropdown_value else state_url_search.split('=')[-1]

    return generate_layout(id_est)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=debug_version, host='0.0.0.0', port=8050)



 

