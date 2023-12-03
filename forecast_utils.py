import psycopg2
import netrc
import pandas as pd
from datetime import datetime
import pytz

# Reusando funciones de queries.py, y sqlCont.py

def getPollutantFromDateRange(conn, table, start_date, end_date, stations):
    """ Gets all the table names of our DB"""
    cur = conn.cursor();
    stations_str = "','".join(stations)
    print(stations_str)
    sql = F""" SELECT fecha, val, id_est FROM {table} 
                WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
                AND id_est IN ('{stations_str}')
                ORDER BY fecha;"""
    print(sql)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return rows

def getPostgresConn():
    """ Makes the connection with the DB"""
    print("Connecting to database....")
    secrets = netrc.netrc()
    login, account, passw = secrets.hosts['OWGIS']

    host ='132.248.8.238'
    # host ='localhost'
    #For Posgresql only
    try:
        conn = psycopg2.connect(database="contingencia", user=login, host=host, password=passw)
    except:
        print("Failed to connect to database")

    print(F"Connected to {host}")

    return conn

def getContaminantes(conn):
    """ Gets all the table names of our DB"""
    cur = conn.cursor();
    cur.execute(""" SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'; """);
    rows = cur.fetchall();
    cur.close();

    return rows

def getPollutantFromDateRange(conn, table, start_date, end_date, stations):
    """ Gets all the table names of our DB"""
    cur = conn.cursor();
    stations_str = "','".join(stations)
    print(stations_str)
    sql = F""" SELECT fecha, val, id_est FROM {table} 
                WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
                AND id_est IN ('{stations_str}')
                ORDER BY fecha;"""
    # print(sql)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return rows

def getDataFrameFromDateRange(table, start_date, end_date, stations):
    """
    example of arguments: getDataFrameFromDateRange('cont_otres', '2017-01-01', '2018-12-12', ['AJM', 'AJU'])
    """
    conn = getPostgresConn()
    cur_data = getPollutantFromDateRange(conn, table, start_date, end_date, stations)
    df = pd.DataFrame(cur_data, columns=["fecha", "val", "id_est"])
    conn.close()
    
    return df

# Crear widgets
# TODO: sacar estaciones directamente de la base de datos
all_stations = ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
          ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
          ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
          ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
          , "XCH"]

sql_tables = ["cont_otres","cont_co","cont_codos","cont_estaciones","cont_no","cont_nodos","cont_nox","cont_pmco","cont_pmdiez","cont_sodos","cont_pmdoscinco","cont_monthly_onesixsix","cont_year_onesixsix"]

def getDataFrameFromDateRange(query):
    conn = getPostgresConn()
    cur = conn.cursor()

    # Agregar la fecha de inicio y fin a la consulta
    query = f"""{query};"""
    cur.execute(query)

    # Obtener los nombres de las columnas de la tabla
    columns = [desc[0] for desc in cur.description]
    print(cur.description)

    # Obtener los datos de la tabla y cargarlos en un DataFrame de Pandas
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=columns)

    # Cerrar la conexión a la base de datos
    cur.close()
    conn.close()

    return df


def calculate_color(value):
    # Convert the value to a percentage between 0 and 1
    value = value
    # Initialize RGB values
    red, green = 0, 0

    if value <= 0.65:  # Green to Yellow
        red = int(255 * (value / 0.5))
        green = 255
    else:  # Yellow to Red
        red = 255
        green = int(255 * ((1 - value) / 0.5))

    return f'rgba({red}, {green}, 0, {0.5 + (0.5*(value))})'


############################################################
# Simulate data for the additional dials
# Note: In your real code, you would calculate these probabilities based on your model
# probabilities = [np.random.uniform(0, 1) for _ in range(4)]
# ###########################################################
from scipy.stats import norm

def probability_2pass_threshold(forecast_level, mu, sigma, thresholdX):
    """
    Calcula la probabilidad de que la desviación en un pronóstico sea suficientemente grande
    como para que el total del pronóstico (nivel del pronóstico más la desviación) supere un umbral específico.

    :param forecast_level: Nivel del pronóstico.
    :param mu: Media de la desviación del pronóstico.
    :param sigma: Desviación estándar de la desviación del pronóstico.
    :param thresholdX: El umbral a superar.
    :return: Probabilidad de superar el umbral en porcentaje.
    """
    # Calcular la diferencia necesaria para superar el umbral
    difference_to_threshold = thresholdX - forecast_level

    # Calcular la probabilidad de que la desviación sea mayor o igual a esa diferencia
    probability = 1 - norm.cdf(difference_to_threshold/1.96, mu, sigma)

    # Convertir a porcentaje
    #probability_percent = probability
    print(forecast_level,mu,sigma,thresholdX)
    print(f'probabilida de {probability}, f_lev:{forecast_level}, threshold:{thresholdX}')
    return probability #_percent




def moving_average_probabilities(values, window_size, mu, sigma, thresholdX):
    mean8probs = []

    # Calcular la media móvil y la probabilidad correspondiente
    for i in range(len(values) - window_size + 1):
        mean_value = values[i:i+window_size].mean()
        prob = probability_2pass_threshold(mean_value, mu, sigma, thresholdX)
        mean8probs.append(prob)

    return mean8probs
