# Instalación de Página de Visualización del Pronóstico Operativo de Calidad del Aire

## Prerrequisitos

Asegúrate de tener instalado el gestor de paquetes Conda/Mamba en tu sistema. Para instrucciones de instalación, visita [Mamba Installation](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

## Instrucciones de Instalación

1. **Clona el Repositorio**:
   Clona este repositorio en tu máquina local usando `git clone`, seguido de la URL del repositorio.
   ```bash
   git clone https://github.com/PA-Segura/forecast_pollution_page/
   ```

2. **Crear Entorno Conda**:
   Utiliza Mamba para crear el entorno desde el archivo `dash-operativo.yml`.
   ```bash
   mamba env create -f dash-operativo.yml
   ```

3. **Activar Entorno**:
   Activa el entorno con Mamba.
   ```bash
   mamba activate dash-operativo
   ```

4. **Configura Archivo `gunicorn_config.py` y `.netrc`**:
   Asegúrate de configurar el archivo `gunicorn_config.py` y `.netrc` para tener el puerto especificado, recursos y credenciales de acceso a la base de datos de pronósticos.

5. **Ejecución**:
   Una vez configurado el entorno y el archivo `.netrc`, puedes iniciar la aplicación con el siguiente comando:
   ```bash
   gunicorn -c gunicorn_config.py dash_ozono_v002:server
   ```
