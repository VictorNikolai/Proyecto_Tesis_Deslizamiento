"""
Todo lo configurable del backend, en un solo sitio.

Cada valor se puede cambiar con una variable de entorno del mismo nombre, que es
como se ajusta en Render (Environment) sin tocar el código.
"""
import json
import os
import tempfile


def _env(nombre, defecto):
    return os.environ.get(nombre, defecto)


RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------- modelos
# Cada modelo es un .onnx más una receta .json con el mismo nombre de base, que
# dice qué canales recibe, cómo se normalizan, qué salida tiene y su umbral.
CARPETA_MODELOS = _env('CARPETA_MODELOS', os.path.join(RAIZ, 'modelos'))
# Si un .onnx llega como puntero de Git LFS, se baja de aquí: <URL>/<archivo>
URL_PESOS_BASE = _env('URL_PESOS_BASE', '')
# Un ONNX con pesos de inicialización no detecta nada aunque corra. Se rechaza
# salvo que esto valga 1, que solo sirve para probar el circuito.
PERMITIR_SIN_ENTRENAR = _env('PERMITIR_SIN_ENTRENAR', '0') == '1'
HILOS_ONNX = int(_env('HILOS_ONNX', '1'))
# Optimización del grafo: 'extendida' da los mismos resultados (diferencia < 1e-6)
# que 'todas' y usa unos 100 MB menos, porque no duplica los pesos en el formato
# interno NCHWc. En 512 MB eso decide si el servicio cabe o no.
NIVEL_OPT_ONNX = _env('NIVEL_OPT_ONNX', 'extendida')
# Hilos para bajar las imágenes. dask usa por defecto uno por núcleo de la
# máquina, y en Render la máquina física tiene muchos aunque el plan dé 0.1 CPU:
# cada hilo abre sus propias conexiones y cachés. Con 4 la descarga es rápida
# (la espera es de red, no de cálculo) y la memoria queda acotada.
HILOS_DESCARGA = int(_env('HILOS_DESCARGA', '4'))
# En el plan gratuito de Render hay 512 MB: mejor un modelo cargado a la vez.
UN_MODELO_A_LA_VEZ = _env('UN_MODELO_A_LA_VEZ', '1') == '1'
# La sesión se carga recién para inferir y se suelta al terminar cada trabajo, así
# no ocupa memoria mientras se bajan las imágenes. Con más RAM se puede poner 0.
SOLTAR_SESION = _env('SOLTAR_SESION', '1') == '1'
# Parches por llamada al modelo. 4 da la misma velocidad que 8 en CPU y ~20 MB menos.
LOTE_INFERENCIA = int(_env('LOTE_INFERENCIA', '4'))
# Al revisar cada modelo, el servidor guarda aquí una copia con los pesos en un
# archivo aparte (.onnx.data). ONNX Runtime los lee directo del disco: unos 95 MB
# dejan de contar como memoria propia del proceso y el resultado es idéntico.
# Vacío para desactivarlo.
CARPETA_COPIAS = _env('CARPETA_COPIAS', os.path.join(
    tempfile.gettempdir(), 'modelos_pesos_aparte'))

# ----------------------------------------------------------------------- CORS
# La URL del Space, p. ej. https://usuario-espacio.hf.space. '*' para probar.
ORIGENES = [o.strip() for o in _env('ORIGENES_PERMITIDOS', '*').split(',') if o.strip()]

# ---------------------------------------------------------- fuente de imágenes
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COLECCION_S2 = 'sentinel-2-l2a'
# ----------------------------------------------------- relieve (B13 y B14)
# Landslide4Sense toma la pendiente (B13) y el DEM (B14) del «ALOS PALSAR DEM» de
# 12.5 m que distribuye ASF con los productos RTC. Ese DEM no sale de PALSAR: es
# SRTM de 30 m remuestreado a 12.5 m (NASA Earthdata, proyecto ALOS PALSAR RTC).
# Por eso la fuente por defecto, que cubo pide a Planetary Computer, es NASADEM
# (SRTM reprocesado, 30 m). Otras opciones:
#   alos-aw3d30   ALOS World 3D-30m de JAXA (óptico PRISM, misión ALOS)
#   cop-dem-glo-30 Copernicus GLO-30
#   asf-alos-palsar el mosaico de los .dem.tif de ASF para la provincia, armado
#                  una vez con herramientas/alos_palsar_asf.py (archivo o URL COG)
# Ninguna es una serie temporal: cada una va con su propia ventana de fechas.
FUENTES_DEM = {
    'nasadem': {'coleccion': 'nasadem', 'banda': 'elevation',
                'fechas': ('2000-01-01', '2000-12-31'),
                'nombre': 'NASADEM (SRTM, 30 m), la base del DEM «ALOS PALSAR» de ASF'},
    'alos-aw3d30': {'coleccion': 'alos-dem', 'banda': 'data',
                    'fechas': ('2016-01-01', '2016-12-31'),
                    'nombre': 'ALOS World 3D-30m (JAXA, 30 m)'},
    'cop-dem-glo-30': {'coleccion': 'cop-dem-glo-30', 'banda': 'data',
                       'fechas': ('2020-01-01', '2023-12-31'),
                       'nombre': 'Copernicus DEM GLO-30 (30 m)'},
    'asf-alos-palsar': {'archivo': True,
                        'nombre': 'ALOS PALSAR RTC (ASF), mosaico de la provincia'},
}
FUENTE_DEM = _env('FUENTE_DEM', 'nasadem')
if FUENTE_DEM not in FUENTES_DEM:
    raise ValueError(f'FUENTE_DEM="{FUENTE_DEM}" no es una de {sorted(FUENTES_DEM)}')
# Para asf-alos-palsar: ruta local o URL (https://...) del GeoTIFF del mosaico.
RUTA_DEM_ASF = _env('RUTA_DEM_ASF', '')
# Constantes de Landslide4Sense, medidas en sus parches (son las mismas en todas
# las regiones): pendiente = grados / 20.292 y elevación = metros / 772.82.
ESCALA_TERRENO_L4S = {'pendiente': 20.292, 'elevacion': 772.82}
RESOLUCION_M = 10
# Lado de cada cubo de cubo. 256 px son 2.56 km: pocas llamadas y poca memoria.
LADO_TESELA_PX = int(_env('LADO_TESELA_PX', '256'))
# Escenas por tesela, las menos nubosas según el metadato. Acota memoria y tiempo.
MAX_ESCENAS = int(_env('MAX_ESCENAS', '6'))
# Clases SCL que se descartan: sin dato, saturado, sombra de nube, nube media,
# nube alta y cirro.
SCL_MALAS = (0, 1, 3, 8, 9, 10)
# Desde la baseline 04.00 (enero de 2022) los niveles digitales de L2A llevan
# sumado 1000. Se resta por escena.
DESPLAZAMIENTO_L2A = 1000.0

# ---------------------------------------------- unidades de Landslide4Sense
# Landslide4Sense trae cada banda dividida por su media de escena. Estas son las
# medias de La Convención (reflectancia L2A en niveles digitales), medidas con
# herramientas/calibrar_escala.py sobre la provincia en temporada seca.
_RUTA_ESCALA = _env('RUTA_ESCALA', os.path.join(RAIZ, 'escala_laconvencion.json'))


def _cargar_escala():
    try:
        with open(_RUTA_ESCALA, encoding='utf-8') as f:
            return json.load(f)['media_dn']
    except Exception:
        return None


ESCALA_L4S = _cargar_escala()

# ------------------------------------------------------------------- límites
MAX_LADO_KM = float(_env('MAX_LADO_KM', '12'))
AVISO_LADO_KM = float(_env('AVISO_LADO_KM', '6'))
SOLAPE = float(_env('SOLAPE', '0.5'))
COBERTURA_MIN = float(_env('COBERTURA_MIN', '0.30'))
MAX_COLA = int(_env('MAX_COLA', '4'))
MINUTOS_RESULTADO = int(_env('MINUTOS_RESULTADO', '30'))

# -------------------------------------------------------------------- ámbito
#   'bloquear': rechaza con 403 lo que caiga fuera. Es lo que va en la entrega.
#   'avisar'  : deja correr y marca el resultado como fuera del ámbito.
#   'libre'   : sin límite, solo para pruebas.
AMBITO = _env('AMBITO', 'bloquear')
NOMBRE_AMBITO = 'provincia de La Convención, Cusco'
RUTA_LIMITE = _env('RUTA_LIMITE', os.path.join(RAIZ, 'limite_laconvencion.geojson'))
BBOX_ESTUDIO = (-13.4489, -73.9805, -11.1988, -71.9730)   # sur, oeste, norte, este
MIN_DENTRO = float(_env('MIN_DENTRO', '0.5'))

# --------------------------------------------------------------------- pruebas
# A 1 no sale a la red: fabrica un área sintética con una cicatriz.
MODO_FALSO = _env('MODO_FALSO', '0') == '1'
CENTRO_POR_DEFECTO = (-12.86, -72.69)    # Quillabamba
