"""
Carga de modelos ONNX con su receta, y la comprobación de que están entrenados.

Los ONNX que había en el repositorio corrían sin error y no detectaban nada: sus
pesos eran los de inicialización de PyTorch, porque al exportar no se cargó el
checkpoint. Un fallo así no se ve en la forma del grafo ni en un error de
ejecución, solo en los pesos. Por eso cada modelo pasa por `revisar_pesos` antes
de usarse: en una capa recién inicializada, el peso máximo es exactamente
1/sqrt(fan_in), la media de |w| es la mitad y la desviación la de una uniforme.
Una capa entrenada se aleja de eso.
"""
import glob
import hashlib
import os
import threading

import numpy as np

from . import config, recetas

CABECERA_LFS = b'version https://git-lfs'
_LOCK = threading.Lock()
_SESIONES = {}
_FICHAS = {}
_RUTA_SESION = {}     # nombre -> ONNX con el que se abre la sesión


# ------------------------------------------------------------------ archivos
def es_puntero_lfs(ruta):
    try:
        with open(ruta, 'rb') as f:
            return f.read(len(CABECERA_LFS)) == CABECERA_LFS
    except OSError:
        return False


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 20), b''):
            h.update(bloque)
    return h.hexdigest()


def _asegurar_archivo(ruta):
    """Si falta o es un puntero de Git LFS, lo baja de URL_PESOS_BASE."""
    if os.path.exists(ruta) and not es_puntero_lfs(ruta):
        return
    if not config.URL_PESOS_BASE:
        raise FileNotFoundError(
            f'{os.path.basename(ruta)} falta o llegó como puntero de Git LFS, y no hay '
            'URL_PESOS_BASE para bajarlo')
    import httpx
    url = f'{config.URL_PESOS_BASE.rstrip("/")}/{os.path.basename(ruta)}'
    print(f'[pesos] bajando {url}', flush=True)
    tmp = ruta + '.parcial'
    with httpx.stream('GET', url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for trozo in r.iter_bytes(1 << 20):
                f.write(trozo)
    if es_puntero_lfs(tmp):
        os.remove(tmp)
        raise RuntimeError(f'{url} devolvió otro puntero LFS, no el archivo')
    os.replace(tmp, ruta)


# --------------------------------------------------------- prueba de los pesos
def _parece_inicializacion(w):
    """¿Esta capa conv tiene la forma exacta de una inicialización y no de un
    peso entrenado? Se reconocen las de PyTorch por defecto, Kaiming y Xavier,
    uniformes o normales, que son las que usan torchvision y
    segmentation_models_pytorch. Un peso entrenado tiene colas y no calza con
    ninguna a la vez en forma y en escala."""
    w = np.asarray(w, np.float64)
    fi = w.shape[1] * w.shape[2] * w.shape[3]
    fo = w.shape[0] * w.shape[2] * w.shape[3]
    c = np.abs(w).max()
    s = w.std()
    if c == 0 or s == 0:
        return False
    cotas = (1 / np.sqrt(fi), np.sqrt(3 / fi), np.sqrt(6 / fi), np.sqrt(6 / (fi + fo)),
             np.sqrt(3 / fo), np.sqrt(6 / fo))
    uniforme = (0.47 < np.abs(w).mean() / c < 0.53 and 0.95 < s / (c / np.sqrt(3)) < 1.05
                and any(abs(c / k - 1) < 0.03 for k in cotas))
    if uniforme:
        return True
    z = (w - w.mean()) / s
    exceso_curtosis = float((z ** 4).mean() - 3)
    desvs = (np.sqrt(2 / fi), np.sqrt(2 / fo), np.sqrt(1 / fi), np.sqrt(2 / (fi + fo)))
    normal = (abs(w.mean()) < 0.05 * s and abs(exceso_curtosis) < 0.15
              and any(abs(s / k - 1) < 0.02 for k in desvs))
    return bool(normal)


def _forma(valor):
    return [d.dim_value if d.HasField('dim_value') else 'N'
            for d in valor.type.tensor_type.shape.dim]


def revisar_pesos(ruta_onnx, copia_aparte=None):
    """Cuántas capas conv tienen pesos indistinguibles de la inicialización
    uniforme de PyTorch, U(-1/sqrt(fan_in), +1/sqrt(fan_in)). De paso devuelve
    la forma de la entrada y de la salida, para no abrir una sesión solo para eso."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(ruta_onnx, load_external_data=True)
    pesos = {t.name: t for t in m.graph.initializer}
    nombres_init = set(pesos)
    entradas = [i for i in m.graph.input if i.name not in nombres_init]
    forma_in = _forma(entradas[0]) if entradas else []
    forma_out = _forma(m.graph.output[0]) if m.graph.output else []
    marcas = []                      # en el orden del grafo
    for nodo in m.graph.node:
        if nodo.op_type != 'Conv' or len(nodo.input) < 2 or nodo.input[1] not in pesos:
            continue
        t = pesos[nodo.input[1]]
        if len(t.dims) != 4 or int(np.prod(t.dims)) < 256:
            continue
        marcas.append(_parece_inicializacion(numpy_helper.to_array(t)))
    del pesos
    ruta_sesion = _guardar_pesos_aparte(m, ruta_onnx, copia_aparte) if copia_aparte else ruta_onnx
    del m
    liberar_memoria()
    convs, en_init = len(marcas), sum(marcas)
    frac = en_init / convs if convs else 0.0
    cabeza = any(marcas[-3:])
    return {'capas_conv': convs, 'capas_en_inicializacion': en_init,
            'fraccion': round(frac, 3), 'cabeza_sin_entrenar': cabeza,
            # la cabeza es la que decide la clase: si está en la inicialización el
            # modelo no aprendió a segmentar, aunque el codificador traiga ImageNet
            'sin_entrenar': frac >= 0.25 or cabeza,
            '_forma_entrada': forma_in, '_forma_salida': forma_out,
            '_ruta_sesion': ruta_sesion}


def _guardar_pesos_aparte(m, ruta_onnx, destino):
    """Copia del ONNX con los pesos en un .onnx.data al lado. Si algo falla se usa
    el original: es una optimización de memoria, no un requisito."""
    import onnx
    try:
        if os.path.getsize(ruta_onnx) < 8 * 1024 * 1024:
            return ruta_onnx                  # modelos chicos: no vale la pena
        os.makedirs(destino, exist_ok=True)
        base = os.path.splitext(os.path.basename(ruta_onnx))[0]
        salida = os.path.join(destino, base + '.onnx')
        datos_ext = salida + '.data'
        # onnx agrega al final de un .data existente en vez de reemplazarlo: sin
        # borrarlo antes, cada arranque lo haría crecer otros 90 MB
        for viejo in (salida, datos_ext):
            if os.path.exists(viejo):
                os.remove(viejo)
        esperado = sum(len(t.raw_data) for t in m.graph.initializer)
        onnx.save_model(m, salida, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=os.path.basename(datos_ext), size_threshold=1024)
        if os.path.getsize(datos_ext) > 1.05 * esperado + 1024 * 1024:
            raise RuntimeError(f'la copia de pesos pesa {os.path.getsize(datos_ext)} bytes')
        return salida
    except Exception as e:
        print(f'[modelos] sin copia con pesos aparte ({type(e).__name__}: {e}); '
              'se usa el ONNX original', flush=True)
        return ruta_onnx


def liberar_memoria():
    """Devuelve al sistema la memoria que Python ya soltó. Sin esto, glibc la
    retiene y el proceso parece ocupar más de lo que usa."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


# ----------------------------------------------------------------- catálogo
def catalogo():
    """nombre -> receta, leyendo modelos/*.json."""
    out = {}
    for ruta in sorted(glob.glob(os.path.join(config.CARPETA_MODELOS, '*.json'))):
        try:
            r = recetas.leer(ruta)
        except Exception as e:
            print(f'[modelos] receta ignorada {os.path.basename(ruta)}: {e}', flush=True)
            continue
        r['_ruta_onnx'] = os.path.join(config.CARPETA_MODELOS, r['onnx'])
        out[r['nombre']] = r
    return out


def ficha(nombre):
    """Lo que /modelos publica de cada modelo. Se calcula una vez."""
    if nombre in _FICHAS:
        return _FICHAS[nombre]
    cat = catalogo()
    if nombre not in cat:
        raise KeyError(nombre)
    r = cat[nombre]
    f = {'nombre': nombre, 'etiqueta': r.get('etiqueta', nombre),
         'autor': r.get('autor'), 'canales': r['canales'], 'lado_px': r['lado_px'],
         'salida': r['salida'], 'umbral': r['umbral'],
         'necesita_dem': recetas.necesita_dem(r),
         'entrenamiento': r.get('entrenamiento'),
         'verificacion_l4s': r.get('verificacion_l4s')}
    try:
        _asegurar_archivo(r['_ruta_onnx'])
        # La revisión carga el ONNX entero en memoria un momento; se hace antes de
        # cualquier sesión para que los dos picos no se sumen.
        pesos = revisar_pesos(r['_ruta_onnx'], config.CARPETA_COPIAS or None)
        _RUTA_SESION[nombre] = pesos.pop('_ruta_sesion')
        forma_in = pesos.pop('_forma_entrada')
        forma_out = pesos.pop('_forma_salida')
        f['entrada'] = {'forma': forma_in}
        f['salida_forma'] = forma_out
        problemas = []
        if len(forma_in) != 4:
            problemas.append(f'el ONNX tiene una entrada de forma {forma_in}; se esperaba N×C×H×W')
        else:
            c, h, w = forma_in[1], forma_in[2], forma_in[3]
            if c != len(r['canales']):
                problemas.append(f'el ONNX pide {c} canales y la receta da {len(r["canales"])}')
            if (h, w) != (r['lado_px'], r['lado_px']):
                problemas.append(f'el ONNX pide {h}x{w} y la receta dice {r["lado_px"]}')
        f['pesos'] = pesos
        f['sha256'] = sha256(r['_ruta_onnx'])
        if pesos['sin_entrenar']:
            f['estado'] = 'sin_entrenar'
            f['detalle'] = (f'{pesos["capas_en_inicializacion"]} de {pesos["capas_conv"]} '
                            'capas tienen los pesos de inicialización: este ONNX se '
                            'exportó sin cargar el entrenamiento y no detecta nada.')
        elif problemas:
            f['estado'] = 'error'
            f['detalle'] = '; '.join(problemas)
        else:
            f['estado'] = 'listo'
    except Exception as e:
        f['estado'] = 'error'
        f['detalle'] = f'{type(e).__name__}: {e}'
    _FICHAS[nombre] = f
    return f


def _crear_sesion(ruta):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = config.HILOS_ONNX
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = {
        'basica': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'extendida': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'todas': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }.get(config.NIVEL_OPT_ONNX, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    return ort.InferenceSession(ruta, so, providers=['CPUExecutionProvider'])


def receta_lista(nombre):
    """La receta de un modelo que puede usarse, sin cargar la sesión; o una
    excepción que explica por qué no. Sirve para rechazar antes de bajar imágenes."""
    cat = catalogo()
    if nombre not in cat:
        raise KeyError(f'modelo "{nombre}" no existe. Disponibles: {sorted(cat)}')
    f = ficha(nombre)
    if f['estado'] == 'sin_entrenar' and not config.PERMITIR_SIN_ENTRENAR:
        raise PermissionError(f['detalle'])
    if f['estado'] == 'error':
        raise RuntimeError(f['detalle'])
    return cat[nombre]


def obtener(nombre):
    """(sesión, receta) lista para inferir, o una excepción que explica por qué no."""
    receta = receta_lista(nombre)
    with _LOCK:
        if nombre not in _SESIONES:
            if config.UN_MODELO_A_LA_VEZ and _SESIONES:
                _SESIONES.clear()
                liberar_memoria()
            _SESIONES[nombre] = _crear_sesion(_RUTA_SESION.get(nombre, receta['_ruta_onnx']))
        return _SESIONES[nombre], receta


def soltar(nombre=None):
    """Descarga la sesión (o todas) para que la memoria quede libre entre trabajos.
    Volver a crearla cuesta menos de un segundo; tenerla cargada mientras se bajan
    las imágenes cuesta más de 100 MB en un plan de 512."""
    with _LOCK:
        if nombre is None:
            _SESIONES.clear()
        else:
            _SESIONES.pop(nombre, None)
    liberar_memoria()


def cargados():
    return sorted(_SESIONES)
