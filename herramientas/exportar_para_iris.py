"""
Arma un proyecto de IRIS (ESA-PhiLab) con teselas reales de La Convención, para
marcar a mano los deslizamientos y tener por fin datos etiquetados de la provincia.

Cada tesela es un GeoTIFF de 7 bandas, en UTM a 10 m, con:
  1-4  B1, B2, B3, B4 de Sentinel-2 L2A (la misma mediana sin nubes que usa el sistema)
  5    B13 pendiente en grados
  6    B14 elevación en metros
  7    probabilidad del modelo elegido (por defecto el ensamble), como guía
  8    contorno de una caja alrededor de cada zona del modelo (umbral sensible 0,30),
       para la vista «Color natural con cajas»: se revisa caja por caja y la forma
       del deslizamiento se dibuja a mano

Usa el código del servidor tal cual (cubo, Planetary Computer y los ONNX de modelos/),
así que se corre en la computadora con el entorno del Paso 1, no en Render.

  python herramientas/exportar_para_iris.py --sur -12.89 --oeste -72.72 --norte -12.84 --este -72.67 \
      --desde 2026-06-01 --hasta 2026-09-30 --salida iris_laconvencion

Luego, con IRIS instalado (Python 3.9):  iris label iris_laconvencion/deslizamientos.json
Y para llevar lo marcado a QGIS y medir el modelo:  python herramientas/importar_de_iris.py iris_laconvencion
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'backend'))
sys.path.insert(0, os.path.join(AQUI, '..'))          # en el repositorio nuevo, el servidor está en la raíz

NOMBRES = ['B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)', 'probabilidad del modelo',
           'cajas del modelo (contorno)']


def contorno_cajas(prob, umbral=0.30, margen=3):
    """1 en el borde de una caja alrededor de cada zona con prob >= umbral; 0 en el resto."""
    from scipy import ndimage
    H, W = prob.shape
    borde = np.zeros((H, W), np.float32)
    lab, _ = ndimage.label(np.nan_to_num(prob) >= umbral)
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        f0, f1 = max(sl[0].start - margen, 0), min(sl[0].stop + margen, H) - 1
        c0, c1 = max(sl[1].start - margen, 0), min(sl[1].stop + margen, W) - 1
        borde[f0, c0:c1 + 1] = borde[f1, c0:c1 + 1] = 1
        borde[f0:f1 + 1, c0] = borde[f0:f1 + 1, c1] = 1
    return borde


def configuracion(nombre, lado, puerto):
    # La ruta va como diccionario y las bandas como $S2.Bn: así funciona igual en IRIS oficial y en el
    # fork gt-csse, que convierte una ruta de texto en {'pictures': ...} y deja de encontrar $Bn.
    return {
        'name': nombre, 'port': puerto,
        'images': {'path': {'S2': 'images/{id}/image.tif'}, 'shape': [lado, lado], 'metadata': 'images/{id}/metadata.json'},
        'segmentation': {'path': 'masks/{id}.png', 'mask_encoding': 'integer',
                         'mask_area': [0, 0, lado, lado], 'score': 'f1',
                         'unverified_threshold': 1, 'test_images': None},
        'classes': [
            {'name': 'Sin deslizamiento', 'colour': [255, 255, 255, 0], 'user_colour': [0, 255, 255, 70],
             'description': 'Ladera con vegetación, río, techo, camino, cultivo o suelo que no es un deslizamiento.'},
            {'name': 'Deslizamiento', 'colour': [255, 40, 60, 110],
             'description': 'Cicatriz de deslizamiento: suelo desnudo en ladera, con escarpa arriba y material arrastrado hacia abajo.'},
            {'name': 'Dudoso', 'colour': [255, 200, 0, 90],
             'description': 'No se puede decidir con esta imagen. No cuenta ni a favor ni en contra del modelo.'}],
        'views': {
            'Color natural con cajas': {
                'description': 'Cada caja amarilla es una zona que marcó el modelo. Dentro, pinta la forma real; '
                               'si no hay deslizamiento, píntala «Sin deslizamiento». Revisa también fuera de las cajas.',
                'type': 'image', 'vmin': 0, 'vmax': 1500,
                'data': ['$S2.B4*(1-$S2.B8)+5000*$S2.B8', '$S2.B3*(1-$S2.B8)+5000*$S2.B8', '$S2.B2*(1-$S2.B8)']},
            'Color natural': {'description': 'B4, B3 y B2 de Sentinel-2.', 'type': 'image', 'vmin': 0, 'vmax': 1500,
                              'data': ['$S2.B4', '$S2.B3', '$S2.B2']},
            'Probabilidad del modelo': {'description': 'Solo de guía: lo que cuenta es lo que se ve en la imagen.',
                                        'type': 'image', 'data': '$S2.B7', 'cmap': 'magma', 'vmin': 0, 'vmax': 1},
            'Pendiente': {'description': 'B13 en grados.', 'type': 'image', 'data': '$S2.B5',
                          'cmap': 'viridis', 'vmin': 0, 'vmax': 60},
            'Relieve': {'description': 'B14 en metros.', 'type': 'image', 'data': '$S2.B6',
                        'cmap': 'terrain', 'vmin': 500, 'vmax': 3500},
            'Bordes': {'description': 'Bordes en las bandas visibles: ayudan a ver escarpas.', 'type': 'image',
                       'data': 'edges($S2.B2+$S2.B3+$S2.B4)', 'cmap': 'gray'},
            'Mapa': {'description': 'Mapa de Bing de la zona.', 'type': 'bingmap'}},
        'view_groups': {'default': ['Color natural con cajas', 'Color natural', 'Pendiente'],
                        'probabilidad': ['Color natural', 'Probabilidad del modelo', 'Pendiente'],
                        'relieve': ['Color natural', 'Relieve', 'Bordes'],
                        'mapa': ['Color natural', 'Mapa']},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    for k in ('sur', 'oeste', 'norte', 'este'):
        ap.add_argument(f'--{k}', type=float, required=True)
    ap.add_argument('--desde', required=True)
    ap.add_argument('--hasta', required=True)
    ap.add_argument('--modelo', default='ensamble')
    ap.add_argument('--nubes', type=float, default=40)
    ap.add_argument('--lado', type=int, default=256, help='lado de cada tesela en píxeles')
    ap.add_argument('--salida', default='iris_laconvencion')
    ap.add_argument('--prefijo', default='tesela')
    ap.add_argument('--puerto', type=int, default=5000)
    ap.add_argument('--umbral_cajas', type=float, default=0.30, help='umbral para dibujar las cajas')
    ap.add_argument('--margen_cajas', type=int, default=3, help='holgura de las cajas en píxeles')
    a = ap.parse_args()

    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import transform as transformar
    from nucleo import datos, procesar

    p = dict(sur=a.sur, oeste=a.oeste, norte=a.norte, este=a.este, desde=a.desde, hasta=a.hasta,
             modelo=a.modelo, nubes_max=a.nubes)
    print('Detectando con', a.modelo, '(baja las imágenes y corre el modelo)...')
    r, extra = procesar.procesar(p, lambda f, t: print(f'  {t}', flush=True))
    z = zipfile.ZipFile(io.BytesIO(extra['zip'][0]))
    with rasterio.MemoryFile(z.read('probabilidad.tif')) as mf, mf.open() as ds:
        prob, tr, crs = ds.read(1), ds.transform, ds.crs
    prob = np.where(prob < 0, 0, prob)
    print('Armando las bandas en la misma malla...')
    plan = datos.planificar((a.sur, a.oeste, a.norte, a.este), lado_min=128)
    mos = datos.componer(plan, a.desde, a.hasta, ['B01', 'B02', 'B03', 'B04'], True, nubes_max=a.nubes)
    cortar = lambda x: datos.recortar(x, plan['ventana'])
    capas = [cortar(mos['bandas'][b]) for b in ('B01', 'B02', 'B03', 'B04')] + \
            [cortar(mos['terreno']['pendiente']), cortar(mos['terreno']['elevacion']), prob]
    capas.append(contorno_cajas(prob, a.umbral_cajas, a.margen_cajas))
    H, W = prob.shape
    L = a.lado
    os.makedirs(os.path.join(a.salida, 'images'), exist_ok=True)
    n = 0
    filas = list(range(0, max(H - L, 0) + 1, L)) or [0]
    cols = list(range(0, max(W - L, 0) + 1, L)) or [0]
    for f0 in filas:
        for c0 in cols:
            n += 1
            idx = f'{a.prefijo}_{n:03d}'
            d = os.path.join(a.salida, 'images', idx)
            os.makedirs(d, exist_ok=True)
            t = tr * Affine.translation(c0, f0)
            with rasterio.open(os.path.join(d, 'image.tif'), 'w', driver='GTiff', width=L, height=L, count=8,
                               dtype='float32', crs=crs, transform=t, compress='deflate') as ds:
                for k, c in enumerate(capas, 1):
                    bloque = np.zeros((L, L), np.float32)
                    trozo = c[f0:f0 + L, c0:c0 + L]
                    bloque[:trozo.shape[0], :trozo.shape[1]] = np.nan_to_num(trozo, nan=0)
                    ds.write(bloque, k)
                    ds.set_band_description(k, NOMBRES[k - 1])
            cx, cy = t * (L / 2, L / 2)
            lon, lat = transformar(crs, 'EPSG:4326', [cx], [cy])
            json.dump({'location': [lon[0], lat[0]], 'fechas': [a.desde, a.hasta], 'modelo': a.modelo,
                       'umbral_modelo': r['umbral'], 'umbral_cajas': a.umbral_cajas, 'escenas': [e['fecha'] for e in r['escenas']]},
                      open(os.path.join(d, 'metadata.json'), 'w'), indent=1)
    json.dump(configuracion('deslizamientos-laconvencion', L, a.puerto),
              open(os.path.join(a.salida, 'deslizamientos.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'\nListo: {n} teselas de {L} x {L} px en {os.path.abspath(a.salida)}')
    print(f'Abre IRIS con:  iris label {os.path.join(a.salida, "deslizamientos.json")}')


if __name__ == '__main__':
    main()
