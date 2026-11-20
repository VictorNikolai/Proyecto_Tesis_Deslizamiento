"""
Deja listas para QGIS las teselas que bajó descargar_desde_puntos.py.

Por cada tesela (images/<id>/image.tif, 8 bandas) escribe en <proyecto>/qgis/:
  <id>_color.tif     color natural (B4, B3, B2) en 8 bits, para verla tal cual en QGIS
  <id>_pendiente.tif pendiente en grados, para los casos dudosos
y además:
  cajas_modelo.geojson   las zonas que marca hoy la U-Net, como guía (no son verdad)
  teselas.geojson        el borde de cada tesela con su id
  revisadas.csv          lista de teselas; pon «si» en la columna revisada al terminar cada una

  python herramientas/preparar_para_qgis.py iris_laconvencion
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('proyecto')
    ap.add_argument('--brillo', type=float, default=1500, help='valor de B4, B3 y B2 que queda en blanco')
    a = ap.parse_args()

    import rasterio
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from scipy import ndimage

    salida = os.path.join(a.proyecto, 'qgis')
    os.makedirs(salida, exist_ok=True)
    cajas, bordes, ids = [], [], []
    for ruta in sorted(glob.glob(os.path.join(a.proyecto, 'images', '*', 'image.tif'))):
        idx = os.path.basename(os.path.dirname(ruta))
        ids.append(idx)
        with rasterio.open(ruta) as ds:
            x = ds.read(); perfil = ds.profile; tr = ds.transform; crs = ds.crs
        meta_p = ruta.replace('image.tif', 'metadata.json')
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        rgb = np.clip(np.stack([x[3], x[2], x[1]]) / a.brillo * 255, 0, 255).astype('uint8')
        p8 = dict(perfil, count=3, dtype='uint8', nodata=None, compress='deflate', photometric='RGB')
        with rasterio.open(os.path.join(salida, f'{idx}_color.tif'), 'w', **p8) as d:
            d.write(rgb)
        p1 = dict(perfil, count=1, dtype='float32', compress='deflate')
        with rasterio.open(os.path.join(salida, f'{idx}_pendiente.tif'), 'w', **p1) as d:
            d.write(x[4].astype('float32'), 1)
        # borde de la tesela
        h, w = x.shape[1:]
        anillo = [tr * (0, 0), tr * (w, 0), tr * (w, h), tr * (0, h), tr * (0, 0)]
        geo = transform_geom(crs, 'EPSG:4326', {'type': 'Polygon', 'coordinates': [anillo]})
        props = {'tesela': idx, 'sinpad': (meta.get('sinpad') or {}).get('id'),
                 'fecha_evento': (meta.get('sinpad') or {}).get('fecha_evento'),
                 'imagen': '/'.join(meta.get('fechas', []))}
        bordes.append({'type': 'Feature', 'properties': props, 'geometry': geo})
        # cajas del modelo al umbral con que se exportó
        u = float(meta.get('umbral_cajas', meta.get('umbral_modelo', 0.45)))
        lab, n = ndimage.label(x[6] >= u)
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            f0, f1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            r = [tr * (c0, f0), tr * (c1, f0), tr * (c1, f1), tr * (c0, f1), tr * (c0, f0)]
            cajas.append({'type': 'Feature', 'properties': {'tesela': idx,
                          'prob_max': round(float(x[6][sl][lab[sl] == k].max()), 2)},
                          'geometry': transform_geom(crs, 'EPSG:4326', {'type': 'Polygon', 'coordinates': [r]})})
    if not ids:
        raise SystemExit(f'no hay teselas en {a.proyecto}/images: primero corre descargar_desde_puntos.py')
    json.dump({'type': 'FeatureCollection', 'features': cajas},
              open(os.path.join(salida, 'cajas_modelo.geojson'), 'w'))
    json.dump({'type': 'FeatureCollection', 'features': bordes},
              open(os.path.join(salida, 'teselas.geojson'), 'w'), ensure_ascii=False)
    ruta_rev = os.path.join(a.proyecto, 'revisadas.csv')
    previas = {}
    if os.path.exists(ruta_rev):
        previas = {r['tesela']: r for r in csv.DictReader(open(ruta_rev, encoding='utf-8-sig'))}
    with open(ruta_rev, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['tesela', 'revisada', 'quien', 'nota'])
        wr.writeheader()
        for idx in ids:
            wr.writerow(previas.get(idx, {'tesela': idx, 'revisada': '', 'quien': '', 'nota': ''}))
    print(f'{len(ids)} teselas listas en {os.path.abspath(salida)}')
    print(f'{len(cajas)} cajas del modelo en cajas_modelo.geojson')
    print(f'al terminar cada tesela, pon «si» en {os.path.abspath(ruta_rev)}')


if __name__ == '__main__':
    main()
