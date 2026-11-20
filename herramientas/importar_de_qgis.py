"""
Convierte los polígonos dibujados en QGIS en máscaras para entrenar y medir el modelo.

Lee una capa de polígonos exportada de QGIS como GeoJSON (por ejemplo marcas.geojson),
con un campo de texto «clase» que vale:
  deslizamiento   la cicatriz de un deslizamiento
  dudoso          no se puede decidir con la imagen (no cuenta ni a favor ni en contra)
Lo que no tiene polígono es «sin deslizamiento».

Solo usa las teselas con «si» en revisadas.csv: una tesela sin revisar no tiene
polígonos y parecería que no hay ningún deslizamiento.

Escribe <proyecto>/masks/<id>.png (0 fondo, 1 deslizamiento, 2 dudoso) en la malla de
cada tesela, el mismo formato que IRIS, así que después se mide con:

  python herramientas/importar_de_qgis.py iris_laconvencion marcas.geojson
  python herramientas/importar_de_iris.py iris_laconvencion --umbral 0.45
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

VALOR = {'deslizamiento': 1, 'dudoso': 2}


def normalizar(texto):
    t = str(texto or '').strip().lower()
    for a, b in (('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u')):
        t = t.replace(a, b)
    return {'1': 'deslizamiento', 'si': 'deslizamiento', '2': 'dudoso'}.get(t, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('proyecto')
    ap.add_argument('marcas', help='GeoJSON exportado de QGIS con el campo «clase»')
    a = ap.parse_args()

    import rasterio
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    from shapely.geometry import box, shape

    gj = json.load(open(a.marcas, encoding='utf-8'))
    crs_marcas = 'EPSG:4326'
    nombre_crs = ((gj.get('crs') or {}).get('properties') or {}).get('name', '')
    if nombre_crs and 'CRS84' not in nombre_crs and '4326' not in nombre_crs:
        crs_marcas = nombre_crs.replace('urn:ogc:def:crs:', '').replace('::', ':')
    polis, raras = [], {}
    for f in gj.get('features', []):
        props = {k.lower(): v for k, v in (f.get('properties') or {}).items()}
        c = normalizar(props.get('clase'))
        if c not in VALOR:
            raras[c] = raras.get(c, 0) + 1
            continue
        if f.get('geometry'):
            polis.append((f['geometry'], VALOR[c]))
    if raras:
        print(f'aviso: polígonos con «clase» que no es deslizamiento ni dudoso, no se usan: {raras}')
    ruta_rev = os.path.join(a.proyecto, 'revisadas.csv')
    if not os.path.exists(ruta_rev):
        raise SystemExit('falta revisadas.csv: corre antes preparar_para_qgis.py')
    revisadas = {r['tesela'] for r in csv.DictReader(open(ruta_rev, encoding='utf-8-sig'))
                 if str(r.get('revisada', '')).strip().lower() in ('si', 'sí', 'x', 'ok', '1')}
    if not revisadas:
        raise SystemExit('ninguna tesela tiene «si» en la columna revisada de revisadas.csv')
    os.makedirs(os.path.join(a.proyecto, 'masks'), exist_ok=True)
    n_tes = n_desl = 0
    for ruta in sorted(glob.glob(os.path.join(a.proyecto, 'images', '*', 'image.tif'))):
        idx = os.path.basename(os.path.dirname(ruta))
        if idx not in revisadas:
            continue
        with rasterio.open(ruta) as ds:
            h, w, tr, crs, b = ds.height, ds.width, ds.transform, ds.crs, ds.bounds
        marco = box(b.left, b.bottom, b.right, b.top)
        formas = []
        for g, v in polis:
            gu = transform_geom(crs_marcas, crs, g)
            if shape(gu).intersects(marco):
                formas.append((gu, v))
        formas.sort(key=lambda x: -x[1])         # primero dudoso, encima deslizamiento
        m = rasterize(formas, out_shape=(h, w), transform=tr, fill=0, dtype='uint8') if formas \
            else np.zeros((h, w), np.uint8)
        Image.fromarray(m).save(os.path.join(a.proyecto, 'masks', f'{idx}.png'))
        n_tes += 1
        n_desl += sum(1 for _, v in formas if v == 1)
        print(f'{idx}: {sum(1 for _, v in formas if v == 1)} deslizamiento(s), '
              f'{sum(1 for _, v in formas if v == 2)} dudoso(s), {int((m == 1).sum())} px marcados')
    print(f'\nmáscaras escritas: {n_tes} teselas revisadas, {n_desl} polígonos de deslizamiento')
    print(f'siguiente: python herramientas/importar_de_iris.py {a.proyecto} --umbral 0.45')


if __name__ == '__main__':
    main()
