"""
Lleva lo marcado en IRIS a QGIS y mide el modelo en La Convención.

Lee un proyecto creado con exportar_para_iris.py (images/<id>/image.tif y masks/<id>.png)
y escribe en <proyecto>/resultados_iris/:
  <id>_etiqueta.tif       la máscara marcada a mano, georreferenciada (1 = deslizamiento)
  etiquetas.geojson       todos los deslizamientos marcados, en grados, con su área
  metricas.json           cómo le fue al modelo contra lo marcado a mano

Las métricas se calculan solo en las teselas que tienen máscara guardada, usando la
banda 7 (la probabilidad del modelo) con el umbral que se indique:
  - por píxel: IoU, F1, precisión y exhaustividad (recall)
  - por deslizamiento: qué parte de los marcados a mano tocó el modelo, por tamaño
  - por zona del modelo: qué parte de sus zonas toca un deslizamiento real

  python herramientas/importar_de_iris.py iris_laconvencion --umbral 0.45
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('proyecto')
    ap.add_argument('--umbral', type=float, help='umbral del modelo; por defecto el que guardó la exportación')
    a = ap.parse_args()

    import rasterio
    from PIL import Image
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from scipy import ndimage
    from shapely.geometry import shape

    salida = os.path.join(a.proyecto, 'resultados_iris')
    os.makedirs(salida, exist_ok=True)
    feats, tp = [], [0, 0, 0]
    tramos = {'<0.1 ha': (1, 10), '0.1-0.5 ha': (10, 50), '0.5-2 ha': (50, 200), '>2 ha': (200, 10 ** 9)}
    toca = {k: [0, 0] for k in tramos}
    zonas = [0, 0]
    usadas, umbrales = [], set()
    dudosos = 0
    for ruta_m in sorted(glob.glob(os.path.join(a.proyecto, 'masks', '*.png'))):
        idx = os.path.splitext(os.path.basename(ruta_m))[0]
        ruta_i = os.path.join(a.proyecto, 'images', idx, 'image.tif')
        if not os.path.exists(ruta_i):
            continue
        meta_p = os.path.join(a.proyecto, 'images', idx, 'metadata.json')
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        u = a.umbral if a.umbral is not None else float(meta.get('umbral_modelo', 0.5))
        umbrales.add(u)
        cruda = np.asarray(Image.open(ruta_m))
        y = cruda == 1
        dudoso = cruda == 2          # clase «Dudoso»: no cuenta en las métricas
        with rasterio.open(ruta_i) as ds:
            prob = ds.read(7); perfil = ds.profile; tr = ds.transform; crs = ds.crs
        perfil.update(count=1, dtype='uint8', nodata=None)
        with rasterio.open(os.path.join(salida, f'{idx}_etiqueta.tif'), 'w', **perfil) as d:
            d.write(y.astype('uint8'), 1)
        for g, v in shapes(y.astype('uint8'), mask=y, transform=tr):
            g4326 = transform_geom(crs, 'EPSG:4326', g)
            area_ha = shape(g).area / 1e4
            feats.append({'type': 'Feature', 'properties': {'tesela': idx, 'area_ha': round(area_ha, 3)},
                          'geometry': g4326})
        m = (prob >= u) & ~dudoso
        tp[0] += int((m & y).sum()); tp[1] += int((m & ~y & ~dudoso).sum()); tp[2] += int((~m & y).sum())
        lab, n = ndimage.label(y)
        for j in range(1, n + 1):
            o = lab == j; t = int(o.sum())
            for k, (lo, hi) in tramos.items():
                if lo <= t < hi:
                    toca[k][0] += bool((m & o).any()); toca[k][1] += 1
        lp, npred = ndimage.label(m)
        for j in range(1, npred + 1):
            zonas[1] += 1; zonas[0] += bool((y & (lp == j)).any())
        dudosos += int(ndimage.label(dudoso)[1])
        usadas.append(idx)
    if not usadas:
        raise SystemExit('no hay máscaras guardadas en masks/: marca y guarda al menos una tesela en IRIS')
    json.dump({'type': 'FeatureCollection', 'features': feats},
              open(os.path.join(salida, 'etiquetas.geojson'), 'w'), ensure_ascii=False)
    a_, b_, c_ = tp
    pr = a_ / max(a_ + b_, 1); rc = a_ / max(a_ + c_, 1)
    met = {'teselas_evaluadas': usadas, 'umbral': sorted(umbrales),
           'por_pixel': {'iou': round(a_ / max(a_ + b_ + c_, 1), 4), 'f1': round(2 * pr * rc / max(pr + rc, 1e-9), 4),
                         'precision': round(pr, 4), 'recall': round(rc, 4), 'tp': a_, 'fp': b_, 'fn': c_},
           'deslizamientos_marcados_tocados_por_el_modelo': {k: {'tocados': h, 'total': t,
                                                                  'fraccion': round(h / t, 3) if t else None}
                                                              for k, (h, t) in toca.items()},
           'zonas_del_modelo_que_tocan_un_deslizamiento': {'si': zonas[0], 'total': zonas[1],
                                                            'fraccion': round(zonas[0] / zonas[1], 3) if zonas[1] else None},
           'zonas_dudosas_excluidas': dudosos,
           'nota': 'Medido contra lo marcado a mano en IRIS. Las zonas «Dudoso» no cuentan. Con pocas teselas el margen de error es grande.'}
    json.dump(met, open(os.path.join(salida, 'metricas.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'teselas con máscara: {len(usadas)} | deslizamientos marcados: {len(feats)}')
    print(f"por píxel: IoU {met['por_pixel']['iou']}  F1 {met['por_pixel']['f1']}  "
          f"precisión {met['por_pixel']['precision']}  recall {met['por_pixel']['recall']}")
    for k, v in met['deslizamientos_marcados_tocados_por_el_modelo'].items():
        if v['total']:
            print(f"  {k:10s}: el modelo toca {v['tocados']} de {v['total']}")
    z = met['zonas_del_modelo_que_tocan_un_deslizamiento']
    print(f"zonas del modelo que tocan un deslizamiento real: {z['si']} de {z['total']}")
    print('archivos en', os.path.abspath(salida))


if __name__ == '__main__':
    main()
