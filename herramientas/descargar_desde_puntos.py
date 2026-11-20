"""
Arma las imágenes de entrenamiento de La Convención a partir de puntos de SINPAD-INDECI.

Para cada punto (coordenada y fecha del evento) arma un cuadrado alrededor y llama a
exportar_para_iris.py, que baja Sentinel-2 L2A y el relieve con el MISMO código del
servidor de Render. Todo queda en un solo proyecto de IRIS, listo para marcar.

Qué fechas usa: la temporada seca (1 de junio a 30 de septiembre) DESPUÉS del evento.
  - evento de enero a mayo  -> desde la temporada seca de ese mismo año
  - evento de junio en adelante -> desde la temporada seca del año siguiente
  - si eso es antes de 2017 (no hay imágenes) -> desde 2017, y el punto queda marcado
    como «antiguo»: la cicatriz puede verse ya cubierta de vegetación
  - entre ese primer año y los dos siguientes, elige el primero con al menos 5 fechas
    despejadas sobre el área (consulta el catálogo de Sentinel-2), para quedar lo más cerca
    del evento sin que la imagen salga tapada de nubes; si ninguno llega, el que tenga más
Con --anio se fuerza un año para todos (por ejemplo, para ver cómo están hoy).
Cada tesela anota qué parte quedó con imagen limpia (sin vacíos ni nubes blancas); bajo 80 % se avisa.

Lee el CSV que exporta QGIS de SINPAD (columnas num_posx, num_posy, fecha dd/mm/aaaa,
ide_sinpad, distrito) o uno simple con columnas lon, lat, fecha e id.

  python herramientas/descargar_desde_puntos.py Deslizamiento0327.csv --salida iris_laconvencion
  python herramientas/descargar_desde_puntos.py puntos.csv --salida iris_2024 --anio 2024 --desde-anio-evento 2016

Deja en la carpeta de salida el proyecto de IRIS (images/, deslizamientos.json) y
puntos_resumen.csv con lo que pasó en cada punto. Si se corta, se vuelve a correr y
salta los puntos ya hechos.

Ojo: la coordenada de SINPAD suele ser la del lugar afectado (un pueblo, una carretera),
no la de la cicatriz exacta. Por eso se baja un cuadrado alrededor y en IRIS se marca lo
que se ve en la imagen, no un círculo en el punto.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
ANIO_MIN = 2017            # el servidor acepta Sentinel-2 L2A desde abril de 2017


def leer_puntos(ruta):
    filas = list(csv.DictReader(open(ruta, encoding='utf-8-sig')))
    out = []
    for k, f in enumerate(filas, 1):
        g = {c.lower().strip(): (v or '').strip() for c, v in f.items() if c}
        lon = g.get('num_posx') or g.get('lon') or g.get('longitud') or g.get('x')
        lat = g.get('num_posy') or g.get('lat') or g.get('latitud') or g.get('y')
        if not lon or not lat:
            continue
        fecha = g.get('fecha', '')
        try:
            d = dt.datetime.strptime(fecha, '%d/%m/%Y').date() if '/' in fecha else dt.date.fromisoformat(fecha)
        except ValueError:
            d = None
        ident = g.get('ide_sinpad') or g.get('id') or g.get('fid') or str(k)
        out.append({'id': ident, 'lon': float(lon), 'lat': float(lat), 'fecha': d,
                    'distrito': g.get('distrito', ''), 'fenomeno': g.get('fenomeno', '')})
    return out


def fechas_despejadas(s, o, n, e, anio, nubes):
    import httpx
    hasta = min(dt.date(anio, 9, 30), dt.date.today() - dt.timedelta(days=1))
    if hasta < dt.date(anio, 6, 1):
        return 0
    cuerpo = {'collections': ['sentinel-2-l2a'], 'bbox': [o, s, e, n],
              'datetime': f'{anio}-06-01T00:00:00Z/{hasta.isoformat()}T23:59:59Z',
              'query': {'eo:cloud_cover': {'lt': nubes}}, 'limit': 500,
              'fields': {'include': ['properties.datetime'], 'exclude': ['assets', 'links', 'geometry', 'bbox']}}
    try:
        r = httpx.post('https://planetarycomputer.microsoft.com/api/stac/v1/search', json=cuerpo, timeout=120)
        return len({f['properties']['datetime'][:10] for f in r.json()['features']})
    except Exception:
        return -1


def ventana(fecha, anio_forzado, s, o, n, e, nubes):
    """(año elegido, evento antiguo, fechas despejadas por año candidato)"""
    if anio_forzado:
        return anio_forzado, False, {}
    if fecha is None:
        primero, antiguo = dt.date.today().year - 1, False
    else:
        a = fecha.year if fecha.month <= 5 else fecha.year + 1
        primero, antiguo = max(a, ANIO_MIN), a < ANIO_MIN
    ultimo = dt.date.today().year if dt.date.today().month >= 10 else dt.date.today().year - 1
    candidatos = [y for y in range(primero, primero + 3) if y <= ultimo] or [primero]
    cuentas = {y: fechas_despejadas(s, o, n, e, y, nubes) for y in candidatos}
    # el primer año con fechas suficientes (lo más cerca del evento, antes de que la ladera se
    # cubra de vegetación); si ninguno llega, el que tenga más
    suficientes = [y for y in candidatos if cuentas[y] >= 5]
    mejor = suficientes[0] if suficientes else max(candidatos, key=lambda y: (cuentas[y], -y))
    return mejor, antiguo, cuentas


def cobertura_tesela(ruta_tif):
    """(parte del área con imagen limpia, parte con nubes blancas que la máscara SCL no quitó).
    En B2 (azul) la vegetación y el suelo quedan bajo 1000; las nubes, muy por encima."""
    import rasterio
    with rasterio.open(ruta_tif) as ds:
        b2 = ds.read(2)
    con_dato = b2 > 0
    nube = con_dato & (b2 > 1000)
    return float((con_dato & ~nube).mean()), float(nube.mean())


def caja(lon, lat, lado_km):
    dlat = lado_km / 2 / 110.574
    dlon = lado_km / 2 / (111.320 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('csv')
    ap.add_argument('--salida', default='iris_laconvencion')
    ap.add_argument('--lado-km', dest='lado_km', type=float, default=2.56,
                    help='lado del cuadrado alrededor de cada punto (2.56 km = una tesela de 256 px)')
    ap.add_argument('--anio', type=int, help='usar la temporada seca de este año para todos los puntos')
    ap.add_argument('--desde-anio-evento', dest='desde', type=int, default=0,
                    help='solo eventos de este año en adelante (por ejemplo 2016)')
    ap.add_argument('--modelo', default='unet', help='modelo que dibuja la probabilidad y las cajas de guía')
    ap.add_argument('--nubes', type=float, default=20, help='nubosidad máxima por escena (criterio de la tesis: 20)')
    ap.add_argument('--umbral-cajas', dest='umbral_cajas', type=float, default=0.45,
                    help='umbral para las cajas de guía (0.45 = el de la U-Net; 0.30 pone muchas más)')
    ap.add_argument('--limite', type=int, help='solo los primeros N puntos (para probar)')
    a = ap.parse_args()

    puntos = [p for p in leer_puntos(a.csv) if not a.desde or (p['fecha'] and p['fecha'].year >= a.desde)]
    if a.limite:
        puntos = puntos[:a.limite]
    os.makedirs(a.salida, exist_ok=True)
    ruta_res = os.path.join(a.salida, 'puntos_resumen.csv')
    hechos = set()
    if os.path.exists(ruta_res):
        hechos = {r['id'] for r in csv.DictReader(open(ruta_res, encoding='utf-8')) if r['estado'] == 'ok'}
    print(f'{len(puntos)} puntos; ya hechos antes: {len(hechos)}')
    nuevo = not os.path.exists(ruta_res)
    res = open(ruta_res, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(res, fieldnames=['id', 'fecha', 'distrito', 'lon', 'lat', 'anio_imagen', 'antiguo',
                                         'fechas_despejadas', 'estado', 'teselas', 'cobertura_min', 'detalle'])
    if nuevo:
        w.writeheader()
    for i, p in enumerate(puntos, 1):
        if p['id'] in hechos:
            continue
        s, o, n, e = caja(p['lon'], p['lat'], a.lado_km)
        anio, antiguo, cuentas = ventana(p['fecha'], a.anio, s, o, n, e, a.nubes)
        hasta = min(dt.date(anio, 9, 30), dt.date.today() - dt.timedelta(days=1))
        prefijo = f"sinpad_{p['id']}"
        print(f"\n[{i}/{len(puntos)}] punto {p['id']} ({p['distrito']}, evento {p['fecha']}): "
              f"imágenes de jun a sep de {anio}{' (evento antiguo)' if antiguo else ''}"
              + (f"; fechas despejadas por año: {cuentas}" if cuentas else ''), flush=True)
        cmd = [sys.executable, os.path.join(AQUI, 'exportar_para_iris.py'),
               '--sur', f'{s:.5f}', '--oeste', f'{o:.5f}', '--norte', f'{n:.5f}', '--este', f'{e:.5f}',
               '--desde', f'{anio}-06-01', '--hasta', hasta.isoformat(), '--modelo', a.modelo,
               '--nubes', str(a.nubes), '--umbral_cajas', str(a.umbral_cajas),
               '--salida', a.salida, '--prefijo', prefijo]
        r = subprocess.run(cmd, capture_output=True, text=True)
        teselas = sorted(glob.glob(os.path.join(a.salida, 'images', f'{prefijo}_*', 'metadata.json')))
        coberturas = []
        if r.returncode == 0 and teselas:
            for m in teselas:                       # deja anotado de qué evento viene cada tesela
                meta = json.load(open(m))
                cob, nube = cobertura_tesela(m.replace('metadata.json', 'image.tif'))
                coberturas.append(cob)
                meta['nubes_blancas'] = round(nube, 3)
                meta['sinpad'] = {'id': p['id'], 'fecha_evento': str(p['fecha']), 'distrito': p['distrito'],
                                  'lon': p['lon'], 'lat': p['lat'], 'antiguo': antiguo}
                meta['cobertura_imagen'] = round(cob, 3)
                json.dump(meta, open(m, 'w'), indent=1, ensure_ascii=False)
            estado = 'ok'
            detalle = '' if min(coberturas) >= 0.8 else 'tesela con nubes: revisa la imagen o prueba con --anio de otro año'
            print(f'   listo: {len(teselas)} tesela(s), imagen limpia en el {min(coberturas)*100:.0f} % del área'
                  + (' (con nubes: revísala)' if detalle else ''))
        else:
            estado = 'error'
            lineas = [l for l in (r.stderr + r.stdout).strip().splitlines() if l.strip()]
            detalle = lineas[-1][:300] if lineas else 'sin detalle'
            print(f'   no se pudo: {detalle}')
        w.writerow({'id': p['id'], 'fecha': p['fecha'], 'distrito': p['distrito'], 'lon': p['lon'],
                    'lat': p['lat'], 'anio_imagen': anio, 'antiguo': antiguo,
                    'fechas_despejadas': cuentas.get(anio, ''), 'estado': estado, 'teselas': len(teselas),
                    'cobertura_min': round(min(coberturas), 3) if coberturas else '', 'detalle': detalle})
        res.flush()
    res.close()
    ok = [r for r in csv.DictReader(open(ruta_res, encoding='utf-8')) if r['estado'] == 'ok']
    print(f"\nTerminado: {len(ok)} puntos con imágenes, "
          f"{sum(int(r['teselas']) for r in ok)} teselas en {os.path.abspath(os.path.join(a.salida, 'images'))}")
    print(f'Resumen por punto: {os.path.abspath(ruta_res)}')
    print(f"Para marcar: iris label {os.path.abspath(os.path.join(a.salida, 'deslizamientos.json'))}")


if __name__ == '__main__':
    main()
