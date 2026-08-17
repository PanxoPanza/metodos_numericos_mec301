"""Utilidad para embeber las animaciones HTML de la unidad 3.

Cada animación es un archivo HTML autocontenido en esta misma carpeta. La función
`animacion` lo lee y lo inserta en un <iframe srcdoc="...">, de modo que:

  * en RISE funciona con el kernel vivo,
  * en el jupyter book publicado el contenido queda embebido en el HTML de salida
    (no depende de rutas relativas ni de que Sphinx copie los archivos).

Uso desde el notebook:
    from interactive.embed import animacion
    animacion('A1_cazador_de_rectas.html')
"""

import html as _html
import os as _os
import warnings as _warnings

from IPython.display import HTML

_AQUI = _os.path.dirname(_os.path.abspath(__file__))


def animacion(archivo, alto=560):
    """Devuelve la animación `archivo` lista para desplegar en el notebook."""
    with open(_os.path.join(_AQUI, archivo), encoding='utf-8') as f:
        codigo = f.read()
    marco = ('<iframe srcdoc="{}" width="100%" height="{}" '
             'style="border:1px solid #ddd; border-radius:6px"></iframe>'
             .format(_html.escape(codigo, quote=True), alto))
    with _warnings.catch_warnings():          # el <iframe> es intencional
        _warnings.simplefilter('ignore', UserWarning)
        return HTML(marco)
