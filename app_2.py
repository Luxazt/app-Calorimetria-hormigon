# ==============================================================================
# PREDICTOR DE RESISTENCIA — HORMIGÓN PROYECTADO (aplicación Streamlit)
# ==============================================================================
# Modelos: Random Forest y XGBoost POR PUNTO (Celda 2), un modelo por edad.
# Archivos en la carpeta de la app (del zip de la Celda 6, carpeta 1_por_punto):
#   modelo_porpunto_rf.pkl   modelo_porpunto_xgb.pkl
#
# CÓMO SE CONSTRUYE LA CURVA (igual que en la memoria):
#   1) cada modelo predice en su edad de ensayo
#   2) por defecto sólo entran las edades con métrica en la validación
#   3) si una edad tiene dos ensayos (4 h y 1 d), se toma la media
#   4) opcionalmente, máximo acumulado para que la resistencia no baje
#   5) curva suavizada: los puntos se unen con PCHIP (escala log del tiempo)
#      curva cruda: los puntos se unen con segmentos rectos
# Los valores entre edades medidas son interpolaciones, no predicciones.
#
# DATOS QUE FALTAN: se rellenan con el imputador del .pkl (mediana de la base
# de datos), nunca con ceros.
# ==============================================================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import PchipInterpolator

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Predictor de Resistencia — Hormigón Proyectado",
                   layout="wide", initial_sidebar_state="expanded")

TXT = "#1E293B"   # texto principal (oscuro, legible sobre fondo claro)
st.markdown(f"""
<style>
    .stApp {{ background-color: #F8FAFC; }}

    /* Cuerpo principal: textos oscuros aunque el navegador esté en modo oscuro */
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
    [data-testid="stMain"] h4, [data-testid="stMain"] p, [data-testid="stMain"] label,
    [data-testid="stMain"] span, [data-testid="stMain"] li {{ color: {TXT} !important; }}
    [data-testid="stMain"] h1 {{ font-weight: 700; letter-spacing: -0.5px; }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ color: #64748B !important; }}
    [data-testid="stMain"] input,
    [data-testid="stMain"] div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important; color: {TXT} !important;
    }}

    /* Barra lateral (calorimetría) */
    section[data-testid="stSidebar"] {{ background-color: #1E293B; }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {{ color: #F1F5F9 !important; }}
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="input"] {{
        background-color: #FFFFFF !important; color: {TXT} !important;
    }}
    section[data-testid="stSidebar"] button {{ color: {TXT} !important; }}

    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricValue"] {{ color: {TXT} !important; font-weight: 700; }}

    .header-band {{
        background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
        height: 5px; border-radius: 3px; margin-bottom: 18px;
    }}
    .fila-modelo {{ display: flex; align-items: center; height: 100%;
                   font-weight: 700; color: {TXT}; font-size: 1.05rem; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════
CARPETA = Path(__file__).parent
ALGORITMOS = {   # nombre en el .pkl -> (archivo, color, trazo, marcador)
    'Random Forest': ('modelo_porpunto_rf.pkl',  '#2563EB', 'solid', 'circle'),
    'XGBoost':       ('modelo_porpunto_xgb.pkl', '#DC2626', 'dash',  'square'),
}


@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_paquete(ruta: str):
    return joblib.load(ruta)


PAQUETES = {}
for alg, (archivo, *_) in ALGORITMOS.items():
    ruta = CARPETA / archivo
    if not ruta.exists():
        st.error(f"Falta `{archivo}` en la carpeta de la aplicación.")
        continue
    try:
        p = cargar_paquete(str(ruta))
    except Exception as err:
        st.error(f"No se pudo cargar `{archivo}`: {err}")
        continue
    if p.get('estrategia') != 'por_punto' or p.get('tipo') != alg:
        st.error(f"`{archivo}` contiene «{p.get('tipo')} / {p.get('estrategia')}», "
                 f"no «{alg} / por_punto». Se ignora.")
        continue
    PAQUETES[alg] = p

if not PAQUETES:
    st.error("No hay ningún modelo válido. Copia en la carpeta de la aplicación los "
             "dos .pkl de `1_por_punto/` del zip exportado por el cuaderno.")
    st.stop()

P_REF = next(iter(PAQUETES.values()))
FEATURES = list(P_REF['features_ok'])
FACT = P_REF.get('factorizacion', {})
COMBOS = P_REF.get('combos_validos', {})
IMPUTADOR = P_REF['imputer']
_MEDIANAS = P_REF.get('medianas')
DEFECTO = dict(zip(FEATURES, np.asarray(
    _MEDIANAS if _MEDIANAS is not None else IMPUTADOR.statistics_, float)))

VARS_CALOR = [v for v in ['punto_max_ace_mWg', 'pendiente_ace_mWgh', 'energia_ace_Jg',
                          'punto_max_ppal_mWg', 'pendiente_ppal_mWgh', 'energia_ppal_Jg',
                          'energia_total',
                          'punto_max_ace_C', 'punto_max_ppal_C', 'energia_temp_Jg']
              if v in FEATURES]
ETIQUETAS = {   # columna -> (texto, unidad, paso)
    'relacion_ac':         ('Relación agua/cemento', '—', 0.01),
    'dosificacion':        ('Dosificación de acelerante', '%', 0.1),
    'punto_max_ace_mWg':   ('Pico de flujo — fase del acelerante', 'mW/g', 0.1),
    'pendiente_ace_mWgh':  ('Pendiente de subida — fase del acelerante', 'mW/g·h', 0.01),
    'energia_ace_Jg':      ('Energía — fase del acelerante', 'J/g', 0.1),
    'punto_max_ppal_mWg':  ('Pico de flujo — fase principal', 'mW/g', 0.1),
    'pendiente_ppal_mWgh': ('Pendiente de subida — fase principal', 'mW/g·h', 0.01),
    'energia_ppal_Jg':     ('Energía — fase principal', 'J/g', 0.1),
    'energia_total':       ('Energía total de hidratación', 'J/g', 0.1),
    'punto_max_ace_C':     ('Temperatura máxima — fase del acelerante', '°C', 0.1),
    'punto_max_ppal_C':    ('Temperatura máxima — fase principal', '°C', 0.1),
    'energia_temp_Jg':     ('Área de la curva de temperatura', 'J/g', 1.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. PREDICCIÓN Y CURVA
# ═══════════════════════════════════════════════════════════════════════════
def nodos_de(p):
    """Una entrada por modelo de punto: columna, horas, serie, validado y MAE."""
    nodos = []
    for col in p['orden']:
        r = p['resultados'][col]
        nodos.append({'col': col, 'horas': float(r['horas']), 'serie': r['serie'],
                      'validado': bool(r.get('validado', True)),
                      'mae': (r.get('metricas') or {}).get('mae')})
    return sorted(nodos, key=lambda d: d['horas'])


def predecir_nodos(p, x, nodos):
    x = np.asarray(x, float).reshape(1, -1)
    pred = [p['resultados'][n['col']]['modelo'].predict(x)[0] for n in nodos]
    return np.maximum(np.asarray(pred, float), 0.0)


def construir_curva(nodos, pred, solo_validados, creciente, suavizada):
    """
    Devuelve (horas, valores, f): los puntos por los que pasa la curva y la
    función f(horas). Suavizada = PCHIP; cruda = segmentos rectos. Ambas en
    escala logarítmica del tiempo, que es la del gráfico.
    """
    usados = {}
    for n, v in zip(nodos, pred):
        if np.isfinite(v) and (n['validado'] or not solo_validados):
            usados.setdefault(round(n['horas'], 6), []).append(float(v))
    if len(usados) < 2:
        return None
    horas = np.array(sorted(usados))
    valores = np.array([np.mean(usados[h]) for h in horas])   # 4 h y 1 d: media
    if creciente:
        valores = np.maximum.accumulate(valores)
    log_h = np.log(horas)
    interp = PchipInterpolator(log_h, valores) if suavizada else None

    def f(h):
        h = np.asarray(h, float)
        dentro = (h >= horas[0] - 1e-9) & (h <= horas[-1] + 1e-9)
        salida = np.full(h.shape, np.nan)
        lh = np.log(h[dentro])
        salida[dentro] = interp(lh) if suavizada else np.interp(lh, log_h, valores)
        return np.maximum(salida, 0.0)

    return horas, valores, f


def etiqueta_edad(h):
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 24:
        return f"{h:g} h"
    return f"{h / 24:g} d"


def entrada_numerica(col, contenedor):
    texto, unidad, paso = ETIQUETAS.get(col, (col, '', 0.1))
    etiqueta = f"{texto} ({unidad})" if unidad not in ('', '—') else texto
    return float(contenedor.number_input(etiqueta, value=float(round(DEFECTO[col], 4)),
                                         step=paso, key=f"num_{col}"))


def selector_categorico(col, etiqueta, contenedor, codigos=None):
    textos = FACT.get(col, [])
    if not textos:
        return DEFECTO.get(col, 0.0)
    codigos = list(range(len(textos))) if codigos is None else [int(c) for c in codigos]
    por_defecto = int(round(DEFECTO.get(col, codigos[0])))
    idx = codigos.index(por_defecto) if por_defecto in codigos else 0
    elegido = contenedor.selectbox(etiqueta, codigos, index=idx,
                                   format_func=lambda k: str(textos[k]), key=f"cat_{col}")
    return float(elegido)


valores = {}

# ═══════════════════════════════════════════════════════════════════════════
# 3. BARRA LATERAL — SÓLO CALORIMETRÍA
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.header("Calorimetría")
hay_calor = st.sidebar.checkbox("Tengo ensayo de calorimetría", value=True)
for v in VARS_CALOR:
    valores[v] = entrada_numerica(v, st.sidebar) if hay_calor else np.nan

# ═══════════════════════════════════════════════════════════════════════════
# 4. CUERPO — PARÁMETROS DE LA MEZCLA Y OPCIONES
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="header-band"></div>', unsafe_allow_html=True)
st.title("Predictor de Resistencia — Hormigón Proyectado")
st.markdown("Introduzca el diseño de la mezcla y los resultados de calorimetría "
            "para obtener la curva de resistencia estimada.")

st.header("Parámetros de la mezcla")
c1, c2, c3 = st.columns(3)
if 'material' in FEATURES:
    valores['material'] = selector_categorico('material', "Material", c1)
if 'cemento' in FEATURES:
    valores['cemento'] = selector_categorico('cemento', "Tipo de cemento", c1)
if 'relacion_ac' in FEATURES:
    valores['relacion_ac'] = entrada_numerica('relacion_ac', c2)
familia_txt = None
if 'tipo_acelerante' in FEATURES:
    valores['tipo_acelerante'] = selector_categorico('tipo_acelerante', "Familia de acelerante", c2)
    familia_txt = str(FACT['tipo_acelerante'][int(valores['tipo_acelerante'])])
if 'dosificacion' in FEATURES:
    valores['dosificacion'] = entrada_numerica('dosificacion', c3)
if 'acelerante' in FEATURES:
    codigos_ok = COMBOS.get(familia_txt) if familia_txt is not None else None
    valores['acelerante'] = selector_categorico(
        'acelerante', "Producto acelerante", c3, codigos=codigos_ok or None)

st.subheader("Opciones de la curva")
o1, o2, o3 = st.columns(3)
algs_sel = [alg for alg in ALGORITMOS if alg in PAQUETES
            and o1.checkbox(alg, value=True, key=f"alg_{alg}")]
tipo_curva = o2.radio("Tipo de curva", ["Suavizada (PCHIP)", "Cruda (sin suavizar)"],
                      help="Cruda: las predicciones unidas con segmentos rectos.")
suavizada = tipo_curva.startswith("Suavizada")
solo_validados = o3.checkbox(
    "Sólo edades validadas", value=True,
    help="Excluye las edades sin métrica en la validación (p. ej. 4 h en penetración).")
creciente = o3.checkbox(
    "Forzar curva creciente", value=True,
    help="La resistencia del hormigón no disminuye con la edad (máximo acumulado).")

# Imputación con el imputador del .pkl (nunca ceros)
for f in FEATURES:
    valores.setdefault(f, np.nan)
x_crudo = pd.DataFrame([[valores[f] for f in FEATURES]], columns=FEATURES)
x_mezcla = IMPUTADOR.transform(x_crudo)[0]
imputadas = [ETIQUETAS.get(f, (f,))[0] for f in FEATURES if pd.isna(valores[f])]
if imputadas:
    st.warning(f"Sin dato en {len(imputadas)} variable(s), rellenadas con la mediana de "
               f"la base de datos: {', '.join(imputadas)}. La predicción depende entonces "
               f"sólo del resto de variables y es menos fiable.")

# ═══════════════════════════════════════════════════════════════════════════
# 5. GRÁFICO
# ═══════════════════════════════════════════════════════════════════════════
st.header("Curva de resistencia estimada")
fig = go.Figure()
NODOS_REF = nodos_de(P_REF)

# Bandas de fondo con el tramo de cada ensayo
for serie, color in [('Penetración', 'rgba(37,99,235,0.06)'),
                     ('Clavo', 'rgba(234,179,8,0.08)'),
                     ('Compresión', 'rgba(22,163,74,0.06)')]:
    hs = [n['horas'] for n in NODOS_REF if n['serie'] == serie]
    if hs:
        fig.add_vrect(x0=min(hs), x1=max(hs), fillcolor=color, line_width=0, layer='below')
        # En un eje logarítmico, Plotly espera la x de las anotaciones en log10.
        # (La etiqueta automática de add_vrect no lo hace y estira el eje.)
        fig.add_annotation(x=(np.log10(min(hs)) + np.log10(max(hs))) / 2, y=1.0,
                           xref='x', yref='paper', yanchor='bottom', showarrow=False,
                           text=serie, font=dict(color="#64748B", size=13))

CURVAS = {}
for alg in algs_sel:
    p = PAQUETES[alg]
    _, color, trazo, marcador = ALGORITMOS[alg]
    nodos = nodos_de(p)
    pred = predecir_nodos(p, x_mezcla, nodos)
    curva = construir_curva(nodos, pred, solo_validados, creciente, suavizada)
    if curva is None:
        st.error(f"{alg}: no hay suficientes edades para dibujar la curva.")
        continue
    horas, vals, f = curva
    CURVAS[alg] = {'nodos': nodos, 'pred': pred, 'horas': horas, 'vals': vals, 'f': f}
    h_linea = np.geomspace(horas[0], horas[-1], 400) if suavizada else horas
    fig.add_trace(go.Scatter(
        x=h_linea, y=f(h_linea), mode='lines', name=alg,
        line=dict(color=color, width=3, dash=trazo),
        hovertemplate=f'{alg}<br>%{{x:.2f}} h · %{{y:.1f}} MPa<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=horas, y=vals, mode='markers', showlegend=False,
        marker=dict(color=color, size=8, symbol=marcador, line=dict(color='white', width=1)),
        text=[etiqueta_edad(h) for h in horas],
        hovertemplate=f'{alg}<br>%{{text}} · %{{y:.2f}} MPa<extra></extra>'))

H_MIN = min(n['horas'] for n in NODOS_REF)
H_MAX = max(n['horas'] for n in NODOS_REF)
TICKS = [h for h in [0.05, 1 / 6, 0.5, 1, 2, 4, 12, 24, 72, 168, 672]
         if H_MIN / 1.3 <= h <= H_MAX * 1.3]
fig.update_layout(
    xaxis=dict(type='log', title=dict(text='Edad (escala logarítmica)', standoff=15,
                                      font=dict(color=TXT, size=15)),
               range=[np.log10(H_MIN / 1.3), np.log10(H_MAX * 1.3)],   # log10 en eje log
               tickvals=TICKS, ticktext=[etiqueta_edad(h) for h in TICKS], tickangle=0,
               gridcolor='#E2E8F0', showline=True, linecolor='#CBD5E1',
               tickfont=dict(size=13, color=TXT)),
    yaxis=dict(title=dict(text='Resistencia estimada (MPa)', standoff=15,
                          font=dict(color=TXT, size=15)),
               gridcolor='#E2E8F0', showline=True, linecolor='#CBD5E1',
               rangemode='tozero', tickfont=dict(size=13, color=TXT)),
    hovermode='closest', plot_bgcolor='white', paper_bgcolor='white',
    font=dict(color=TXT, size=14),
    legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5,
                font=dict(size=14, color=TXT)),
    margin=dict(l=80, r=40, t=60, b=120), height=650)
# theme=None: usa exactamente estos colores (el tema de Streamlit los cambia
# a blanco cuando el navegador está en modo oscuro)
st.plotly_chart(fig, width='stretch', theme=None)
st.caption(("Curva suavizada con PCHIP. " if suavizada else "Curva cruda: segmentos rectos. ")
           + "Los marcadores son predicciones del modelo en las edades medidas; la línea "
             "entre ellos es una interpolación. A 4 h y 1 d, si hay dos ensayos "
             "validados, se muestra su media.")

# ═══════════════════════════════════════════════════════════════════════════
# 6. EDADES CLAVE
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Predicción a edades clave")
HITOS = [(24, "1 día"), (168, "7 días"), (672, "28 días")]


def mae_en(nodos, horas):
    maes = [n['mae'] for n in nodos
            if abs(n['horas'] - horas) < 1e-6 and n['validado'] and n['mae'] is not None]
    return float(np.mean(maes)) if maes else None


cab = st.columns([1, 2, 2, 2])
for (h, nombre), c in zip(HITOS, cab[1:]):
    c.markdown(f"**{nombre}**")
for alg, d in CURVAS.items():
    fila = st.columns([1, 2, 2, 2])
    fila[0].markdown(f'<div class="fila-modelo">{alg}</div>', unsafe_allow_html=True)
    for (h, nombre), c in zip(HITOS, fila[1:]):
        v = d['f'](np.array([h]))[0]
        c.metric(nombre, f"{v:.1f} MPa" if np.isfinite(v) else "N/A",
                 label_visibility="collapsed")
        mae = mae_en(d['nodos'], h)
        if mae is not None:
            c.caption(f"error medio en validación: ± {mae:.1f} MPa")

# ═══════════════════════════════════════════════════════════════════════════
# 7. DETALLE Y FIABILIDAD
# ═══════════════════════════════════════════════════════════════════════════
with st.expander("Predicciones en cada edad medida"):
    for alg, d in CURVAS.items():
        tabla = pd.DataFrame({
            'Edad': [etiqueta_edad(n['horas']) for n in d['nodos']],
            'Ensayo': [n['serie'] for n in d['nodos']],
            'Predicción (MPa)': np.round(d['pred'], 2),
            'Validada': ['sí' if n['validado'] else 'no' for n in d['nodos']],
            'MAE validación (MPa)': [None if n['mae'] is None else round(n['mae'], 2)
                                     for n in d['nodos']],
        })
        st.markdown(f"**{alg}**")
        st.dataframe(tabla, hide_index=True, width='stretch')
        st.download_button(f"Descargar predicciones de {alg} (CSV)",
                           tabla.to_csv(index=False).encode('utf-8'),
                           file_name=f"predicciones_porpunto_{alg.split()[0].lower()}.csv",
                           mime='text/csv', key=f"csv_{alg}")

with st.expander("Sobre los modelos y su fiabilidad"):
    st.markdown("**Estrategia:** un modelo independiente para cada edad de ensayo (Celda 2).")
    for alg in algs_sel:
        g = (PAQUETES[alg].get('metricas') or {}).get('Global') or {}
        if g:
            st.markdown(f"- **{alg}**: MAE = {g['mae']:.2f} ± {g.get('mae_sd', 0):.2f} MPa · "
                        f"R² agrupado = {g['r2']:.3f} · {PAQUETES[alg].get('protocolo', '')}")
    st.markdown(
        "Las métricas proceden de una validación agrupada por mezcla: cada modelo se "
        "evaluó sobre mezclas que no había visto. Las predicciones sólo son fiables "
        "para mezclas parecidas a las de la base de datos (mismas familias de "
        "acelerante y rangos de dosificación y calorimetría similares).")
