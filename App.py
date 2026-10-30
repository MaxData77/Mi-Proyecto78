"""
Procesamiento De Datos
======================
Sube un Excel (.xlsx / .xls) o un CSV y la app genera, automáticamente:

  * una vista previa de los datos con el tipo detectado de cada columna,
  * gráficos Plotly (barras, líneas, área, dispersión, circular, histograma, caja, correlación),
  * insights (máximos, mínimos, promedios, correlaciones, tendencias, valores atípicos),
  * un "Modo Aprendizaje" que explica por qué se eligió cada gráfico y cómo leerlo.

Todo vive en este único archivo. Orden del código:
  1. Configuración y estilo          5. Gráficos
  2. Utilidades                      6. Insights
  3. Carga y limpieza de datos       7. Interfaz (render_*)
  4. Detección de tipos de columna   8. Programa principal (main)

Ejecutar en local:  streamlit run app.py
"""

from __future__ import annotations

import csv
import inspect
import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# =============================================================================
# 1. CONFIGURACIÓN Y ESTILO
# =============================================================================

st.set_page_config(
    page_title="Procesamiento De Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# El modo oscuro se activa desde .streamlit/config.toml (ver instrucciones al final del README).
# Los gráficos usan siempre un fondo oscuro sólido para que también se vean bien al descargarlos.
BG_COLOR = "#0E1117"
PALETTE = ["#7C85F5", "#34D399", "#FBBF24", "#F87171", "#38BDF8", "#C084FC", "#2DD4BF", "#FB923C"]
ACCENT = PALETTE[0]

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1250px; }
      h1 { letter-spacing: -0.02em; }
      [data-testid="stMetric"] {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          padding: 14px 16px; border-radius: 10px;
      }
      .soft { color: #8B8E99; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Agregaciones disponibles: etiqueta visible -> función de pandas
AGG_OPTIONS = {"Suma": "sum", "Promedio": "mean", "Mediana": "median", "Conteo": "count"}
AGG_NAME = {"sum": "Suma", "mean": "Promedio", "median": "Mediana", "count": "Conteo"}
AGG_NOUN = {"sum": "suma", "mean": "promedio", "median": "mediana", "count": "cantidad de registros"}

# Nombres de columnas que casi siempre son identificadores (no sirven como medida)
ID_LIKE = re.compile(r"^(id|#|n[ºo°]\.?|nro\.?|n[uú]mero|index|[ií]ndice|folio|c[oó]digo)$", re.IGNORECASE)

MAX_SCATTER_POINTS = 20_000


# =============================================================================
# 2. UTILIDADES
# =============================================================================

def fit(func) -> dict:
    """Argumento de ancho completo según la versión de Streamlit.

    Las versiones nuevas reemplazan `use_container_width=True` por `width="stretch"`.
    Se detecta cuál acepta la función para que la app funcione en ambas.
    """
    params = inspect.signature(func).parameters
    return {"use_container_width": True} if "use_container_width" in params else {"width": "stretch"}


def fmt(x, decimals: int = 2) -> str:
    """Formatea números al estilo latino: 1.234,56"""
    if x is None or pd.isna(x):
        return "—"
    if float(x).is_integer():
        decimals = 0
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_r(r: float) -> str:
    """Coeficiente de correlación con signo y coma decimal: +0,85"""
    return f"{r:+.2f}".replace(".", ",")


def slugify(text: str) -> str:
    """Convierte un título en un nombre de archivo seguro."""
    return re.sub(r"[^\w-]+", "_", text.lower(), flags=re.UNICODE).strip("_")[:60] or "grafico"


def as_label(s: pd.Series) -> pd.Series:
    """Texto legible para usar una columna como etiqueta de categoría (nulos -> '(vacío)')."""
    out = s.dt.strftime("%Y-%m-%d") if pd.api.types.is_datetime64_any_dtype(s) else s.astype(str)
    return out.where(s.notna(), "(vacío)")


def make_unique(names: list[str]) -> list[str]:
    """Evita nombres de columna repetidos: 'Ventas', 'Ventas (2)', ..."""
    seen: dict[str, int] = {}
    result = []
    for name in names:
        base = name or "Columna"
        seen[base] = seen.get(base, 0) + 1
        result.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return result


# =============================================================================
# 3. CARGA Y LIMPIEZA DE DATOS
# =============================================================================

@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes, filename: str) -> list[str]:
    """Hojas de un Excel (lista vacía si es CSV)."""
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    return []


def _read_csv(data: bytes) -> pd.DataFrame:
    """Lee un CSV probando codificaciones y detectando el separador (coma, punto y coma, tab...)."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        for sep in (None, ","):  # None = detectar automáticamente
            try:
                return pd.read_csv(io.BytesIO(data), sep=sep, engine="python", encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                break  # probar la siguiente codificación
            except (pd.errors.ParserError, csv.Error) as exc:
                last_error = exc
    raise ValueError("No pude interpretar el CSV. Revisa que tenga encabezados y un separador válido.") from last_error


def _to_number(text: pd.Series) -> pd.Series:
    """Texto -> número. Acepta '1.234,56', '12,5', '$1500', '12%'."""
    t = text.astype(str).str.strip().str.replace(r"[\s$€£%]", "", regex=True)
    direct = pd.to_numeric(t, errors="coerce")  # punto decimal: 1234.56
    latin = pd.to_numeric(t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    # Se usa la lectura que logre convertir más valores. Si empatan, la directa
    # (por eso "1.500" se lee como 1,5: un punto solo es ambiguo).
    return latin if latin.notna().sum() > direct.notna().sum() else direct


def smart_convert(s: pd.Series) -> pd.Series:
    """Si una columna de texto en realidad contiene números o fechas (>=90%), la convierte."""
    filled = s.dropna().astype(str).str.strip()
    filled = filled[filled != ""]
    if filled.empty:
        return s

    # 1) Números escritos como texto (se omiten códigos con ceros a la izquierda, ej. "00123")
    has_leading_zero = filled.str.match(r"^0\d+").any()
    if not has_leading_zero and _to_number(filled).notna().mean() >= 0.9:
        return _to_number(s)

    # 2) Fechas escritas como texto (solo si "parecen" fechas, para no convertir texto cualquiera)
    if filled.str.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() >= 0.9:
        iso = filled.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() >= 0.9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=not iso)
        if parsed.notna().sum() / max(len(filled), 1) >= 0.9:
            return parsed
    return s


def tidy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: quita filas/columnas vacías, normaliza nombres y convierte tipos."""
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    df.columns = make_unique([str(c).strip() for c in df.columns])
    df = df.reset_index(drop=True)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = smart_convert(df[col])
    return df


@st.cache_data(show_spinner="Leyendo el archivo…")
def load_dataframe(file_bytes: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    """Archivo subido -> DataFrame limpio. Se cachea para no releer en cada interacción."""
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet or 0)
    else:
        df = _read_csv(file_bytes)
    df = tidy_dataframe(df)
    if df.empty:
        raise ValueError("El archivo no tiene datos.")
    return df


def sample_dataframe() -> pd.DataFrame:
    """Datos de ejemplo (ventas) para probar la app sin subir nada."""
    rng = np.random.default_rng(42)
    n = 180
    products = {"Notebook": 650_000, "Monitor": 180_000, "Teclado": 35_000, "Mouse": 18_000}
    product = rng.choice(list(products), n, p=[0.2, 0.25, 0.3, 0.25])
    units = rng.poisson(6, n) + 1
    price = np.array([products[p] for p in product]) * rng.normal(1, 0.04, n)
    sales = (units * price).round(0)
    sales[17] *= 6  # un valor atípico a propósito
    df = pd.DataFrame(
        {
            "Fecha": pd.date_range("2025-01-01", periods=n, freq="D"),
            "Región": rng.choice(["Norte", "Centro", "Sur", "Oriente"], n),
            "Producto": product,
            "Unidades": units,
            "Ventas": sales,
            "Descuento (%)": rng.choice([0, 5, 10, 15], n, p=[0.5, 0.25, 0.15, 0.1]),
        }
    )
    df["Costo"] = (df["Ventas"] * rng.normal(0.62, 0.05, n)).round(0)
    df.loc[rng.choice(n, 6, replace=False), "Región"] = None  # algunos vacíos
    return df


# =============================================================================
# 4. DETECCIÓN DE TIPOS DE COLUMNA
# =============================================================================

@dataclass
class ColumnTypes:
    numeric: list[str] = field(default_factory=list)      # medidas: ventas, unidades...
    categorical: list[str] = field(default_factory=list)  # pocas categorías: región, producto...
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)         # texto libre o con demasiadas categorías
    ids: list[str] = field(default_factory=list)          # identificadores numéricos

    def label(self, col: str) -> str:
        for name, cols in (("Numérica", self.numeric), ("Categórica", self.categorical), ("Fecha", self.datetime),
                           ("Identificador", self.ids), ("Texto", self.text)):
            if col in cols:
                return name
        return "Texto"


def _is_running_number(s: pd.Series) -> bool:
    """True para columnas tipo 1, 2, 3, 4... (números correlativos)."""
    s = s.dropna()
    return len(s) >= 5 and s.is_unique and s.is_monotonic_increasing and bool((s.diff().dropna() == 1).all())


def detect_types(df: pd.DataFrame, max_categories: int) -> ColumnTypes:
    """Clasifica cada columna como numérica, categórica, fecha, texto o identificador."""
    types = ColumnTypes()
    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            types.categorical.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            types.datetime.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            is_id = bool(ID_LIKE.match(col.strip())) or _is_running_number(s)
            (types.ids if is_id else types.numeric).append(col)
        else:
            n_unique = s.nunique(dropna=True)
            looks_like_id = n_rows > 20 and n_unique > 0.5 * n_rows
            if 2 <= n_unique <= max_categories and not looks_like_id:
                types.categorical.append(col)
            else:
                types.text.append(col)
    return types


# =============================================================================
# 5. GRÁFICOS
# =============================================================================

@dataclass
class ChartSpec:
    """Un gráfico listo para mostrar, con su explicación del Modo Aprendizaje."""
    key: str
    title: str
    fig: go.Figure
    why: str
    interpret: str
    tips: list[str]


# Explicaciones base de cada tipo de gráfico (Modo Aprendizaje)
LESSONS = {
    "Barras": dict(
        why="Las barras comparan magnitudes entre categorías distintas. Es el gráfico más fácil de leer cuando la pregunta es «¿cuál es mayor?».",
        interpret="Compara el largo de las barras: la más larga es la categoría con mayor valor. Fíjate en las diferencias grandes; las pequeñas pueden ser ruido.",
        tips=["Ordena las categorías de mayor a menor para leerlas como un ranking.",
              "Si hay más de 10 o 12 categorías, agrupa las menores en «Otros».",
              "Mantén el eje en cero: si no, las diferencias parecen más grandes de lo que son."],
    ),
    "Líneas": dict(
        why="Las líneas muestran cómo cambia un valor a lo largo del tiempo, porque el ojo sigue la pendiente casi sin esfuerzo.",
        interpret="Mira la dirección general (sube, baja o se mantiene), los picos y caídas bruscas, y si el patrón se repite (estacionalidad).",
        tips=["Usa pocas series para que no se enreden.",
              "Prueba un promedio móvil para suavizar el ruido y ver la tendencia real.",
              "Compara contra el mismo período anterior para saber si el cambio es normal."],
    ),
    "Área": dict(
        why="El área es una línea con relleno: enfatiza el volumen total acumulado, no solo la forma de la curva.",
        interpret="Lo que importa es cuánto crece el acumulado y a qué ritmo: una pendiente más pronunciada significa un período más fuerte.",
        tips=["Un acumulado siempre sube; busca los cambios de pendiente, no la subida en sí.",
              "Si necesitas comparar valores de cada período, usa líneas.",
              "Agrega una meta como referencia para ver cuándo se alcanza."],
    ),
    "Dispersión": dict(
        why="La dispersión pone dos variables numéricas una contra otra para ver si existe relación entre ellas.",
        interpret="Si los puntos forman una diagonal hay relación (positiva si sube, negativa si baja). Una nube sin forma indica que casi no se relacionan. Los puntos aislados son candidatos a anomalías.",
        tips=["Revisa los puntos alejados: pueden ser errores de datos o casos especiales.",
              "Una relación visible no prueba causa y efecto; hay que confirmarla con más datos.",
              "Colorea los puntos por una categoría para descubrir subgrupos."],
    ),
    "Circular": dict(
        why="El gráfico circular muestra qué parte del total aporta cada categoría. Funciona bien con pocas categorías (hasta unas 6).",
        interpret="Busca las porciones dominantes y cuántas categorías se reparten el total. Es difícil comparar porciones de tamaño parecido.",
        tips=["Si hay muchas porciones pequeñas, agrúpalas en «Otros» (la app lo hace con las menores).",
              "Para comparar valores con precisión, las barras son más claras.",
              "Los porcentajes suman 100%: úsalo para hablar de participación, no de crecimiento."],
    ),
    "Histograma": dict(
        why="El histograma muestra cómo se distribuyen los valores de una variable: dónde se concentran y qué tan dispersos están.",
        interpret="La barra más alta marca el rango más frecuente. Si la forma es simétrica, los datos son parejos; si tiene una cola larga hacia un lado, hay valores extremos. La caja de arriba resume mediana y atípicos.",
        tips=["Compara promedio y mediana: si difieren mucho, hay valores extremos.",
              "Cambia el número de intervalos mentalmente: la forma general debería mantenerse.",
              "Una segunda 'montaña' sugiere dos grupos mezclados en los datos."],
    ),
    "Caja": dict(
        why="El diagrama de caja compara la distribución de una variable entre categorías, no solo su promedio.",
        interpret="La línea central es la mediana, la caja contiene el 50% central de los datos y los puntos sueltos son valores atípicos. Cajas más altas = valores mayores; cajas más largas = más variabilidad.",
        tips=["Dos categorías con igual mediana pueden tener variabilidad muy distinta.",
              "Los puntos fuera de los bigotes merecen revisión: ¿error o caso especial?",
              "Con pocas filas por categoría la caja es poco confiable."],
    ),
    "Mapa de correlación": dict(
        why="El mapa de correlación resume en una sola imagen qué columnas numéricas se mueven juntas y cuáles no.",
        interpret="Los valores van de -1 a 1. Cerca de 1: cuando una sube, la otra también. Cerca de -1: cuando una sube, la otra baja. Cerca de 0: no hay relación lineal.",
        tips=["Empieza por los cuadros más intensos (rojo o azul) y mira esa pareja en un gráfico de dispersión.",
              "Correlación no implica causalidad: dos variables pueden moverse juntas por una tercera.",
              "Ignora la diagonal: siempre es 1 (cada columna contra sí misma)."],
    ),
}


def style_figure(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    """Estilo común de todos los gráficos: tema oscuro, fondo sólido, título a la izquierda."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        colorway=PALETTE,
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        font=dict(family="sans-serif", size=13),
    )
    return fig


def default_title(kind: str, x: str | None, y: str | None, agg: str) -> str:
    """Título automático de un gráfico."""
    what = f"{AGG_NAME[agg]} de {y}" if y and agg != "count" else "Cantidad de registros"
    return {
        "Barras": f"{what} por {x}",
        "Líneas": f"{what} según {x}",
        "Área": f"Acumulado de {y} según {x}",
        "Dispersión": f"{y} vs {x}",
        "Circular": f"Participación en {y if y and agg != 'count' else 'registros'} por {x}",
        "Histograma": f"Distribución de {x}",
        "Caja": f"Distribución de {y} por {x}",
        "Mapa de correlación": "Correlación entre variables numéricas",
    }[kind]


def _need(*cols: str | None) -> None:
    """Valida que se hayan elegido las columnas necesarias."""
    if any(not c for c in cols):
        raise ValueError("Falta elegir columnas para este gráfico.")


def _group(df: pd.DataFrame, x: str, y: str | None, agg: str) -> tuple[pd.DataFrame, str, str]:
    """Agrupa por categoría (x) y agrega la medida (y). Devuelve (tabla, columna de valor, título del eje)."""
    _need(x)
    if y and x == y:
        raise ValueError("Elige columnas distintas para el eje X y el eje Y.")
    d = pd.DataFrame({x: as_label(df[x])})
    if agg == "count" or not y:
        table = d.groupby(x).size().reset_index(name="Registros")
        return table, "Registros", "Cantidad de registros"
    d[y] = pd.to_numeric(df[y], errors="coerce")
    table = d.groupby(x)[y].agg(agg).reset_index()
    return table, y, f"{AGG_NAME[agg]} de {y}"


def _series_over_x(df: pd.DataFrame, x: str, y: str, agg: str) -> pd.DataFrame:
    """Serie (x, y) agregada. Si x es fecha y el rango es largo, agrupa por mes."""
    _need(x, y)
    if x == y:
        raise ValueError("Elige columnas distintas para el eje X y el eje Y.")
    d = pd.DataFrame({x: df[x], y: pd.to_numeric(df[y], errors="coerce")}).dropna()
    if d.empty:
        raise ValueError("No hay datos válidos con esta selección.")
    if pd.api.types.is_datetime64_any_dtype(d[x]):
        if (d[x].max() - d[x].min()).days > 180:
            grouped = d.groupby(pd.Grouper(key=x, freq="MS"))[y].agg(agg)
        else:
            grouped = d.groupby(d[x].dt.normalize())[y].agg(agg)
        grouped = grouped.dropna() if agg != "sum" else grouped
    else:
        grouped = d.groupby(x)[y].agg(agg).sort_index()
    out = grouped.reset_index()
    out.columns = [x, y]
    return out


def make_chart(kind: str, df: pd.DataFrame, *, x: str | None = None, y: str | None = None,
               color: str | None = None, agg: str = "sum", top_n: int = 15,
               numeric_cols: list[str] | None = None, title: str | None = None) -> go.Figure:
    """Construye un gráfico Plotly. Lanza ValueError con un mensaje claro si la selección no sirve."""
    title = title or default_title(kind, x, y, agg)

    if kind == "Barras":
        table, val, axis_title = _group(df, x, y, agg)
        table = table.sort_values(val, ascending=False).head(top_n)
        horizontal = len(table) > 8  # muchas categorías: barras horizontales, etiquetas más legibles
        if horizontal:
            fig = px.bar(table, x=val, y=x, orientation="h", text_auto=".3s")
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(xaxis_title=axis_title, yaxis_title=x)
        else:
            fig = px.bar(table, x=x, y=val, text_auto=".3s")
            fig.update_layout(xaxis_title=x, yaxis_title=axis_title)
        fig.update_traces(marker_color=ACCENT, textposition="outside", cliponaxis=False)

    elif kind == "Líneas":
        series = _series_over_x(df, x, y, agg)
        fig = px.line(series, x=x, y=y, markers=len(series) <= 60)
        fig.update_traces(line_color=ACCENT, marker_color=ACCENT)
        fig.update_layout(xaxis_title=x, yaxis_title=f"{AGG_NAME[agg]} de {y}")

    elif kind == "Área":
        series = _series_over_x(df, x, y, "sum")
        series[y] = series[y].cumsum()
        fig = px.area(series, x=x, y=y)
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(124,133,245,0.30)")
        fig.update_layout(xaxis_title=x, yaxis_title=f"Acumulado de {y}")

    elif kind == "Dispersión":
        _need(x, y)
        if x == y:
            raise ValueError("Elige columnas distintas para el eje X y el eje Y.")
        d = pd.DataFrame({x: pd.to_numeric(df[x], errors="coerce"), y: pd.to_numeric(df[y], errors="coerce")})
        if color and color not in (x, y):
            d[color] = as_label(df[color])
        else:
            color = None
        d = d.dropna(subset=[x, y])
        if len(d) < 2:
            raise ValueError("No hay suficientes pares de valores numéricos.")
        if len(d) > MAX_SCATTER_POINTS:
            d = d.sample(MAX_SCATTER_POINTS, random_state=0)
            title += f" (muestra de {fmt(MAX_SCATTER_POINTS)} filas)"
        fig = px.scatter(d, x=x, y=y, color=color, opacity=0.8, color_discrete_sequence=PALETTE)
        xv, yv = d[x].to_numpy(dtype=float), d[y].to_numpy(dtype=float)
        if len(d) >= 3 and np.ptp(xv) > 0:  # línea de tendencia (ajuste lineal)
            slope, intercept = np.polyfit(xv, yv, 1)
            xs = np.array([xv.min(), xv.max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Tendencia lineal",
                                     line=dict(color="#FBBF24", dash="dash")))
        fig.update_layout(xaxis_title=x, yaxis_title=y)

    elif kind == "Circular":
        table, val, _ = _group(df, x, y, "count" if (agg == "count" or not y) else "sum")
        table = table.sort_values(val, ascending=False)
        if (table[val] < 0).any():
            raise ValueError("El gráfico circular necesita valores positivos.")
        if len(table) > 8:  # agrupa las categorías menores en «Otros»
            rest = table.iloc[7:][val].sum()
            table = pd.concat([table.head(7), pd.DataFrame({x: ["Otros"], val: [rest]})], ignore_index=True)
        fig = px.pie(table, names=x, values=val, hole=0.45, color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="percent", textposition="inside", sort=False)

    elif kind == "Histograma":
        _need(x)
        d = pd.DataFrame({x: pd.to_numeric(df[x], errors="coerce")}).dropna()
        if d.empty:
            raise ValueError("La columna no tiene valores numéricos.")
        fig = px.histogram(d, x=x, nbins=30, marginal="box")
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(xaxis_title=x, yaxis_title="Cantidad de registros")

    elif kind == "Caja":
        _need(x, y)
        if x == y:
            raise ValueError("Elige columnas distintas para el eje X y el eje Y.")
        d = pd.DataFrame({x: as_label(df[x]), y: pd.to_numeric(df[y], errors="coerce")}).dropna()
        keep = d[x].value_counts().head(12).index  # máximo 12 cajas para que se lea bien
        d = d[d[x].isin(keep)]
        if d.empty:
            raise ValueError("No hay datos válidos con esta selección.")
        fig = px.box(d, x=x, y=y, color=x, points="outliers", color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, xaxis_title=x, yaxis_title=y)

    elif kind == "Mapa de correlación":
        cols = numeric_cols or []
        if len(cols) < 2:
            raise ValueError("Se necesitan al menos 2 columnas numéricas.")
        corr = df[cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title="r"))

    else:
        raise ValueError(f"Tipo de gráfico desconocido: {kind}")

    return style_figure(fig, title)


def explain(kind: str, x: str | None = None, y: str | None = None, agg: str = "sum",
            reason: str | None = None, extra: str = "") -> dict:
    """Texto del Modo Aprendizaje: por qué este gráfico, qué interpretar y cómo mejorar."""
    base = LESSONS[kind]
    noun = AGG_NOUN[agg]
    measure = f'la {noun} de "{y}"' if y and agg != "count" else "la cantidad de registros"
    context = {
        "Barras": f'Aquí se compara {measure} entre las categorías de "{x}". ',
        "Líneas": f'Aquí se sigue {measure} a lo largo de "{x}". ',
        "Área": f'Aquí se ve cuánto suma "{y}" acumulado a lo largo de "{x}". ',
        "Dispersión": f'Aquí se cruzan "{x}" (horizontal) y "{y}" (vertical). ',
        "Circular": f'Aquí se reparte {measure if y and agg != "count" else "el total de registros"} entre las categorías de "{x}". ',
        "Histograma": f'Aquí se ve cómo se reparten los valores de "{x}". ',
        "Caja": f'Aquí se compara la distribución de "{y}" entre las categorías de "{x}". ',
        "Mapa de correlación": "",
    }[kind]
    return dict(why=reason or base["why"], interpret=context + base["interpret"] + (" " + extra if extra else ""),
                tips=base["tips"])


def auto_charts(df: pd.DataFrame, types: ColumnTypes, agg: str, top_n: int) -> tuple[list[ChartSpec], list[str]]:
    """Elige y construye automáticamente los gráficos más útiles según los tipos de columna.

    Devuelve (gráficos, avisos de gráficos que no se pudieron construir).
    """
    specs: list[ChartSpec] = []
    skipped: list[str] = []
    nums, cats, dates = types.numeric, types.categorical, types.datetime
    small_cats = [c for c in cats if df[c].nunique() <= 8]   # aptas para circular
    box_cats = [c for c in cats if df[c].nunique() <= 12]    # aptas para cajas

    def add(key: str, kind: str, reason: str, extra: str = "", **kw) -> None:
        try:
            title = default_title(kind, kw.get("x"), kw.get("y"), kw.get("agg", agg))
            fig = make_chart(kind, df, agg=agg, top_n=top_n, title=title, **kw)
            lesson = explain(kind, kw.get("x"), kw.get("y"), kw.get("agg", agg), reason, extra)
            specs.append(ChartSpec(key=key, title=title, fig=fig, **lesson))
        except Exception as exc:  # un gráfico que falla no debe romper toda la página
            skipped.append(f"{kind} ({key}): {exc}")

    if nums:
        add("hist", "Histograma", f'Para entender cómo se distribuyen los valores de "{nums[0]}" y detectar extremos.', x=nums[0])
    if cats:
        add("bar1", "Barras", f'Hay una categoría ("{cats[0]}")'
            + (f' y una medida ("{nums[0]}"): las barras permiten comparar categorías al instante.' if nums
               else ": las barras muestran cuántos registros tiene cada una."),
            x=cats[0], y=nums[0] if nums else None)
    if len(cats) > 1:
        add("bar2", "Barras", f'Segunda categoría disponible ("{cats[1]}"): conviene ver si el patrón se repite.',
            x=cats[1], y=nums[0] if nums else None)
    if dates and nums:
        add("line", "Líneas", f'Hay fechas ("{dates[0]}") y valores numéricos: una línea muestra la evolución y las tendencias.',
            x=dates[0], y=nums[0])
        add("area", "Área", f'Misma serie de tiempo, pero acumulada: muestra el volumen total que se va sumando.',
            x=dates[0], y=nums[0])
    if len(nums) >= 2:
        a, b, r = best_pair(df, nums)
        color = small_cats[0] if small_cats else None
        add("scatter", "Dispersión",
            f'"{a}" y "{b}" son las dos columnas numéricas más relacionadas (r = {fmt_r(r)}); la dispersión permite verlo.',
            extra=f"La correlación entre ambas es r = {fmt_r(r)}.", x=a, y=b, color=color)
    if small_cats:
        add("pie", "Circular", f'"{small_cats[0]}" tiene pocas categorías: un circular muestra qué parte del total aporta cada una.',
            x=small_cats[0], y=nums[0] if nums else None)
    if box_cats and nums:
        add("box", "Caja", f'Para comparar la variabilidad de "{nums[0]}" entre las categorías de "{box_cats[0]}", no solo su promedio.',
            x=box_cats[0], y=nums[0])
    if len(nums) >= 2:
        add("corr", "Mapa de correlación", "Resume de una vez qué columnas numéricas se mueven juntas.", numeric_cols=nums[:12])
    return specs, skipped


def best_pair(df: pd.DataFrame, nums: list[str]) -> tuple[str, str, float]:
    """Pareja de columnas numéricas con mayor correlación (en valor absoluto)."""
    best = (nums[0], nums[1], 0.0)
    best_abs = -1.0
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            r = df[a].corr(df[b])
            if pd.notna(r) and abs(r) > best_abs:
                best, best_abs = (a, b, float(r)), abs(r)
    return best


# =============================================================================
# 6. INSIGHTS
# =============================================================================

@dataclass
class Insights:
    highlights: list[str]          # resumen en lenguaje natural
    stats: pd.DataFrame            # estadísticos por columna numérica
    extremes: list[str]            # máximos y mínimos
    correlations: pd.DataFrame     # pares con relación relevante
    trends: list[str]              # tendencias en el tiempo
    outliers: pd.DataFrame         # valores atípicos por columna
    categories: list[str]          # categoría dominante
    missing: pd.DataFrame          # valores vacíos por columna


def row_label(df: pd.DataFrame, idx: int, label_cols: list[str]) -> str:
    """Descripción de una fila: 'Región: Norte, Producto: Monitor' o 'fila 18'."""
    parts = [f"{c}: {df.at[idx, c]}" for c in label_cols[:2] if pd.notna(df.at[idx, c])]
    return ", ".join(parts) if parts else f"fila {idx + 1}"


def trend_of(df: pd.DataFrame, date_col: str, num_col: str) -> dict | None:
    """Tendencia lineal de una medida en el tiempo."""
    d = df[[date_col, num_col]].dropna().sort_values(date_col)
    if len(d) < 6:
        return None
    x = (d[date_col] - d[date_col].min()).dt.total_seconds().to_numpy() / 86400  # días desde el inicio
    y = d[num_col].to_numpy(dtype=float)
    if np.ptp(x) == 0 or np.std(y) == 0:
        return None
    slope = np.polyfit(x, y, 1)[0]
    r = float(np.corrcoef(x, y)[0, 1])
    third = max(len(d) // 3, 1)
    first, last = y[:third].mean(), y[-third:].mean()
    change = (last - first) / abs(first) * 100 if first != 0 else None
    return {"slope": slope, "r": r, "first": first, "last": last, "change": change}


def compute_insights(df: pd.DataFrame, types: ColumnTypes) -> Insights:
    """Análisis automático: extremos, promedios, correlaciones, tendencias, atípicos, categorías y vacíos."""
    nums, cats, dates = types.numeric, types.categorical, types.datetime
    label_cols = cats + types.text
    highlights: list[str] = []

    # --- Panorama y valores vacíos ---
    total_cells = df.size
    n_missing = int(df.isna().sum().sum())
    missing_pct = n_missing / total_cells * 100 if total_cells else 0
    highlights.append(
        f"El conjunto tiene {fmt(len(df))} filas y {len(df.columns)} columnas "
        f"({len(nums)} numéricas, {len(cats)} categóricas, {len(dates)} de fecha). "
        + (f"El {fmt(missing_pct, 1)}% de las celdas está vacío." if n_missing else "No hay celdas vacías.")
    )
    miss = df.isna().sum()
    missing = pd.DataFrame({"Columna": miss.index, "Vacíos": miss.values, "% vacío": (miss.values / max(len(df), 1) * 100).round(1)})
    missing = missing[missing["Vacíos"] > 0].sort_values("Vacíos", ascending=False).reset_index(drop=True)

    # --- Estadísticos y extremos ---
    stats = pd.DataFrame()
    extremes: list[str] = []
    if nums:
        desc = df[nums].agg(["count", "mean", "median", "std", "min", "max"]).T
        desc.columns = ["Datos", "Promedio", "Mediana", "Desv. estándar", "Mínimo", "Máximo"]
        stats = desc.round(2).rename_axis("Columna").reset_index()
        for col in nums[:5]:
            s = df[col].dropna()
            if s.empty:
                continue
            i_max, i_min = s.idxmax(), s.idxmin()
            extremes.append(
                f'**{col}**: máximo **{fmt(s.max())}** ({row_label(df, i_max, label_cols)}); '
                f'mínimo **{fmt(s.min())}** ({row_label(df, i_min, label_cols)}); '
                f'promedio {fmt(s.mean())}, mediana {fmt(s.median())}.'
            )
        first = df[nums[0]].dropna()
        if not first.empty:
            highlights.append(
                f'En "{nums[0]}", el valor más alto es {fmt(first.max())} y el más bajo {fmt(first.min())}; '
                f"el promedio es {fmt(first.mean())} y la mediana {fmt(first.median())}."
            )

    # --- Correlaciones ---
    rows = []
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            r = df[a].corr(df[b])
            if pd.notna(r) and abs(r) >= 0.5:
                strength = "muy fuerte" if abs(r) >= 0.8 else "fuerte" if abs(r) >= 0.65 else "moderada"
                direction = "positiva (suben juntas)" if r > 0 else "negativa (una sube, la otra baja)"
                rows.append({"Columna A": a, "Columna B": b, "r": round(float(r), 2), "Lectura": f"{strength.capitalize()}, {direction}"})
    correlations = pd.DataFrame(rows, columns=["Columna A", "Columna B", "r", "Lectura"])
    if not correlations.empty:
        correlations = correlations.reindex(correlations["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        top = correlations.iloc[0]
        highlights.append(f'"{top["Columna A"]}" y "{top["Columna B"]}" tienen una correlación {top["Lectura"].split(",")[0].lower()} (r = {fmt_r(top["r"])}).')

    # --- Tendencias en el tiempo ---
    trends: list[str] = []
    if dates and nums:
        for col in nums[:3]:
            t = trend_of(df, dates[0], col)
            if not t:
                continue
            if abs(t["r"]) < 0.2:
                text = f'**{col}** no muestra una tendencia clara a lo largo de "{dates[0]}".'
            else:
                direction = "al alza" if t["slope"] > 0 else "a la baja"
                change = f' (de {fmt(t["first"])} a {fmt(t["last"])}, {fmt(abs(t["change"]), 1)}% {"más" if t["change"] > 0 else "menos"})' if t["change"] is not None else ""
                text = f'**{col}** va {direction} en el tiempo{change}.'
            trends.append(text)
        if trends:
            highlights.append(trends[0].replace("**", ""))

    # --- Valores atípicos (regla del rango intercuartílico) ---
    out_rows = []
    for col in nums:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bad = s[(s < lo) | (s > hi)]
        if len(bad):
            worst = (bad - s.median()).abs().idxmax()
            out_rows.append({"Columna": col, "Atípicos": len(bad), "% de filas": round(len(bad) / len(s) * 100, 1),
                             "Rango esperado": f"{fmt(lo)} a {fmt(hi)}", "Más extremo": f"{fmt(df.at[worst, col])} ({row_label(df, worst, label_cols)})"})
    outliers = pd.DataFrame(out_rows, columns=["Columna", "Atípicos", "% de filas", "Rango esperado", "Más extremo"])
    if not outliers.empty:
        worst_row = outliers.sort_values("Atípicos", ascending=False).iloc[0]
        highlights.append(f'"{worst_row["Columna"]}" tiene {int(worst_row["Atípicos"])} valor(es) atípico(s); el más extremo es {worst_row["Más extremo"]}.')

    # --- Categoría dominante ---
    categories: list[str] = []
    for col in cats[:3]:
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue
        text = f'**{col}**: "{counts.index[0]}" es la más frecuente ({fmt(counts.iloc[0] / counts.sum() * 100, 1)}% de las filas)'
        if nums:
            sums = df.groupby(col)[nums[0]].sum()
            if (sums >= 0).all() and sums.sum() > 0:
                top = sums.idxmax()
                text += f'; "{top}" aporta el {fmt(sums.max() / sums.sum() * 100, 1)}% del total de "{nums[0]}"'
        categories.append(text + ".")
    if categories:
        highlights.append(categories[0].replace("**", ""))

    return Insights(highlights[:6], stats, extremes, correlations, trends, outliers, categories, missing)


# =============================================================================
# 7. INTERFAZ
# =============================================================================

@st.cache_data(show_spinner=False, max_entries=64)
def fig_to_png(fig_json: str) -> bytes | None:
    """Imagen PNG del gráfico (requiere kaleido). Devuelve None si el servidor no puede generarla."""
    try:
        return pio.from_json(fig_json).to_image(format="png", scale=2)
    except Exception:
        return None


def plot_config(title: str) -> dict:
    """Barra de herramientas de Plotly. El ícono de cámara descarga el PNG directamente desde el navegador."""
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": slugify(title), "scale": 2},
    }


def render_chart_card(spec: ChartSpec, learning: bool, png_enabled: bool) -> None:
    """Un gráfico con sus botones de descarga y su explicación."""
    with st.container(border=True):
        st.plotly_chart(spec.fig, key=f"plot_{spec.key}", config=plot_config(spec.title), theme=None, **fit(st.plotly_chart))

        name = slugify(spec.title)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Descargar HTML interactivo", data=spec.fig.to_html(include_plotlyjs="cdn"),
                               file_name=f"{name}.html", mime="text/html", key=f"html_{spec.key}", **fit(st.download_button))
        with c2:
            if png_enabled:
                png = fig_to_png(spec.fig.to_json())
                if png:
                    st.download_button("Descargar imagen PNG", data=png, file_name=f"{name}.png",
                                       mime="image/png", key=f"png_{spec.key}", **fit(st.download_button))
                else:
                    st.caption("No pude generar el PNG aquí; usa el ícono de cámara del gráfico.")
            else:
                st.caption("Imagen: ícono de cámara del gráfico.")

        if learning:
            with st.expander("Cómo leer este gráfico"):
                st.markdown(f"**¿Por qué este gráfico?** {spec.why}")
                st.markdown(f"**¿Qué puedes interpretar?** {spec.interpret}")
                st.markdown("**Cómo mejorar el análisis**\n" + "\n".join(f"- {t}" for t in spec.tips))


def render_preview(df: pd.DataFrame, types: ColumnTypes) -> None:
    st.subheader("Vista previa de los datos")
    c = st.columns(5)
    c[0].metric("Filas", fmt(len(df)))
    c[1].metric("Columnas", len(df.columns))
    c[2].metric("Numéricas", len(types.numeric))
    c[3].metric("Categóricas", len(types.categorical))
    c[4].metric("Fechas", len(types.datetime))

    st.dataframe(df, height=380, **fit(st.dataframe))

    st.markdown("##### Tipo detectado por columna")
    info = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": [types.label(col) for col in df.columns],
        "Valores distintos": [df[col].nunique() for col in df.columns],
        "% vacío": [round(df[col].isna().mean() * 100, 1) for col in df.columns],
        "Ejemplo": [str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—" for col in df.columns],
    })
    st.dataframe(info, hide_index=True, **fit(st.dataframe))
    st.markdown('<p class="soft">Si un tipo está mal detectado, corrígelo en «Ajustes de columnas», en la barra lateral.</p>',
                unsafe_allow_html=True)


def render_charts(df: pd.DataFrame, types: ColumnTypes, agg: str, top_n: int, learning: bool, png_enabled: bool) -> None:
    st.subheader("Gráficos generados automáticamente")
    if not types.numeric and not types.categorical and not types.datetime:
        st.warning("No encontré columnas numéricas, categóricas ni de fecha para graficar.")
        return
    specs, skipped = auto_charts(df, types, agg, top_n)
    if not specs:
        st.warning("No pude generar gráficos automáticos con estos datos. Prueba en «Gráfico personalizado».")
    for i in range(0, len(specs), 2):  # dos gráficos por fila
        for col, spec in zip(st.columns(2), specs[i:i + 2]):
            with col:
                render_chart_card(spec, learning, png_enabled)
    if skipped:
        with st.expander(f"Gráficos omitidos ({len(skipped)})"):
            for msg in skipped:
                st.write("- " + msg)


def render_insights(df: pd.DataFrame, types: ColumnTypes, learning: bool) -> None:
    st.subheader("Insights")
    ins = compute_insights(df, types)

    st.markdown("##### Hallazgos clave")
    for h in ins.highlights:
        st.markdown(f"- {h}")

    if ins.extremes:
        st.markdown("##### Máximos, mínimos y promedios")
        for e in ins.extremes:
            st.markdown(f"- {e}")
        if not ins.stats.empty:
            with st.expander("Tabla de estadísticos"):
                st.dataframe(ins.stats, hide_index=True, **fit(st.dataframe))
        if learning:
            st.info("**Promedio vs. mediana:** el promedio se deja arrastrar por valores extremos; la mediana es el valor "
                    "del medio y resiste mejor. Si son muy distintos, hay extremos que conviene revisar.")

    st.markdown("##### Correlaciones")
    if ins.correlations.empty:
        st.write("No hay correlaciones relevantes (|r| ≥ 0,5) entre las columnas numéricas.")
    else:
        st.dataframe(ins.correlations, hide_index=True, **fit(st.dataframe))
        if learning:
            st.info("**Correlación (r):** va de -1 a 1. Cerca de 1, las dos columnas suben juntas; cerca de -1, una sube "
                    "cuando la otra baja; cerca de 0, no se relacionan linealmente. Que se muevan juntas no prueba que una cause la otra.")

    st.markdown("##### Tendencias en el tiempo")
    if ins.trends:
        for t in ins.trends:
            st.markdown(f"- {t}")
        if learning:
            st.info("**Tendencia:** compara el promedio del primer tercio del período con el del último, y mira qué tan "
                    "ordenada es la subida o bajada. Una tendencia débil puede ser solo ruido.")
    else:
        st.write("Se necesita una columna de fechas y una numérica (con al menos 6 registros) para calcular tendencias.")

    st.markdown("##### Valores atípicos")
    if ins.outliers.empty:
        st.write("No se detectaron valores atípicos.")
    else:
        st.dataframe(ins.outliers, hide_index=True, **fit(st.dataframe))
        if learning:
            st.info("**Valor atípico:** un dato muy lejos del resto (fuera de 1,5 veces el rango intercuartílico). "
                    "Puede ser un error de digitación o un caso especial: revisa la fila antes de descartarlo.")

    if ins.categories:
        st.markdown("##### Categorías")
        for c in ins.categories:
            st.markdown(f"- {c}")

    st.markdown("##### Valores vacíos")
    if ins.missing.empty:
        st.write("No hay valores vacíos.")
    else:
        st.dataframe(ins.missing, hide_index=True, **fit(st.dataframe))


def render_custom(df: pd.DataFrame, types: ColumnTypes, agg: str, top_n: int, learning: bool, png_enabled: bool) -> None:
    """Constructor manual: el usuario elige tipo de gráfico y columnas."""
    st.subheader("Gráfico personalizado")
    st.caption(f"Elige el tipo y las columnas. La agregación usada es: {AGG_NAME[agg]} (se cambia en la barra lateral).")

    kinds = ["Barras", "Líneas", "Área", "Dispersión", "Circular", "Histograma", "Caja", "Mapa de correlación"]
    kind = st.selectbox("Tipo de gráfico", kinds, key="custom_kind")
    all_cols = list(df.columns)
    nums = types.numeric + types.ids
    none = "(ninguno)"
    x = y = color = None
    numeric_cols: list[str] = []

    c1, c2, c3 = st.columns(3)
    if kind in ("Barras", "Circular"):
        x = c1.selectbox("Categorías (eje X)", types.categorical + types.text + [c for c in all_cols if c not in types.categorical + types.text], key=f"cx_{kind}")
        y_choice = c2.selectbox("Valor", ["(contar registros)"] + nums, key=f"cy_{kind}")
        y = None if y_choice == "(contar registros)" else y_choice
    elif kind in ("Líneas", "Área"):
        x = c1.selectbox("Eje X (tiempo u orden)", types.datetime + [c for c in all_cols if c not in types.datetime], key=f"cx_{kind}")
        y = c2.selectbox("Valor (eje Y)", nums, key=f"cy_{kind}") if nums else None
    elif kind == "Dispersión":
        x = c1.selectbox("Eje X", nums, key=f"cx_{kind}") if nums else None
        y = c2.selectbox("Eje Y", nums[1:] + nums[:1], key=f"cy_{kind}") if nums else None
        c = c3.selectbox("Color por", [none] + types.categorical, key=f"cc_{kind}")
        color = None if c == none else c
    elif kind == "Histograma":
        x = c1.selectbox("Columna numérica", nums, key=f"cx_{kind}") if nums else None
    elif kind == "Caja":
        x = c1.selectbox("Categorías (eje X)", types.categorical + types.text, key=f"cx_{kind}") if (types.categorical or types.text) else None
        y = c2.selectbox("Valor (eje Y)", nums, key=f"cy_{kind}") if nums else None
    else:  # Mapa de correlación
        numeric_cols = st.multiselect("Columnas numéricas", types.numeric, default=types.numeric[:12], key=f"cn_{kind}")

    try:
        title = default_title(kind, x, y, agg)
        fig = make_chart(kind, df, x=x, y=y, color=color, agg=agg, top_n=top_n, numeric_cols=numeric_cols, title=title)
        lesson = explain(kind, x, y, agg)
        render_chart_card(ChartSpec("custom", title, fig, **lesson), learning, png_enabled)
    except ValueError as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo generar el gráfico: {exc}")


def render_welcome() -> None:
    """Pantalla inicial, antes de cargar datos."""
    st.markdown("### Convierte una tabla en gráficos que se entienden")
    st.write("Sube un archivo desde la barra lateral o prueba con datos de ejemplo.")
    cols = st.columns(3)
    steps = [
        ("Sube tus datos", "Excel (.xlsx, .xls) o CSV. La app detecta qué columnas son números, categorías y fechas."),
        ("Explora los gráficos", "Genera barras, líneas, área, dispersión, circular, histograma, caja y mapa de correlación."),
        ("Entiende lo que ves", "Los insights resumen lo importante y el Modo Aprendizaje explica cómo leer cada gráfico."),
    ]
    for col, (title, text) in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(f'<p class="soft">{text}</p>', unsafe_allow_html=True)


# =============================================================================
# 8. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    st.title("Procesamiento De Datos")
    st.markdown('<p class="soft">Sube una tabla y obtén gráficos, insights y explicaciones para entender tus datos.</p>',
                unsafe_allow_html=True)

    # ---------- Barra lateral: datos ----------
    with st.sidebar:
        st.header("Datos")
        uploaded = st.file_uploader("Sube tu archivo", type=["xlsx", "xls", "csv"],
                                    help="Formatos: Excel (.xlsx, .xls) o CSV. La primera fila debe tener los encabezados.")
        if st.button("Probar con datos de ejemplo", **fit(st.button)):
            st.session_state["use_sample"] = True
        if uploaded is not None:
            st.session_state["use_sample"] = False

    # ---------- Carga ----------
    df_raw: pd.DataFrame | None = None
    source = ""
    try:
        if uploaded is not None:
            data = uploaded.getvalue()
            sheets = list_sheets(data, uploaded.name)
            sheet = None
            if len(sheets) > 1:
                sheet = st.sidebar.selectbox("Hoja del Excel", sheets)
            df_raw = load_dataframe(data, uploaded.name, sheet)
            source = f"{uploaded.name}|{sheet}|{len(data)}"
        elif st.session_state.get("use_sample"):
            df_raw = sample_dataframe()
            source = "ejemplo"
            st.sidebar.caption("Mostrando datos de ejemplo.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"No pude leer el archivo: {exc}")
        st.stop()

    if df_raw is None:
        render_welcome()
        return

    # ---------- Barra lateral: opciones ----------
    with st.sidebar:
        st.header("Opciones")
        agg_label = st.selectbox("Cómo agregar los valores", list(AGG_OPTIONS), help="Se usa en barras, líneas y circular cuando una categoría o fecha se repite.")
        agg = AGG_OPTIONS[agg_label]
        top_n = st.slider("Máximo de categorías en barras", 5, 30, 15)
        max_categories = st.slider("Máximo de valores distintos para considerar una columna «categórica»", 5, 50, 20)
        learning = st.toggle("Modo Aprendizaje", value=True, help="Muestra explicaciones junto a cada gráfico e insight.")
        png_enabled = st.toggle("Botones de descarga PNG", value=False,
                                help="Genera la imagen en el servidor (requiere kaleido). El ícono de cámara del gráfico siempre funciona.")
        with st.expander("Ajustes de columnas"):
            exclude = st.multiselect("Excluir del análisis", list(df_raw.columns), key=f"ex_{source}")
            numeric_now = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]) and c not in exclude]
            as_category = st.multiselect("Tratar números como categorías", numeric_now, key=f"cat_{source}",
                                         help="Útil para columnas como año, turno o calificación.")

    # ---------- Preparación ----------
    df = df_raw.drop(columns=exclude)
    for col in as_category:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if df.empty or len(df.columns) == 0:
        st.warning("No quedan columnas para analizar. Revisa «Excluir del análisis».")
        return
    types = detect_types(df, max_categories)

    # ---------- Secciones ----------
    tab_data, tab_charts, tab_insights, tab_custom = st.tabs(["Datos", "Gráficos", "Insights", "Gráfico personalizado"])
    with tab_data:
        render_preview(df, types)
    with tab_charts:
        render_charts(df, types, agg, top_n, learning, png_enabled)
    with tab_insights:
        render_insights(df, types, learning)
    with tab_custom:
        render_custom(df, types, agg, top_n, learning, png_enabled)

    st.markdown('<p class="soft">Los datos se procesan en memoria durante tu sesión; la app no los guarda en disco.</p>',
                unsafe_allow_html=True)


main()

