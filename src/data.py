"""Dados: download, limpeza, reamostragem horária, features, divisão e janelas.

Usado por train.py, evaluate.py, predict.py, pela API e pelo notebook.
`python -m src.data` baixa o dataset, prepara tudo e imprime um resumo.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
URL_DATASET = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
NOME_TXT = "household_power_consumption.txt"
CACHE_HORARIO = DATA_DIR / "horario.pkl"
RESUMO_JSON = DATA_DIR / "resumo.json"
CACHE_MASCARA = DATA_DIR / "preenchidos.pkl"

ALVO = "Global_active_power"
JANELA_ENTRADA = 168  # horas
HORIZONTE = 24  # horas
LACUNA_CURTA = 6  # horas: lacunas até este tamanho são interpoladas

COLUNAS_MEDIDAS = [
    ALVO,
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
FEATURES_CALENDARIO = ["hora_sin", "hora_cos", "dia_sin", "dia_cos", "fim_de_semana"]
FEATURES = COLUNAS_MEDIDAS + FEATURES_CALENDARIO
INDICE_ALVO = FEATURES.index(ALVO)  # 0

NOMES_DATETIME = ("datetime", "data_hora", "timestamp", "data", "date_time")


# ---------------------------------------------------------------- download e leitura
def baixar_dataset(destino: Path = DATA_DIR) -> Path:
    """Baixa e extrai o dataset se ainda não existir. Devolve o caminho do .txt."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    txt = destino / NOME_TXT
    if txt.exists():
        return txt
    print(f"Baixando {URL_DATASET} ...")
    req = urllib.request.Request(URL_DATASET, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        conteudo = resp.read()
    print(f"  {len(conteudo) / 1e6:.1f} MB baixados; extraindo...")
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        nome = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(nome) as origem, open(txt, "wb") as saida:
            saida.write(origem.read())
    return txt


def _para_numerico(df: pd.DataFrame, colunas) -> pd.DataFrame:
    """Converte colunas para float aceitando '?' e vírgula decimal."""
    for c in colunas:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype("string").str.strip().str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s.replace({"?": None, "": None}), errors="coerce")
        elif c in df.columns:
            df[c] = df[c].astype("float64")
    return df


def limpar_bruto(df: pd.DataFrame) -> pd.DataFrame:
    """Une Date+Time (dd/mm/aaaa hh:mm:ss) em índice datetime e converte as medições."""
    texto = df["Date"].astype("string").str.strip() + " " + df["Time"].astype("string").str.strip()
    try:
        indice = pd.to_datetime(texto, format="%d/%m/%Y %H:%M:%S")
    except (ValueError, TypeError):
        indice = pd.to_datetime(texto, dayfirst=True, errors="coerce")
        if indice.isna().mean() > 0.5:
            raise ValueError("Datas ilegíveis: use o formato dd/mm/aaaa nas colunas Date e Time.")
    df = df.drop(columns=["Date", "Time"])
    df.index = pd.DatetimeIndex(indice, name="datetime")
    df = df[~df.index.isna()]
    df = _para_numerico(df, [c for c in COLUNAS_MEDIDAS if c in df.columns])
    return df[[c for c in COLUNAS_MEDIDAS if c in df.columns]].sort_index()


def carregar_bruto(caminho: Path) -> pd.DataFrame:
    """Lê o arquivo UCI (sep=';', '?' como NaN) com índice datetime por minuto."""
    df = pd.read_csv(caminho, sep=";", na_values=["?"], low_memory=False,
                     dtype={"Date": str, "Time": str})
    return limpar_bruto(df)


# ---------------------------------------------------------------- série horária
def _preencher_serie(s: pd.Series) -> pd.Series:
    """Interpola lacunas curtas; lacunas longas recebem a mesma hora da semana anterior."""
    na = s.isna()
    if not na.any():
        return s
    grupos = (na != na.shift()).cumsum()
    tamanho = na.groupby(grupos).transform("sum")
    curta = na & (tamanho <= LACUNA_CURTA)
    interpolada = s.interpolate(method="time", limit_area="inside")
    s = s.where(~curta, interpolada)
    for _ in range(100):  # semana anterior (pode precisar repetir em lacunas de várias semanas)
        if not s.isna().any():
            break
        s = s.fillna(s.shift(168))
    for _ in range(100):  # início da série: usa a semana seguinte
        if not s.isna().any():
            break
        s = s.fillna(s.shift(-168))
    return s.ffill().bfill()


def para_horario(df: pd.DataFrame, devolver_mascara: bool = False):
    """Reamostra para média horária e preenche faltantes (interpolação curta;
    lacunas longas com a mesma hora da semana anterior).

    `df.attrs['faltantes_preenchidos']` guarda o número de horas sem alvo antes do preenchimento.
    Com `devolver_mascara=True` devolve também a série booleana das horas preenchidas.
    """
    colunas = [c for c in COLUNAS_MEDIDAS if c in df.columns]
    horario = df[colunas].resample("h").mean()
    mascara = horario[ALVO].isna()
    for c in colunas:
        horario[c] = _preencher_serie(horario[c])
    horario.index.name = "datetime"
    horario.attrs["faltantes_preenchidos"] = int(mascara.sum())
    return (horario, mascara) if devolver_mascara else horario


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta seno/cosseno de hora e dia da semana e flag de fim de semana."""
    df = df.copy()
    hora = df.index.hour.to_numpy()
    dia = df.index.day_of_week.to_numpy()
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["dia_sin"] = np.sin(2 * np.pi * dia / 7)
    df["dia_cos"] = np.cos(2 * np.pi * dia / 7)
    df["fim_de_semana"] = (dia >= 5).astype(float)
    return df


def dividir(df: pd.DataFrame, treino: float = 0.70, validacao: float = 0.15):
    """Divisão cronológica. Devolve (treino, validacao, teste)."""
    n = len(df)
    i1 = int(n * treino)
    i2 = int(n * (treino + validacao))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def criar_janelas(valores: np.ndarray, indice_alvo: int,
                  entrada: int = JANELA_ENTRADA, horizonte: int = HORIZONTE,
                  passo: int = 1):
    """Devolve X (N, entrada, F) e y (N, horizonte) a partir de valores já normalizados.

    A janela i usa as linhas [i, i+entrada) como entrada e [i+entrada, i+entrada+horizonte)
    como alvo. `passo` > 1 pula janelas (usado apenas no treino, se necessário).
    """
    valores = np.asarray(valores, dtype=np.float32)
    if valores.ndim == 1:
        valores = valores[:, None]
    n = len(valores) - entrada - horizonte + 1
    if n <= 0:
        f = valores.shape[1]
        return np.empty((0, entrada, f), np.float32), np.empty((0, horizonte), np.float32)
    vista = np.lib.stride_tricks.sliding_window_view(valores[: n + entrada - 1], entrada, axis=0)
    X = np.ascontiguousarray(vista.transpose(0, 2, 1)[::passo])
    alvo = valores[entrada:, indice_alvo]
    y = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(alvo, horizonte)[:n][::passo])
    return X, y


# ---------------------------------------------------------------- pipeline
def carregar_horario(forcar: bool = False) -> pd.DataFrame:
    """Série horária limpa (com cache em data/horario.pkl)."""
    if CACHE_HORARIO.exists() and not forcar:
        df = pd.read_pickle(CACHE_HORARIO)
        if RESUMO_JSON.exists():
            df.attrs.update(json.loads(RESUMO_JSON.read_text(encoding="utf-8")))
        return df
    txt = baixar_dataset()
    bruto = carregar_bruto(txt)
    horario, mascara = para_horario(bruto, devolver_mascara=True)
    mascara.to_pickle(CACHE_MASCARA)
    resumo = {
        "linhas_brutas": int(len(bruto)),
        "faltantes_brutos": int(bruto[ALVO].isna().sum()),
        "pct_faltantes_brutos": float(bruto[ALVO].isna().mean() * 100),
        "inicio_bruto": str(bruto.index.min()),
        "fim_bruto": str(bruto.index.max()),
        "horas": int(len(horario)),
        "faltantes_preenchidos": horario.attrs["faltantes_preenchidos"],
    }
    horario.attrs.update(resumo)
    horario.to_pickle(CACHE_HORARIO)
    RESUMO_JSON.write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    return horario


def parametros_imputacao(df_treino: pd.DataFrame) -> dict:
    """Parâmetros (só do treino) para estimar colunas ausentes em uploads horários."""
    a, b = np.polyfit(df_treino[ALVO].to_numpy(), df_treino["Global_intensity"].to_numpy(), 1)
    return {
        "intensidade_a": float(a),
        "intensidade_b": float(b),
        "medias": {c: float(df_treino[c].mean()) for c in COLUNAS_MEDIDAS},
    }


def completar_colunas(df: pd.DataFrame, imputacao: dict) -> tuple[pd.DataFrame, list[str]]:
    """Estima medições ausentes (intensidade pela relação linear com o alvo, demais pela média
    do treino). Devolve o DataFrame completo e a lista de colunas estimadas."""
    df = df.copy()
    estimadas = []
    for c in COLUNAS_MEDIDAS:
        if c in df.columns:
            continue
        if c == "Global_intensity":
            df[c] = imputacao["intensidade_a"] * df[ALVO] + imputacao["intensidade_b"]
        else:
            df[c] = imputacao["medias"][c]
        estimadas.append(c)
    return df[COLUNAS_MEDIDAS], estimadas


def preparar_tudo(df_horario: pd.DataFrame | None = None, passo_treino: int = 1) -> dict:
    """Pipeline completo: devolve os conjuntos em janelas, o scaler (ajustado só
    no treino) e os DataFrames horários de cada divisão."""
    if df_horario is None:
        df_horario = carregar_horario()
    df_feat = adicionar_features(df_horario[COLUNAS_MEDIDAS])
    tr, va, te = dividir(df_feat)

    scaler = StandardScaler().fit(tr[FEATURES].to_numpy())  # só treino
    saida = {"scaler": scaler, "df_treino": tr, "df_validacao": va, "df_teste": te,
             "imputacao": parametros_imputacao(tr)}
    for nome, parte, passo in (("treino", tr, passo_treino), ("validacao", va, 1), ("teste", te, 1)):
        valores = scaler.transform(parte[FEATURES].to_numpy())
        X, y = criar_janelas(valores, INDICE_ALVO, passo=passo)
        inicio = parte.index[JANELA_ENTRADA: JANELA_ENTRADA + len(y) * passo: passo] if len(y) else parte.index[:0]
        saida[f"X_{nome}"], saida[f"y_{nome}"] = X, y
        saida[f"inicio_{nome}"] = inicio[: len(y)]  # primeira hora prevista de cada janela
    return saida


def desnormalizar_alvo(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return y * scaler.scale_[INDICE_ALVO] + scaler.mean_[INDICE_ALVO]


# ---------------------------------------------------------------- CSV do usuário (API)
def ler_csv_usuario(conteudo: bytes) -> pd.DataFrame:
    """Lê um CSV enviado (formato bruto UCI ou horário) e devolve a série horária limpa.

    Levanta ValueError com mensagem em português quando o arquivo é inválido.
    """
    if not conteudo or not conteudo.strip():
        raise ValueError("O arquivo enviado está vazio.")
    texto = None
    for cod in ("utf-8-sig", "latin-1"):
        try:
            texto = conteudo.decode(cod)
            break
        except UnicodeDecodeError:
            continue
    primeira = texto.splitlines()[0]
    sep = ";" if primeira.count(";") >= primeira.count(",") and ";" in primeira else ","
    try:
        df = pd.read_csv(io.StringIO(texto), sep=sep, na_values=["?"], dtype=str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Não foi possível ler o arquivo como CSV. Verifique se é um texto separado "
                         "por ';' (formato UCI) ou ',' (formato horário).") from exc
    df.columns = [str(c).strip() for c in df.columns]

    if ALVO not in df.columns:
        raise ValueError(f"O CSV precisa da coluna '{ALVO}' (consumo ativo em kW).")

    if "Date" in df.columns and "Time" in df.columns:
        df = limpar_bruto(df)
    else:
        col_dt = next((c for c in df.columns if c.lower() in NOMES_DATETIME), None)
        if col_dt is None:
            raise ValueError("O CSV precisa de uma coluna 'datetime' (formato horário) "
                             "ou das colunas 'Date' e 'Time' (formato UCI).")
        bruto = df[col_dt].astype("string").str.strip()
        indice = pd.to_datetime(bruto, errors="coerce", format="ISO8601")
        if indice.isna().mean() > 0.5:
            indice = pd.to_datetime(bruto, errors="coerce", dayfirst=True, format="mixed")
        if indice.isna().mean() > 0.5:
            raise ValueError(f"Datas ilegíveis na coluna '{col_dt}'. Use AAAA-MM-DD HH:MM.")
        df = df.drop(columns=[col_dt])
        df.index = pd.DatetimeIndex(indice, name="datetime")
        df = df[~df.index.isna()]
        df = _para_numerico(df, [c for c in COLUNAS_MEDIDAS if c in df.columns])
        df = df[[c for c in COLUNAS_MEDIDAS if c in df.columns]].sort_index()

    if df[ALVO].notna().sum() == 0:
        raise ValueError(f"A coluna '{ALVO}' não tem nenhum valor numérico.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    horario = para_horario(df)
    if len(horario) < JANELA_ENTRADA:
        raise ValueError(f"Histórico curto demais: {len(horario)} h. São necessárias pelo menos "
                         f"{JANELA_ENTRADA} h (7 dias) para prever.")
    return horario


def resumo_serie(df: pd.DataFrame) -> dict:
    """Resumo no formato de resposta de /api/upload e /api/exemplo."""
    fmt = "%Y-%m-%dT%H:%M:%S"
    return {
        "inicio": df.index.min().strftime(fmt),
        "fim": df.index.max().strftime(fmt),
        "horas": int(len(df)),
        "faltantes_preenchidos": int(df.attrs.get("faltantes_preenchidos", 0)),
        "media_kw": round(float(df[ALVO].mean()), 3),
        "pico_kw": round(float(df[ALVO].max()), 3),
    }


if __name__ == "__main__":
    horario = carregar_horario()
    a = horario.attrs
    print("=== Dataset bruto (UCI 235) ===")
    print(f"Linhas por minuto: {a['linhas_brutas']:,}")
    print(f"Período: {a['inicio_bruto']} -> {a['fim_bruto']}")
    print(f"Minutos sem medição ('?'): {a['faltantes_brutos']:,} ({a['pct_faltantes_brutos']:.2f}%)")
    print("\n=== Série horária ===")
    print(f"Horas: {len(horario):,}  ({horario.index.min()} -> {horario.index.max()})")
    print(f"Horas sem dado antes do preenchimento: {a['faltantes_preenchidos']}")
    print(f"NaN restantes após preenchimento: {int(horario.isna().sum().sum())}")
    print(horario[ALVO].describe().round(3).to_string())

    dados = preparar_tudo(horario)
    print("\n=== Divisão cronológica e janelas (168 h -> 24 h) ===")
    for nome in ("treino", "validacao", "teste"):
        parte = dados[f"df_{nome}"]
        print(f"{nome:<10} {parte.index.min()} -> {parte.index.max()}  ({len(parte):,} h)  "
              f"X={dados[f'X_{nome}'].shape}  y={dados[f'y_{nome}'].shape}")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"Scaler ajustado só no treino: média do alvo = {dados['scaler'].mean_[INDICE_ALVO]:.4f} kW")
