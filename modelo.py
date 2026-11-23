"""
Panel de mercado: régimen + volatilidad esperada.
Se ejecuta cada mañana en GitHub Actions y escribe docs/datos.json,
que la página docs/index.html muestra.
"""
import os, json, datetime as dt, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
INICIO = "2010-01-01"
ACCIONES = ['AAPL','ADBE','ADI','ADP','AMAT','AMD','AMGN','AMZN','ASML','AVGO','BKNG','CDNS','COST','CRWD','CSCO',
            'CTAS','GILD','GOOGL','HON','INTC','INTU','ISRG','KLAC','LRCX','MAR','MDLZ','MELI','META','MRVL','MSFT',
            'MU','NFLX','NVDA','ON','ORLY','PANW','PEP','PYPL','QCOM','REGN','SBUX','SNPS','TSLA','TSM','TXN','VRTX']
INDICES = ['QQQ', 'SPY']
NOMBRES = {'AAPL':'Apple','ADBE':'Adobe','ADI':'Analog Devices','ADP':'ADP','AMAT':'Applied Materials','AMD':'AMD',
 'AMGN':'Amgen','AMZN':'Amazon','ASML':'ASML','AVGO':'Broadcom','BKNG':'Booking','CDNS':'Cadence','COST':'Costco',
 'CRWD':'CrowdStrike','CSCO':'Cisco','CTAS':'Cintas','GILD':'Gilead','GOOGL':'Alphabet','HON':'Honeywell','INTC':'Intel',
 'INTU':'Intuit','ISRG':'Intuitive Surgical','KLAC':'KLA','LRCX':'Lam Research','MAR':'Marriott','MDLZ':'Mondelez',
 'MELI':'MercadoLibre','META':'Meta','MRVL':'Marvell','MSFT':'Microsoft','MU':'Micron','NFLX':'Netflix','NVDA':'NVIDIA',
 'ON':'ON Semiconductor','ORLY':"O'Reilly",'PANW':'Palo Alto Networks','PEP':'PepsiCo','PYPL':'PayPal','QCOM':'Qualcomm',
 'REGN':'Regeneron','SBUX':'Starbucks','SNPS':'Synopsys','TSLA':'Tesla','TSM':'TSMC','TXN':'Texas Instruments',
 'VRTX':'Vertex','QQQ':'Nasdaq-100 (QQQ)','SPY':'S&P 500 (SPY)'}
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "datos.json")


def descargar():
    import yfinance as yf
    tick = ACCIONES + INDICES + ['^VIX', '^VIX3M']
    raw = yf.download(tick, start=INICIO, auto_adjust=True, progress=False, group_by='ticker', threads=True)
    px = {}
    for t in tick:
        try:
            df = raw[t][['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
            if len(df) > 300: px[t] = df
        except KeyError:
            pass
    return px


def datos_simulados():
    rng = np.random.default_rng(1); idx = pd.bdate_range('2010-01-04', dt.date.today() - dt.timedelta(days=1)); px = {}
    for t in ACCIONES + INDICES + ['^VIX', '^VIX3M']:
        n = len(idx); s = 0.012 * np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.3)
        c = (20 if t.startswith('^') else 100) * np.exp(np.cumsum(rng.normal(0.0004, 1, n) * s))
        o = c * (1 + rng.normal(0, 0.004, n)); h = np.maximum(c, o) * (1 + abs(rng.normal(0, 1, n)) * s)
        l = np.minimum(c, o) * (1 - abs(rng.normal(0, 1, n)) * s)
        px[t] = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': rng.integers(1e6, 5e6, n).astype(float)}, index=idx)
    return px


def main():
    PX = datos_simulados() if os.environ.get("MOCK") else descargar()
    QQ = PX['QQQ']; IDX = QQ.index; q_ret = QQ['Close'].pct_change()
    vix = PX['^VIX']['Close'].reindex(IDX).ffill()
    vix3m = PX['^VIX3M']['Close'].reindex(IDX).ffill() if '^VIX3M' in PX else vix

    # ---------- mercado y régimen ----------
    M = pd.DataFrame(index=IDX)
    M['vix'] = vix; M['vix_estructura'] = vix / vix3m
    M['nasdaq_vs_media200'] = QQ['Close'] / QQ['Close'].rolling(200).mean() - 1
    M['nasdaq_vol_20d'] = q_ret.rolling(20).std() * np.sqrt(252)
    cierres = pd.DataFrame({t: PX[t]['Close'].reindex(IDX) for t in ACCIONES if t in PX})
    M['amplitud'] = (cierres > cierres.rolling(50).mean()).where(cierres.notna()).mean(axis=1)
    M['nasdaq_rango_hoy'] = np.log(QQ['High'] / QQ['Low'])

    pm = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
    E = pd.DataFrame(index=IDX)
    E['vix_pct'] = pm(M['vix']); E['vol_pct'] = pm(M['nasdaq_vol_20d'])
    E['estructura'] = ((M['vix_estructura'] - 0.80) / 0.30).clip(0, 1)
    E['amplitud_baja'] = 1 - M['amplitud']
    E['bajo_media200'] = (-M['nasdaq_vs_media200'] / 0.15).clip(0, 1)
    ESTRES = E.mean(axis=1, skipna=False).dropna()
    u50 = ESTRES.expanding(min_periods=500).quantile(0.50).shift(1)
    u90 = ESTRES.expanding(min_periods=500).quantile(0.90).shift(1)
    REG = pd.Series(np.where(ESTRES > u90, 'Pánico', np.where(ESTRES > u50, 'Tensión', 'Calma')), ESTRES.index)[u90.notna()]

    q5 = QQ['Open'].shift(-6) / QQ['Open'].shift(-1) - 1
    qr = np.log(QQ['High'] / QQ['Low']).shift(-1)
    t = pd.DataFrame({'reg': REG, 'q5': q5, 'rng': qr}).dropna(); t = t[t.index >= '2019-01-01']
    tabla_reg = []
    for r in ['Calma', 'Tensión', 'Pánico']:
        g = t[t.reg == r]
        if len(g) == 0: continue
        tabla_reg.append({'regimen': r, 'pct_dias': len(g) / len(t) * 100, 'qqq_5d_prom': g.q5.mean() * 100,
                          'semanas_positivas': (g.q5 > 0).mean() * 100, 'peor_5pct': g.q5.quantile(0.05) * 100,
                          'mejor_5pct': g.q5.quantile(0.95) * 100, 'rango_diario': g.rng.mean() * 100})

    # ---------- volatilidad ----------
    filas = []
    for tk in ACCIONES + INDICES:
        if tk not in PX: continue
        df = PX[tk].reindex(IDX); o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        rg = np.log(h / l); r = c.pct_change(); f = pd.DataFrame(index=IDX)
        f['l_rango_hoy'] = np.log(rg.clip(lower=1e-4)); f['l_rango_5d'] = np.log(rg.rolling(5).mean().clip(lower=1e-4))
        f['l_rango_22d'] = np.log(rg.rolling(22).mean().clip(lower=1e-4))
        f['gap_abs'] = (o / c.shift() - 1).abs(); f['ret_1d'] = r
        f['beta'] = r.rolling(120).cov(q_ret) / q_ret.rolling(120).var()
        f['volumen_relativo'] = df['Volume'] / df['Volume'].rolling(20).mean()
        f['rango_hoy'] = rg; f['y'] = np.log(rg.shift(-1).clip(lower=1e-4)); f['ticker'] = tk
        filas.append(f)
    V = pd.concat(filas).join(M, how='left').replace([np.inf, -np.inf], np.nan)
    V['dia_semana'] = V.index.dayofweek
    V = V[(V.index >= '2012-01-01') & V['l_rango_22d'].notna()]
    HAR = ['l_rango_hoy', 'l_rango_5d', 'l_rango_22d']
    EXT = HAR + ['gap_abs', 'ret_1d', 'vix', 'vix_estructura', 'nasdaq_vol_20d', 'nasdaq_rango_hoy', 'amplitud', 'beta', 'volumen_relativo', 'dia_semana']
    L = V.dropna(subset=HAR + ['y'])
    gb = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200, random_state=0)

    # validación: últimos 2 años completos + año en curso
    ult = L.index.max().year; val = []
    for a in [ult - 2, ult - 1, ult]:
        tr = L[L.index < pd.Timestamp(f'{a}-01-01') - pd.Timedelta(days=10)]; te = L[L.index.year == a]
        if len(te) < 500: continue
        m1 = LinearRegression().fit(tr[HAR], tr['y']); m2 = gb().fit(tr[EXT], tr['y'])
        val.append({'anio': a, 'r2_har': r2_score(te['y'], m1.predict(te[HAR])), 'r2_modelo': r2_score(te['y'], m2.predict(te[EXT]))})
    usar_gb = np.mean([v['r2_modelo'] for v in val]) > np.mean([v['r2_har'] for v in val]) + 0.005 if val else True
    modelo = gb().fit(L[EXT], L['y']) if usar_gb else LinearRegression().fit(L[HAR], L['y'])

    ultimo = V.index.max(); hoy = V[V.index == ultimo].set_index('ticker')
    pred = np.exp(modelo.predict(hoy[EXT] if usar_gb else hoy[HAR])) * 100
    normal = V[V.index > ultimo - pd.Timedelta(days=365)].groupby('ticker')['rango_hoy'].median() * 100
    vol = []
    for i, tk in enumerate(hoy.index):
        precio = float(PX[tk]['Close'].iloc[-1]); prev = float(PX[tk]['Close'].iloc[-2])
        vol.append({'ticker': tk, 'nombre': NOMBRES.get(tk, tk), 'precio': round(precio, 2), 'var_dia': round((precio / prev - 1) * 100, 2),
                    'rango_esperado': round(float(pred[i]), 3), 'rango_normal': round(float(normal.get(tk, np.nan)), 3),
                    'vs_normal': round(float(pred[i] / normal.get(tk, np.nan)), 2), 'indice': tk in INDICES})

    # ---------- historia para el gráfico (2 años) ----------
    hz = ESTRES[ESTRES.index >= ultimo - pd.Timedelta(days=730)]
    historia = [{'f': d.strftime('%Y-%m-%d'), 'e': round(float(v), 3), 'r': REG.get(d, ''),
                 'q': round(float(QQ['Close'].get(d, np.nan)), 2), 'u50': round(float(u50.get(d, np.nan)), 3), 'u90': round(float(u90.get(d, np.nan)), 3)}
                for d, v in hz.items()]

    out = {
        'fecha_datos': str(ESTRES.index[-1].date()),
        'generado_utc': dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
        'simulado': bool(os.environ.get("MOCK")),
        'regimen': REG.iloc[-1], 'estres': round(float(ESTRES.iloc[-1]), 3),
        'umbral_tension': round(float(u50.iloc[-1]), 3), 'umbral_panico': round(float(u90.iloc[-1]), 3),
        'componentes': {k: round(float(v), 3) for k, v in E.iloc[-1].items()},
        'mercado': {'vix': round(float(M['vix'].iloc[-1]), 2), 'vix_estructura': round(float(M['vix_estructura'].iloc[-1]), 3),
                    'amplitud': round(float(M['amplitud'].iloc[-1]) * 100, 1), 'nasdaq_vs_media200': round(float(M['nasdaq_vs_media200'].iloc[-1]) * 100, 2)},
        'tabla_regimen': [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in tabla_reg],
        'validacion_vol': [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in val],
        'modelo_vol': 'boosting' if usar_gb else 'HAR',
        'volatilidad': sorted(vol, key=lambda x: -x['rango_esperado']),
        'historia': historia,
    }
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, 'w') as f:
        f.write(json.dumps(out, ensure_ascii=False).replace('NaN', 'null'))
    print('Listo:', out['fecha_datos'], out['regimen'], f"estrés {out['estres']}", '· modelo vol:', out['modelo_vol'])


if __name__ == '__main__':
    main()
