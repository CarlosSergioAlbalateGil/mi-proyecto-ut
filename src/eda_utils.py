"""
eda_utils.py
============
Funciones de utilidad para el Análisis Exploratorio de Datos (EDA) de medidas
radar FMCW indoor almacenadas en HDF5.

Estructura del tensor R  →  shape (N_t, N_f, 2, 2, 4)
    eje 0  (N_t)  — captura temporal
    eje 1  (N_f)  — bin de frecuencia
    eje 2  ( 2 )  — índice TX  (0 o 1)
    eje 3  ( 2 )  — índice RX  (0 o 1)
    eje 4  ( 4 )  — antena receptora física (0-3)

Configuraciones:
    (TX=0, RX=0) y (TX=1, RX=1) → Monostático
    (TX=0, RX=1) y (TX=1, RX=0) → Bistático

Secciones
---------
1. Carga y descripción
2. Filtrado y pre-procesado
3. Visualización
4. Estadísticas
5. Guardado
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

C: float = 3e8  # Velocidad de la luz (m/s)

CONFIG_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "Mono TX0-RX0",
    (1, 1): "Mono TX1-RX1",
    (0, 1): "Bi   TX0-RX1",
    (1, 0): "Bi   TX1-RX0",
}

_WINDOWS: dict[str, object] = {
    "hamming":  np.hamming,
    "hann":     np.hanning,
    "blackman": np.blackman,
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y DESCRIPCIÓN
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> dict:
    """
    Carga un archivo HDF5 de medidas FMCW y devuelve un diccionario estructurado.

    Parameters
    ----------
    path : str or Path
        Ruta al archivo .h5.

    Returns
    -------
    dict con claves:
        'timestamps' : np.ndarray (N_t,)
        'frequency'  : np.ndarray (N_f,)   — en Hz
        'R'          : np.ndarray complex  (N_t, N_f, 2, 2, 4)
        'path'       : Path
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        timestamps = np.array(f["timestamps"])
        frequency  = np.array(f["frequency"])
        R          = np.array(f["R"])
    return {"timestamps": timestamps, "frequency": frequency, "R": R, "path": path}


def describe_dataset(data: dict) -> None:
    """
    Imprime un resumen descriptivo y la interpretación física del dataset.

    Parameters
    ----------
    data : dict
        Salida de :func:`load_data`.
    """
    R    = data["R"]
    freq = data["frequency"]
    ts   = data["timestamps"]

    bw        = freq[-1] - freq[0]
    df        = freq[1]  - freq[0]
    range_res = C / bw          # convenio: c/BW por bin  (igual que compute_range_profile)
    max_range = C / df          # N bins × range_per_bin = (BW/Δf) × (c/BW) = c/Δf


    print("=" * 62)
    print("  DESCRIPCIÓN DEL DATASET")
    print("=" * 62)
    print(f"  Archivo : {data['path'].name}")
    print()
    print("  ── Dimensiones del tensor R ──────────────────────────")
    dim_names = ["tiempo", "frecuencia", "TX_idx", "RX_idx", "rx_antenna"]
    for i, (size, name) in enumerate(zip(R.shape, dim_names)):
        print(f"    eje {i}  ({size:>4d})  → {name}")
    print()
    print("  ── Configuraciones TX/RX ─────────────────────────────")
    print(f"    {'(TX,RX)':<10}  {'Tipo':<14}  Label")
    for (tx, rx), label in CONFIG_LABELS.items():
        tipo = "Monostático" if tx == rx else "Bistático  "
        print(f"    ({tx}, {rx})        {tipo}   {label}")
    print()
    print("  ── Frecuencias ───────────────────────────────────────")
    print(f"    Rango      : {freq.min()*1e-9:.4f} – {freq.max()*1e-9:.4f} GHz")
    print(f"    BW         : {bw*1e-9:.4f} GHz")
    print(f"    Paso Δf    : {df*1e-6:.3f} MHz")
    print(f"    Res. rango : {range_res*100:.2f} cm")
    print(f"    Max. rango : {max_range:.2f} m")
    print()
    print("  ── Timestamps ────────────────────────────────────────")
    print(f"    N capturas : {len(ts)}")
    print(f"    Rango      : [{ts.min():.4f}, {ts.max():.4f}]")
    print(f"    Duración   : {ts.max()-ts.min():.4f} (u.t.)")
    print()
    amp = np.abs(R)
    print("  ── Estadísticas |R| ──────────────────────────────────")
    print(f"    dtype      : {R.dtype}")
    print(f"    min        : {amp.min():.4e}")
    print(f"    max        : {amp.max():.4e}")
    print(f"    mean       : {amp.mean():.4e}")
    db_dyn = 20 * np.log10(amp.max() / (amp.mean() + 1e-12))
    print(f"    Dinámica   : {db_dyn:.1f} dB  (max vs mean)")
    print("=" * 62)


def get_config_labels() -> dict[tuple[int, int], str]:
    """Devuelve el mapa (TX_idx, RX_idx) → etiqueta descriptiva."""
    return CONFIG_LABELS.copy()


# ──────────────────────────────────────────────────────────────────────────────
# 2. FILTRADO Y PRE-PROCESADO
# ──────────────────────────────────────────────────────────────────────────────

def select_band(data: dict, f_min_ghz: float, f_max_ghz: float) -> dict:
    """
    Recorta el dataset al rango de frecuencias [f_min_ghz, f_max_ghz] GHz.

    Returns
    -------
    Nuevo dict con 'frequency' y 'R' recortados.
    """
    freq = data["frequency"]
    mask = (freq >= f_min_ghz * 1e9) & (freq <= f_max_ghz * 1e9)
    result = data.copy()
    result["frequency"] = freq[mask]
    result["R"]         = data["R"][:, mask, :, :, :]
    return result


def apply_window(signal: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """
    Aplica una ventana espectral al **eje 1** (frecuencia) del array.

    Parameters
    ----------
    signal      : array con la frecuencia en el eje 1.
    window_type : 'hamming' | 'hann' | 'blackman'

    Returns
    -------
    Array de igual forma que `signal` multiplicado por la ventana.
    """
    if window_type not in _WINDOWS:
        raise ValueError(f"window_type debe ser uno de {list(_WINDOWS)}")
    n   = signal.shape[1]
    win = _WINDOWS[window_type](n)
    # reshape para broadcast sobre todos los ejes salvo el de frecuencia
    shape    = [1] * signal.ndim
    shape[1] = n
    return signal * win.reshape(shape)


def compute_range_profile(
    data: dict,
    tx: int,
    rx: int,
    antenna: int,
    window: bool = True,
    window_type: str = "hamming",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el Range Profile (dominio distancia) vía FFT del espectro.

    El range profile se obtiene aplicando FFT a lo largo del eje de frecuencia,
    transformando los datos desde el dominio frecuencial al dominio de retardo/distancia.
    Convenio de escala: range_per_bin = c / BW  (igual que en 00_prueba.ipynb).

    Notas de implementación
    -----------------------
    * Se usa ``np.fft.fft`` (no ifft) para mantener consistencia con el ejemplo
      de referencia. La ifft incluiría un factor de normalización 1/N que bajaría
      las amplitudes ~60 dB sin aportar información adicional.
    * Se devuelven todos los N bins porque la señal R es compleja (complex64);
      a diferencia de señales reales, no hay simetría que permita descartar la
      segunda mitad.
    * La escala del eje X es ``k * c / BW`` (sin factor 2), siguiendo el convenio
      del notebook de referencia.

    Parameters
    ----------
    data        : dict del dataset (salida de :func:`load_data`).
    tx, rx      : índices TX y RX.
    antenna     : índice de antena receptora (0–3).
    window      : si aplicar ventana espectral antes de la FFT.
    window_type : tipo de ventana ('hamming' | 'hann' | 'blackman').

    Returns
    -------
    ranges  : np.ndarray (N_f,)  — distancias en metros  [k * c / BW]
    rp_mean : np.ndarray (N_f,)  — amplitud media del range profile
    """
    signal = data["R"][:, :, tx, rx, antenna]   # (N_t, N_f)
    freq   = data["frequency"]

    if window:
        signal = apply_window(signal, window_type)

    # FFT a lo largo del eje de frecuencia → dominio de retardo/distancia
    rp_full = np.fft.fft(signal, axis=1)        # (N_t, N_f) — todos los bins

    bw            = freq[-1] - freq[0]          # ancho de banda total
    range_per_bin = C / bw                      # convenio: c / BW por bin
    ranges        = np.arange(rp_full.shape[1]) * range_per_bin
    rp_mean       = np.abs(rp_full).mean(axis=0)

    return ranges, rp_mean

# ──────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def plot_spectrum(
    data: dict,
    tx: int,
    rx: int,
    antenna: int,
    show_windowed: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Grafica el espectro medio en dB  (con y sin ventana Hamming).

    Returns
    -------
    plt.Axes con el gráfico.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    freq   = data["frequency"]
    signal = data["R"][:, :, tx, rx, antenna]
    label  = CONFIG_LABELS.get((tx, rx), f"TX{tx}-RX{rx}")
    f_ghz  = freq * 1e-9

    mean_db = 20 * np.log10(np.abs(signal).mean(axis=0) + 1e-12)
    ax.plot(f_ghz, mean_db, label=f"{label} | Ant {antenna}")

    if show_windowed:
        win_db = 20 * np.log10(np.abs(apply_window(signal)).mean(axis=0) + 1e-12)
        ax.plot(f_ghz, win_db, "--", alpha=0.75,
                label=f"{label} | Ant {antenna} (Hamming)")

    ax.set_xlabel("Frecuencia (GHz)")
    ax.set_ylabel("Amplitud (dB)")
    ax.set_title(title or f"Espectro medio — {label} | Antena {antenna}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_all_configs_spectrum(data: dict, antenna: int = 0) -> None:
    """
    Figura 2×2 con el espectro medio de las 4 configuraciones TX/RX.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Espectros medios — Antena RX {antenna}",
                 fontsize=13, fontweight="bold")
    for ax, (tx, rx) in zip(axes.flat, CONFIG_LABELS):
        plot_spectrum(data, tx, rx, antenna,
                      show_windowed=True, title=CONFIG_LABELS[(tx, rx)], ax=ax)
    plt.tight_layout()
    plt.show()


def plot_range_profile(
    data: dict,
    tx: int,
    rx: int,
    antenna: int,
    show_windowed: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Grafica el Range Profile medio en dB (sin ventana y, opcionalmente, con Hamming).

    Parameters
    ----------
    data          : dict del dataset.
    tx, rx        : índices TX y RX.
    antenna       : índice de antena receptora (0–3).
    show_windowed : si superponer también la versión con ventana Hamming.
    title         : título opcional.
    ax            : eje matplotlib existente (opcional).

    Returns
    -------
    plt.Axes con el gráfico.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    label = CONFIG_LABELS.get((tx, rx), f"TX{tx}-RX{rx}")

    # Curva sin ventana
    ranges, rp = compute_range_profile(data, tx, rx, antenna, window=False)
    rp_db = 20 * np.log10(rp + 1e-12)
    ax.plot(ranges, rp_db, label=f"{label} | Ant {antenna}")

    # Curva con ventana Hamming (opcional)
    if show_windowed:
        ranges_w, rp_w = compute_range_profile(data, tx, rx, antenna, window=True)
        rp_w_db = 20 * np.log10(rp_w + 1e-12)
        ax.plot(ranges_w, rp_w_db, "--", alpha=0.75,
                label=f"{label} | Ant {antenna} (Hamming)")
        ax.legend(fontsize=9)

    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Amplitud (dB)")
    ax.set_title(title or f"Range Profile — {label} | Ant {antenna}")
    ax.grid(True, alpha=0.3)
    return ax


def plot_all_configs_range_profile(data: dict, antenna: int = 0) -> None:
    """
    Figura 2×2 con los Range Profiles de las 4 configuraciones TX/RX.
    Cada subgráfica muestra la curva sin ventana y la versión Hamming.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Range Profiles — Antena RX {antenna}",
                 fontsize=13, fontweight="bold")
    for ax, (tx, rx) in zip(axes.flat, CONFIG_LABELS):
        plot_range_profile(data, tx, rx, antenna,
                           show_windowed=True, title=CONFIG_LABELS[(tx, rx)], ax=ax)
    plt.tight_layout()
    plt.show()


def plot_temporal_evolution(
    data: dict,
    tx: int,
    rx: int,
    antenna: int,
    freq_idx: int | None = None,
    title: str | None = None,
) -> None:
    """
    Evolución temporal de la potencia para una configuración y frecuencia.

    Parameters
    ----------
    freq_idx : si None, usa el índice de frecuencia con mayor potencia media.
    """
    signal = data["R"][:, :, tx, rx, antenna]
    freq   = data["frequency"]
    ts     = data["timestamps"]
    label  = CONFIG_LABELS.get((tx, rx), f"TX{tx}-RX{rx}")

    if freq_idx is None:
        freq_idx = int(np.argmax(np.abs(signal).mean(axis=0)))

    power_db   = 20 * np.log10(np.abs(signal[:, freq_idx]) + 1e-12)
    t_relative = ts - ts[0]

    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_relative, power_db)
    ax.set_xlabel("Tiempo relativo (s)")
    ax.set_ylabel("Potencia (dB)")
    ax.set_title(title or
                 f"Evolución temporal — {label} | Ant {antenna} | "
                 f"f = {freq[freq_idx]*1e-9:.4f} GHz")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_antenna_matrix(data: dict, freq_idx: int, timestamp_idx: int) -> None:
    """
    Heatmap de potencia (dB) por configuración TX/RX y antena receptora.

    Filas   : 4 configuraciones (TX×RX)
    Columnas: 4 antenas receptoras físicas
    """
    R       = data["R"]
    freq    = data["frequency"]
    configs = list(CONFIG_LABELS.keys())

    power = np.zeros((4, 4))
    for i, (tx, rx) in enumerate(configs):
        for ant in range(4):
            val          = np.abs(R[timestamp_idx, freq_idx, tx, rx, ant])
            power[i, ant] = 20 * np.log10(val + 1e-12)

    labels_cfg = [CONFIG_LABELS[c] for c in configs]

    _, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(power, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Potencia (dB)")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Ant {i}" for i in range(4)])
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels_cfg, fontsize=9)
    ax.set_title(
        f"Potencia por config y antena\n"
        f"t_idx={timestamp_idx} | f = {freq[freq_idx]*1e-9:.4f} GHz"
    )
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{power[i, j]:.0f}",
                    ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4. ESTADÍSTICAS
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary_stats(data: dict) -> pd.DataFrame:
    """
    Estadísticas por configuración TX/RX y antena receptora.

    Returns
    -------
    pd.DataFrame con columnas:
        config, tipo, antena, potencia_media_dB, desv_std_temporal_dB,
        dinamica_dB, freq_pico_GHz
    """
    R    = data["R"]
    freq = data["frequency"]
    rows = []

    for (tx, rx), label in CONFIG_LABELS.items():
        tipo = "Monostático" if tx == rx else "Bistático"
        for ant in range(4):
            signal     = np.abs(R[:, :, tx, rx, ant])          # (N_t, N_f)
            mean_spec  = signal.mean(axis=0)                    # (N_f,)
            power_t    = 20 * np.log10(signal.mean(axis=1) + 1e-12)
            mean_pwr   = float(10 * np.log10(mean_spec.mean() + 1e-12))
            std_t      = float(np.std(power_t))
            dynamic    = float(20 * np.log10(
                (mean_spec.max() + 1e-12) / (mean_spec.mean() + 1e-12)))
            freq_peak  = float(freq[np.argmax(mean_spec)] * 1e-9)
            rows.append({
                "config":               label,
                "tipo":                 tipo,
                "antena":               ant,
                "potencia_media_dB":    round(mean_pwr, 2),
                "desv_std_temporal_dB": round(std_t,    2),
                "dinamica_dB":          round(dynamic,  2),
                "freq_pico_GHz":        round(freq_peak, 4),
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 5. GUARDADO
# ──────────────────────────────────────────────────────────────────────────────

def save_processed(data: dict, output_path: str | Path) -> None:
    """
    Guarda el dataset (posiblemente filtrado) en un archivo .npz comprimido.

    Los arrays complejos se guardan separando parte real e imaginaria de R
    para compatibilidad máxima con numpy.

    Parameters
    ----------
    data        : dict con 'timestamps', 'frequency', 'R'.
    output_path : ruta de salida (ej. '../data/processed/data_full.npz').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        timestamps = data["timestamps"],
        frequency  = data["frequency"],
        R_real     = data["R"].real,
        R_imag     = data["R"].imag,
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"✓ Guardado CSI complejo en: {output_path}  ({size_mb:.1f} MB)")

def save_processed_power_pdp(data: dict, output_path: str | Path) -> None:
    """
    Guarda los datos de potencia (espectro) y PDP (Power Delay Profile) en un
    archivo .npz comprimido, para su uso en redes neuronales tradicionales.

    Parameters
    ----------
    data        : dict con 'timestamps', 'frequency', 'R'.
    output_path : ruta de salida (ej. '../data/processed/data_power_pdp.npz').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    R = data["R"]
    freq = data["frequency"]
    
    # 1. Potencia (espectro)
    power = np.abs(R)**2
    
    # 2. PDP (Power Delay Profile)
    # Aplicamos ventana de Hamming a lo largo del eje de frecuencia (axis 1)
    N_f = freq.shape[0]
    ham = np.hamming(N_f).astype(np.float32).reshape(1, N_f, 1, 1, 1)
    R_windowed = R * ham
    
    # Transformada al dominio del retardo
    cir = np.fft.fft(R_windowed, axis=1)
    pdp = np.abs(cir)**2
    
    np.savez_compressed(
        output_path,
        timestamps = data["timestamps"],
        frequency  = freq,
        power      = power.astype(np.float32),
        pdp        = pdp.astype(np.float32),
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"✓ Guardado Potencia y PDP en: {output_path}  ({size_mb:.1f} MB)")
