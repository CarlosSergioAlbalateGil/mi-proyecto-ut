"""
csi_tensor.py
=============
Utilidades para preparar el tensor CSI complejo procedente de las medidas FMCW
y dejarlo listo para el entrenamiento de Neural Operators (FNOs).

El tensor de entrada tiene forma:
    R  :  (N_t, N_f, N_tx, N_rx, N_ant)   dtype=complex64
con N_tx=2, N_rx=2, N_ant=4.

La fase se preserva en todo momento; nunca se trabaja solo con magnitud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
C = 3e8  # velocidad de la luz (m/s)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Carga desde .npz
# ──────────────────────────────────────────────────────────────────────────────

def load_csi_from_npz(path: str | Path) -> dict:
    """
    Carga un archivo ``.npz`` procesado (generado por :func:`eda_utils.save_processed`)
    y reconstruye el tensor complejo ``R``.

    Los archivos .npz guardan las partes real e imaginaria por separado
    (``R_real``, ``R_imag``) para compatibilidad máxima con numpy.

    Parameters
    ----------
    path : str or Path
        Ruta al archivo ``.npz``.

    Returns
    -------
    dict con claves:
        ``'timestamps'`` : np.ndarray  (N_t,)
        ``'frequency'``  : np.ndarray  (N_f,)  — Hz
        ``'R'``          : np.ndarray  complex64  (N_t, N_f, 2, 2, 4)
        ``'path'``       : Path
    """
    path = Path(path)
    data = np.load(path)

    R = (data["R_real"] + 1j * data["R_imag"]).astype(np.complex64)

    return {
        "timestamps": data["timestamps"],
        "frequency":  data["frequency"],
        "R":          R,
        "path":       path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Normalización (preserva fase)
# ──────────────────────────────────────────────────────────────────────────────

NormMode = Literal["maxabs", "per_subcarrier", "none"]


def normalize_csi(
    R: np.ndarray,
    mode: NormMode = "maxabs",
) -> Tuple[np.ndarray, dict]:
    """
    Normaliza el tensor CSI complejo preservando la información de fase.

    Modes
    -----
    ``'maxabs'``
        Divide por el máximo valor absoluto global del tensor.
        Rango resultante: ``|R_norm| ∈ [0, 1]``.
        Recomendado para redes CV (complex-valued): preserva la estructura
        de fase y evita escalar de forma diferente cada subportadora.

    ``'per_subcarrier'``
        Divide cada subportadora (eje ``N_f``) por su propio máximo absoluto.
        Útil cuando hay variación espectral muy grande, pero altera las
        relaciones de amplitud relativa entre subportadoras.

    ``'none'``
        Sin normalización; devuelve el tensor original sin cambios.

    Parameters
    ----------
    R    : np.ndarray complex64  (N_t, N_f, N_tx, N_rx, N_ant)
    mode : NormMode

    Returns
    -------
    R_norm : np.ndarray complex64  — tensor normalizado
    meta   : dict con estadísticas de normalización (para poder desnormalizar)
    """
    meta: dict = {"mode": mode}

    if mode == "maxabs":
        scale = np.abs(R).max()
        meta["scale"] = float(scale)
        R_norm = R / scale

    elif mode == "per_subcarrier":
        # shape para broadcasting: (1, N_f, 1, 1, 1)
        scale = np.abs(R).max(axis=(0, 2, 3, 4), keepdims=True)
        meta["scale"] = scale          # array (1, N_f, 1, 1, 1)
        R_norm = R / scale

    elif mode == "none":
        R_norm = R
        meta["scale"] = 1.0

    else:
        raise ValueError(f"mode '{mode}' no reconocido. Usa: 'maxabs', 'per_subcarrier', 'none'.")

    return R_norm.astype(np.complex64), meta


def denormalize_csi(R_norm: np.ndarray, meta: dict) -> np.ndarray:
    """
    Invierte la normalización aplicada por :func:`normalize_csi`.

    Parameters
    ----------
    R_norm : np.ndarray complex64
    meta   : dict devuelto por :func:`normalize_csi`

    Returns
    -------
    np.ndarray complex64 — tensor en escala original
    """
    return (R_norm * meta["scale"]).astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Split temporal train / val / test
# ──────────────────────────────────────────────────────────────────────────────

def split_train_val_test(
    R:          np.ndarray,
    timestamps: np.ndarray,
    ratios:     Sequence[float] = (0.70, 0.15, 0.15),
) -> dict:
    """
    Divide el tensor en splits **temporales contiguos** (sin mezcla aleatoria)
    para evitar data-leakage entre tramas temporalmente correladas.

    Parameters
    ----------
    R          : np.ndarray complex64  (N_t, N_f, 2, 2, 4)
    timestamps : np.ndarray  (N_t,)
    ratios     : (train_ratio, val_ratio, test_ratio).  Deben sumar 1.

    Returns
    -------
    dict con claves ``'train'``, ``'val'``, ``'test'``, cada una conteniendo:
        ``'R'``          : np.ndarray
        ``'timestamps'`` : np.ndarray
        ``'n_samples'``  : int
        ``'t_start'``    : float
        ``'t_end'``      : float
    """
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError(f"Los ratios deben sumar 1.0. Suma actual: {r_train+r_val+r_test:.4f}")

    N = R.shape[0]
    i_val  = int(N * r_train)
    i_test = int(N * (r_train + r_val))

    splits = {
        "train": (0,      i_val),
        "val":   (i_val,  i_test),
        "test":  (i_test, N),
    }

    result = {}
    for name, (s, e) in splits.items():
        R_split  = R[s:e]
        ts_split = timestamps[s:e]
        result[name] = {
            "R":          R_split,
            "timestamps": ts_split,
            "n_samples":  e - s,
            "t_start":    float(ts_split[0])  if len(ts_split) else float("nan"),
            "t_end":      float(ts_split[-1]) if len(ts_split) else float("nan"),
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. Representación Re/Im apilada (entrada alternativa para NOs reales)
# ──────────────────────────────────────────────────────────────────────────────

def to_real_imag_stack(R: np.ndarray) -> np.ndarray:
    """
    Convierte un tensor complejo a su representación real apilando
    parte real e imaginaria en el último eje.

    Input  : (..., complex64)
    Output : (..., 2)  con [..., 0] = Re  y  [..., 1] = Im

    Esta representación es un input válido para NOs que no soporten
    números complejos nativamente.

    Parameters
    ----------
    R : np.ndarray complex64

    Returns
    -------
    np.ndarray float32  (misma forma que R pero con eje extra de tamaño 2)
    """
    return np.stack([R.real, R.imag], axis=-1).astype(np.float32)


def from_real_imag_stack(R_ri: np.ndarray) -> np.ndarray:
    """
    Invierte :func:`to_real_imag_stack`:
    ``(..., 2) → (...) complex64``
    """
    return (R_ri[..., 0] + 1j * R_ri[..., 1]).astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Canal en dominio de retardo (CIR)
# ──────────────────────────────────────────────────────────────────────────────

def compute_cir(
    R:        np.ndarray,
    freq:     np.ndarray,
    n_taps:   int | None = None,
    window:   bool       = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la Channel Impulse Response (CIR) a partir del CFR complejo
    mediante IFFT a lo largo del eje de frecuencia.

    Eje de frecuencia: dim 1  (N_f).

    Parameters
    ----------
    R      : np.ndarray complex64  (N_t, N_f, N_tx, N_rx, N_ant)
    freq   : np.ndarray  (N_f,)   — frecuencias en Hz
    n_taps : número de taps de salida. Si ``None`` se usa ``N_f``.
    window : si ``True`` aplica ventana Hamming antes de la IFFT.

    Returns
    -------
    cir      : np.ndarray complex64  (N_t, n_taps, N_tx, N_rx, N_ant)
    tau_axis : np.ndarray  (n_taps,)  — eje de retardo en segundos
    """
    N_f   = R.shape[1]
    df    = float(freq[1] - freq[0])         # paso de frecuencia (Hz)
    BW    = float(freq[-1] - freq[0])         # ancho de banda (Hz)

    if n_taps is None:
        n_taps = N_f

    R_in = R.copy()
    if window:
        ham = np.hamming(N_f).astype(np.float32)
        # broadcast a lo largo de todos los ejes excepto N_f
        ham = ham.reshape(1, N_f, 1, 1, 1)
        R_in = R_in * ham

    # IFFT sobre el eje de frecuencia (dim 1)
    cir = np.fft.ifft(R_in, n=n_taps, axis=1).astype(np.complex64)

    # Eje de retardo: τ = k / BW  para k = 0, 1, …, n_taps-1
    tau_axis = np.arange(n_taps) / BW

    return cir, tau_axis


# ──────────────────────────────────────────────────────────────────────────────
# 6. Guardado de splits
# ──────────────────────────────────────────────────────────────────────────────

def save_csi_splits(
    splits:    dict,
    frequency: np.ndarray,
    norm_meta: dict,
    output_dir: str | Path,
    prefix:     str = "csi",
) -> None:
    """
    Guarda los splits train/val/test en archivos ``.npz`` dentro de
    ``output_dir``, incluyendo el tensor complejo y la representación Re/Im.

    Archivos generados (por split):
        ``<prefix>_train.npz``, ``<prefix>_val.npz``, ``<prefix>_test.npz``

    Cada archivo contiene:
        ``R_real``     — parte real   (N_split, N_f, 2, 2, 4) float32
        ``R_imag``     — parte imag   (N_split, N_f, 2, 2, 4) float32
        ``R_ri``       — pila Re/Im   (N_split, N_f, 2, 2, 4, 2) float32
        ``timestamps`` — (N_split,)
        ``frequency``  — (N_f,)
        ``norm_mode``  — string
        ``norm_scale`` — escalar o array (para desnormalizar)

    Parameters
    ----------
    splits     : salida de :func:`split_train_val_test`
    frequency  : np.ndarray (N_f,)
    norm_meta  : salida de :func:`normalize_csi`
    output_dir : directorio de salida (se crea si no existe)
    prefix     : prefijo para los nombres de archivo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, sp in splits.items():
        out_path = output_dir / f"{prefix}_{name}.npz"
        R = sp["R"]
        R_ri = to_real_imag_stack(R)

        # norm_scale puede ser un array; lo convertimos a float si es escalar
        scale = norm_meta["scale"]
        if isinstance(scale, np.ndarray):
            scale_save = scale
        else:
            scale_save = np.float32(scale)

        np.savez_compressed(
            out_path,
            R_real      = R.real.astype(np.float32),
            R_imag      = R.imag.astype(np.float32),
            R_ri        = R_ri,
            timestamps  = sp["timestamps"],
            frequency   = frequency,
            norm_mode   = norm_meta["mode"],
            norm_scale  = scale_save,
        )

        size_mb = out_path.stat().st_size / 1e6
        print(
            f"  ✓ {name:5s} → {out_path.name}"
            f"  shape={R.shape}  ({size_mb:.1f} MB)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Informe de shapes
# ──────────────────────────────────────────────────────────────────────────────

def print_csi_report(
    splits:    dict,
    frequency: np.ndarray,
    norm_meta: dict,
) -> None:
    """
    Imprime un resumen de los splits CSI con la información relevante
    para el diseño del FNO.

    Parameters
    ----------
    splits    : salida de :func:`split_train_val_test`
    frequency : np.ndarray (N_f,)
    norm_meta : salida de :func:`normalize_csi`
    """
    BW    = float(frequency[-1] - frequency[0])
    N_f   = len(frequency)
    df    = BW / (N_f - 1)
    total = sum(s["n_samples"] for s in splits.values())

    print("=" * 62)
    print("  INFORME DE SHAPES — TENSOR CSI")
    print("=" * 62)
    print(f"  Frecuencia mín / máx : {frequency[0]*1e-9:.4f} / {frequency[-1]*1e-9:.4f} GHz")
    print(f"  Subportadoras (N_f)  : {N_f}")
    print(f"  BW                   : {BW*1e-6:.2f} MHz")
    print(f"  df                   : {df*1e-3:.3f} kHz")
    print(f"  Normalización        : {norm_meta['mode']}")
    print()
    print(f"  {'Split':<8} {'N_samples':>10} {'%':>6}  {'t_start':>10}  {'t_end':>10}  Shape")
    print(f"  {'-'*8} {'-'*10} {'-'*6}  {'-'*10}  {'-'*10}  -----")
    for name, sp in splits.items():
        n   = sp["n_samples"]
        pct = 100 * n / total if total else 0
        shp = str(sp["R"].shape)
        print(
            f"  {name:<8} {n:>10d} {pct:>5.1f}%"
            f"  {sp['t_start']:>10.2f}  {sp['t_end']:>10.2f}  {shp}"
        )
    print()
    print("  Tensor input para FNO:")
    R_ex = next(iter(splits.values()))["R"]
    ri_ex = to_real_imag_stack(R_ex)
    print(f"    Complejo    : {R_ex.shape}  dtype={R_ex.dtype}")
    print(f"    Re/Im stack : {ri_ex.shape}  dtype={ri_ex.dtype}")
    print("=" * 62)
