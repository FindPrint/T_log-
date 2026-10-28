# -*- coding: utf-8 -*-
"""
UNIVERSAL_LOG v3.5.2-H1 — Pipeline autonome + Monte Carlo Harmonique GUIDÉ
Dérivé de Tlog v3.5-H1 (fichier original intact, jamais modifié).

v3.5.2 : tests de bruit (étape 3, additif), analyse CSD et diagnostic
de normalisation généralisés aux 29 estimateurs de complexité.
Références bruit blanc empiriques par estimateur et par longueur
(graine 1234, R=8, mise en cache). Score variance_proxy_score
historique inchangé (Hurst seul, comparabilité conservée).
Spectre T_log toujours absent (travail futur).

Scan, Monte Carlo, stabilité bootstrap, définitions des Q, méthode
F1/F2 : inchangés par rapport à v3.5-H1.
"""

# =============================================================================
# CONFIGURATION UTILISATEUR (Modifier ces paramètres selon vos besoins)
# =============================================================================

# =============================================================================
# MODE D'ANALYSE
# =============================================================================
# 'fast'   : Analyse rapide, paramètres fixes, pas de bootstrap
#            - w_size: 80, test: cohen_d, n_est: 1, boot: False
# 'robust' : Analyse robuste, paramètres fixes, bootstrap activé
#            - w_size: 150, test: cliff_delta, n_est: 5, boot: True
# 'stable' : Analyse stable, paramètres modifiables ci-dessous, bootstrap activé
#            - Utilise les paramètres personnalisés définis ci-dessous
ANALYSIS_MODE = 'stable'  # Options: 'fast', 'robust', 'stable'

# =============================================================================
# PARAMÈTRES PERSONNALISABLES (Mode 'stable' uniquement)
# =============================================================================
# Ces paramètres sont utilisés uniquement en mode 'stable'.
# En modes 'fast' et 'robust', ils sont ignorés au profit des valeurs prédéfinies.

# Test statistique
STATISTICAL_TEST = 'cohen_d'  # Options: 'cohen_d', 'cliff_delta'

# Paramètres de fenêtrage
WINDOW_SIZE = 80              # Taille de la fenêtre glissante
MIN_GROUP_SIZE = 4            # Taille minimale des groupes pour tests statistiques
MIN_SERIES_LENGTH = 80       # Longueur minimale des séries pour analyse

# Paramètres de bootstrap
N_ESTIMATORS = 4              # Nombre d'estimateurs pour bootstrap (mode stable)
ENABLE_BOOTSTRAP = True      # Activer/désactiver le bootstrap

# Paramètres de triangulation
FULL_TRIANGULATION = False   # À activer seulement avec réplications réelles ou en analyse de sensibilité

# Reproductibilité : une même graine donne les mêmes tirages Monte Carlo.
# Elle peut être remplacée dans ``opt_cfg`` lors d'un appel programmatique.
RANDOM_SEED = 42

# =============================================================================
# MONTE CARLO HARMONIQUE GUIDÉ (v1)
# =============================================================================
# Probabilité qu'un tirage utilise une règle dérivée d'une constante
# mathématique (φ, π, e, √2, √3). Le complément reste uniforme classique.
HARMONIC_PROB = 0.45

# Constantes mathématiques utilisées pour guider les tirages
PHI   = (1 + 5 ** 0.5) / 2   # règle d'or ≈ 1.6180339887
PI    = 3.141592653589793
E     = 2.718281828459045
SQRT2 = 1.4142135623730951
SQRT3 = 1.7320508075688772

# =============================================================================
# COLONNE DE CONTRÔLE (CSV)
# =============================================================================
# Nom de la colonne à utiliser comme axe de progression/contrôle
# Si None, la première colonne numérique sera utilisée automatiquement
DEFAULT_CONTROL_COLUMN = 'Energy_Diff'  # Exemple: 'Progression', 'Time', 'Step', etc.

# =============================================================================
# CHEMINS PAR DÉFAUT
# =============================================================================
DEFAULT_INPUT_CSV = None      # Auto-détection du premier CSV dans le dossier
DEFAULT_OUTPUT_DIR = './results_v3_4_harmonic_v3'

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from scipy import stats
except ImportError as exc:
    print(
        "Dépendance manquante : scipy. Installez les dépendances du pipeline "
        "(par exemple : python -m pip install scipy numpy pandas matplotlib seaborn).",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc
import logging
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour les graphiques
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import seaborn as sns

# Tentative d'import de Numba pour optimisation SampleEn
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: prange vide si numba non disponible
    def prange(x):
        return range(x)

# Import multiprocessing pour parallélisation
import multiprocessing as mp
from functools import partial

# Configuration matplotlib pour des graphiques de qualité
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

# Conserver les avertissements visibles : ils signalent souvent une estimation
# instable, une série trop courte ou une donnée insuffisante.
warnings.filterwarnings('default')

# =============================================================================
# ESTIMATEURS DE COMPLEXITÉ — Versions standalone exportables
# =============================================================================
antropy = None
nolds = None

def get_katz_fd(s):
    s = np.asarray(s).flatten()
    n = len(s)
    if n < 3:
        return np.nan
    diffs = np.diff(s)
    l_sum = np.sum(np.sqrt(1.0 + diffs**2))
    d_max = np.max(np.sqrt(np.arange(n)**2 + (s - s[0])**2))
    if l_sum < 1e-12 or d_max < 1e-12:
        return 1.0
    return np.log10(n - 1) / (np.log10(d_max) - np.log10(l_sum) + np.log10(n - 1))

def higuchi_fd(s, k_max=10):
    """Dimension fractale de Higuchi (version standalone)."""
    s = np.asarray(s).flatten()
    n = len(s)
    k_max = min(k_max, n // 4)
    lk, k_vals = [], []
    for k in range(1, k_max + 1):
        lengths = []
        for m in range(k):
            n_elements = (n - 1 - m) // k
            if n_elements < 2:
                continue
            idxA = m + np.arange(1, n_elements + 1) * k
            idxB = m + np.arange(n_elements) * k
            sum_diffs = np.sum(np.abs(s[idxA] - s[idxB]))
            norm_factor = (n - 1) / (n_elements * k)
            lengths.append(sum_diffs * norm_factor / k)
        if lengths:
            lk.append(np.mean(lengths))
            k_vals.append(k)
    if len(lk) >= 3:
        slope, *_ = stats.linregress(np.log(1.0 / np.array(k_vals)), np.log(lk))
        return float(slope) if abs(slope) < 5.0 else np.nan
    return np.nan

def dfa_exponent(s):
    """DFA: Detrended Fluctuation Analysis (version standalone)."""
    s = np.asarray(s).flatten()
    n = len(s)
    y = np.cumsum(s - np.mean(s))
    scales = np.unique(np.logspace(np.log10(5), np.log10(n // 4), num=10).astype(int))
    fluctuations, scales_used = [], []
    for sc in scales:
        segments = n // sc
        if segments < 1:
            continue
        y_trunc = y[:segments * sc]
        seg_matrix = y_trunc.reshape(segments, sc)
        x = np.arange(sc)
        x_mean = (sc - 1) / 2.0
        x_var = np.var(x)
        if x_var < 1e-12:
            continue
        y_m = np.mean(seg_matrix, axis=1, keepdims=True)
        cov = np.mean((seg_matrix - y_m) * (x - x_mean), axis=1)
        slope = cov / x_var
        intercept = y_m.squeeze() - slope * x_mean
        trend = slope[:, None] * x + intercept[:, None]
        rms = np.mean((seg_matrix - trend) ** 2, axis=1)
        fluctuations.append(np.sqrt(np.mean(rms)))
        scales_used.append(sc)
    if len(scales_used) >= 3:
        slope, *_ = stats.linregress(np.log(scales_used), np.log(fluctuations))
        return float(2.0 - slope)
    return np.nan

def hurst_rs(s):
    """Exposant de Hurst par méthode R/S (version standalone)."""
    s = np.asarray(s).flatten()
    n = len(s)
    max_l = n // 2
    if max_l < 8:
        return np.nan
    lengths = np.unique(np.logspace(np.log10(8), np.log10(max_l), num=10).astype(int))
    vals, lengths_used = [], []
    for l in lengths:
        chunks = n // l
        if chunks < 1:
            continue
        seg = s[:chunks * l].reshape(chunks, l)
        means = np.mean(seg, axis=1, keepdims=True)
        stds = np.std(seg, axis=1, ddof=1)
        valid = stds > 1e-10
        if not np.any(valid):
            continue
        seg_v = seg[valid]
        means_v = means[valid]
        stds_v = stds[valid]
        y = np.cumsum(seg_v - means_v, axis=1)
        y0 = np.hstack([np.zeros((len(y), 1)), y])
        r = np.max(y0, axis=1) - np.min(y0, axis=1)
        rs = r / stds_v
        if len(rs) > 0:
            vals.append(np.mean(rs))
            lengths_used.append(l)
    if len(lengths_used) >= 3:
        slope, *_ = stats.linregress(np.log(lengths_used), np.log(vals))
        return float(slope)
    return np.nan

def _count_matches_numba(x, n, r, m_len):
    """Version Numba-optimisée pour compter les correspondances de templates."""
    count = 0
    for i in range(n - m_len):
        for j in range(i + 1, n - m_len):
            match = True
            for k in range(m_len):
                if abs(x[i + k] - x[j + k]) >= r:
                    match = False
                    break
            if match:
                count += 1
    return count

# Compiler la fonction Numba si disponible
if NUMBA_AVAILABLE:
    _count_matches_numba = jit(nopython=True)(_count_matches_numba)

def sample_entropy_individual(s, m=2):
    """Sample Entropy (version standalone avec optimisation Numba si disponible)."""
    x = np.asarray(s).flatten()[:min(400, len(s))]
    n = len(x)
    r = 0.2 * np.std(x)
    if r < 1e-12 or n < m + 2:
        return np.nan
    
    try:
        if NUMBA_AVAILABLE:
            # Version optimisée Numba
            cm = _count_matches_numba(x, n, r, m)
            cm1 = _count_matches_numba(x, n, r, m + 1)
        else:
            # Version fallback vectorisée
            def count_matches(m_len):
                count = 0
                templates = np.array([x[i:i+m_len] for i in range(n - m_len + 1)])
                for i in range(len(templates)):
                    diffs = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
                    count += np.sum(diffs < r)
                return count
            cm = count_matches(m)
            cm1 = count_matches(m + 1)
        
        if cm > 0 and cm1 > 0:
            return float(-np.log(cm1 / cm))
    except Exception:
        pass
    return np.nan

def compute_complexity_estimators(series, min_len=50):
    """Calcule les 5 estimateurs de complexité sur une série unique.
    
    Centralise le calcul des estimateurs pour éviter la duplication.
    Utilise les fonctions standalone définies ci-dessus.
    """
    s = np.asarray(series).flatten()
    if len(s) < min_len:
        return {k: np.nan for k in ['Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn']}
    
    nan_mask = np.isnan(s)
    if np.all(nan_mask):
        return {k: 0.0 for k in ['Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn']}
    if np.any(nan_mask):
        s = np.where(nan_mask, np.nanmean(s), s)
    
    if np.std(s) < 1e-10:
        return {k: 0.0 for k in ['Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn']}

    # Z-score normalization
    s = (s - np.mean(s)) / np.std(s)

    return {
        'Higuchi':  higuchi_fd(s),
        'DFA':      dfa_exponent(s),
        'Katz':     get_katz_fd(s),
        'Hurst':    hurst_rs(s),
        'SampleEn': sample_entropy_individual(s),
    }

ESTIMATOR_NAMES = ['Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn', 'Petrosian', 'Sevcik', 'BoxCounting', 'PermutationEntropy', 'CorrelationD2', 'GrassbergerProcaccia', 'InformationD1', 'VisibilityGraph', 'TwoNN', 'DANCo', 'Fisher', 'LevinaBickel', 'PCAIntrinsic', 'LocalPCA', 'GraphDimension', 'PersistentHomology', 'MFA', 'KolmogorovLZ', 'SpectralBeta', 'SpectralDimension', 'Hurst2mH', 'Lyapunov', 'MultifractalDeltaAlpha', 'SingularityFmax']
EXTRA_ESTIMATOR_NAMES = [n for n in ESTIMATOR_NAMES if n not in ('Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn')]


def apply_complexity(series, flux_name='', mode_cfg=None):
    """Calcule les 5 estimateurs de complexité sur une série unique.
    
    Préfère les bibliothèques `antropy`/`nolds` si disponibles, sinon
    utilise les implémentations natives (higuchi_fd, dfa_exponent, etc.).
    """
    s = np.asarray(series).flatten()
    min_len = mode_cfg.get('min_len', 50) if mode_cfg else 50
    
    if len(s) < min_len:
        return {k: np.nan for k in ESTIMATOR_NAMES}
    
    # Gestion des NaN
    nan_mask = np.isnan(s)
    if np.all(nan_mask):
        return {k: 0.0 for k in ESTIMATOR_NAMES}
    elif np.any(nan_mask):
        s = np.where(nan_mask, np.nanmean(s), s)
    
    # Preprocessing: diff if discrete or highly seasonal
    if flux_name in ('Halogenation_Signal', 'Lethality_Signal', 'Vaccination_Signal', 'Protection_Loss', 'Log_Returns', 'Momentum', 'Solar_Radiation', 'Evapotranspiration'):
        if np.std(s) > 1e-10:
            s = np.diff(s)
            
    if np.std(s) < 1e-10:
        return {k: 0.0 for k in ESTIMATOR_NAMES}

    # Z-score normalization
    if np.std(s) > 1e-10:
        s = (s - np.mean(s)) / np.std(s)
        
    est = {}
    
    # Higuchi Fractal Dimension
    if antropy is not None:
        try:
            val = float(antropy.higuchi_fd(s))
            est['Higuchi'] = val if not np.isnan(val) else 1.5
        except Exception:
            est['Higuchi'] = higuchi_fd(s, k_max=10) if not np.isnan(higuchi_fd(s, k_max=10)) else 1.5
    else:
        val = higuchi_fd(s, k_max=max(2, min(10, len(s) // 3)))
        est['Higuchi'] = val if not np.isnan(val) else 1.0

    # Detrended Fluctuation Analysis (DFA)
    if nolds is not None:
        try:
            val = float(2.0 - nolds.dfa(s))
            est['DFA'] = val if not np.isnan(val) else 1.0
        except Exception:
            est['DFA'] = dfa_exponent(s) if not np.isnan(dfa_exponent(s)) else 1.0
    else:
        val = dfa_exponent(s)
        est['DFA'] = val if not np.isnan(val) else 1.0

    # Katz Fractal Dimension
    val = get_katz_fd(s)
    est['Katz'] = val if not np.isnan(val) else 1.2

    # Hurst Exponent (R/S)
    if nolds is not None:
        try:
            val = float(nolds.hurst_rs(s))
            est['Hurst'] = val if not np.isnan(val) else 0.5
        except Exception:
            est['Hurst'] = hurst_rs(s) if not np.isnan(hurst_rs(s)) else 0.5
    else:
        val = hurst_rs(s)
        est['Hurst'] = val if not np.isnan(val) else 0.5

    # Sample Entropy (SampleEn)
    if antropy is not None:
        try:
            val = float(antropy.sample_entropy(s[:min(1000, len(s))]))
            est['SampleEn'] = val if not np.isnan(val) else 0.5
        except Exception:
            est['SampleEn'] = sample_entropy_individual(s) if not np.isnan(sample_entropy_individual(s)) else 0.5
    else:
        val = sample_entropy_individual(s)
        est['SampleEn'] = val if not np.isnan(val) else 0.5
            
    for _k, _v in _compute_extra_estimators(s).items():
        est[_k] = _v
    return est


def _to_float_or_nan(value):
    """Convertit une valeur d'estimateur en flottant fini, sinon NaN."""
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _compute_extra_estimators(z):
    """Les 24 estimateurs au-delà des 5 historiques, par fonctions
    standalone existantes, sur la série déjà traitée (même série que
    les 5). Échec -> NaN, jamais d'exception. (v3.5-H1 : pas de
    couche spectre, pas de formule T_log — les estimateurs sont
    les colonnes, les Q et les F1/F2.)"""
    out = {}
    out['Petrosian'] = _to_float_or_nan(_safe_call(petrosian_fd, z))
    out['Sevcik'] = _to_float_or_nan(_safe_call(sevcik_fd, z))
    out['BoxCounting'] = _to_float_or_nan(_safe_call(box_count_fd, z))
    out['PermutationEntropy'] = _to_float_or_nan(_safe_call(permutation_entropy, z))
    _beta = _to_float_or_nan(_safe_call(spectral_beta, z))
    out['SpectralBeta'] = _beta
    out['SpectralDimension'] = (5.0 - _beta) / 2.0 if np.isfinite(_beta) else np.nan
    out['CorrelationD2'] = _to_float_or_nan(_safe_call(correlation_dimension_d2, z))
    out['GrassbergerProcaccia'] = _to_float_or_nan(_safe_call(grassberger_procaccia, z))
    out['InformationD1'] = _to_float_or_nan(_safe_call(information_dimension_d1, z))
    out['VisibilityGraph'] = _to_float_or_nan(_safe_call(visibility_graph_dimension, z))
    out['TwoNN'] = _to_float_or_nan(_safe_call(two_nn_dimension, z))
    out['DANCo'] = _to_float_or_nan(_safe_call(danco_dimension, z))
    out['Fisher'] = _to_float_or_nan(_safe_call(fisher_dimension, z))
    out['LevinaBickel'] = _to_float_or_nan(_safe_call(levina_bickel, z))
    _part = _safe_call(pca_participation, z)
    if isinstance(_part, tuple) and len(_part) >= 1:
        out['PCAIntrinsic'] = _to_float_or_nan(_part[0])
    else:
        out['PCAIntrinsic'] = _to_float_or_nan(_part)
    out['LocalPCA'] = _to_float_or_nan(_safe_call(local_pca_dimension, z))
    out['GraphDimension'] = _to_float_or_nan(_safe_call(graph_cluster_dimension, z))
    out['PersistentHomology'] = _to_float_or_nan(_safe_call(persistent_homology_dimension, z))
    out['MFA'] = _to_float_or_nan(_safe_call(mfa_dimension, z))
    out['KolmogorovLZ'] = _to_float_or_nan(_safe_call(kolmogorov_lz_dimension, z))
    _h = _to_float_or_nan(_safe_call(hurst_rs, z))
    out['Hurst2mH'] = 2.0 - _h if np.isfinite(_h) else np.nan
    out['Lyapunov'] = _to_float_or_nan(_safe_call(lyapunov_rosenstein, z))
    try:
        _delta_alpha, _alphas, _fs, _fmax = mfdfa_spectrum(z)
    except Exception:
        _delta_alpha, _fmax = np.nan, np.nan
    out['MultifractalDeltaAlpha'] = _to_float_or_nan(_delta_alpha)
    out['SingularityFmax'] = _to_float_or_nan(_fmax)
    assert set(out) == set(EXTRA_ESTIMATOR_NAMES), set(out) ^ set(EXTRA_ESTIMATOR_NAMES)
    return out


def loupe_candidate_bounds(n_windows):
    """Bornes du nombre de seuils candidats.

    1001 n'est pas une résolution scientifique : au-delà du nombre de
    fenêtres, plusieurs candidats produisent exactement la même coupure.
    801 est un plafond de calcul, pas la valeur retenue.
    """
    hi = int(max(12, min(int(max(n_windows, 1)), 801)))
    lo = int(min(21, max(8, hi // 8)))
    if lo >= hi:
        lo = max(5, hi - 1)
    return lo, hi


def _zscore_series(s):
    s = np.asarray(s, dtype=float).flatten()
    s = s[np.isfinite(s)]
    if len(s) < 8:
        return s
    sd = np.std(s, ddof=1)
    if sd < 1e-12:
        return np.zeros_like(s)
    return (s - np.mean(s)) / sd


def petrosian_fd(s):
    s = np.asarray(s, dtype=float).flatten()
    n = len(s)
    if n < 4:
        return np.nan
    diff = np.diff(s)
    signs = np.sign(diff)
    signs[signs == 0] = 0
    nz = signs[signs != 0]
    nd = int(np.sum(nz[1:] * nz[:-1] < 0)) if len(nz) > 1 else 0
    if nd == 0:
        return 1.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nd))))


def sevcik_fd(s):
    s = np.asarray(s, dtype=float).flatten()
    n = len(s)
    if n < 4:
        return np.nan
    span = np.ptp(s)
    if span < 1e-12:
        return 1.0
    y = (s - np.min(s)) / span
    dx = 1.0 / (n - 1)
    dy = np.diff(y)
    length = float(np.sum(np.hypot(dx, dy)))
    if length <= 1.0:
        return 1.0
    return float(1.0 + np.log(length) / np.log(2.0 * (n - 1)))


def box_count_fd(s):
    s = np.asarray(s, dtype=float).flatten()
    n = len(s)
    if n < 16:
        return np.nan
    span = np.ptp(s)
    if span < 1e-12:
        return 1.0
    y = (s - np.min(s)) / span
    xs, ys = [], []
    for g in (4, 6, 8, 12, 16, 24):
        if g >= n:
            continue
        ix = np.minimum(g - 1, np.floor(np.linspace(0, 1, n) * g).astype(int))
        iy = np.minimum(g - 1, np.floor(y * g).astype(int))
        occupied = len(set(zip(ix.tolist(), iy.tolist())))
        xs.append(np.log(g))
        ys.append(np.log(max(occupied, 1)))
    if len(xs) < 3:
        return np.nan
    slope, *_ = stats.linregress(xs, ys)
    return float(slope)


def permutation_entropy(s, m=3, delay=1):
    s = np.asarray(s, dtype=float).flatten()
    n = len(s) - (m - 1) * delay
    if n < 12:
        return np.nan
    patterns = {}
    for i in range(n):
        window = s[i:i + m * delay:delay]
        order = tuple(np.argsort(window, kind='stable'))
        patterns[order] = patterns.get(order, 0) + 1
    h = 0.0
    for count in patterns.values():
        p = count / n
        h -= p * np.log(p)
    fact = 1
    for i in range(2, m + 1):
        fact *= i
    return float(h / np.log(fact))


def spectral_beta(s):
    s = np.asarray(s, dtype=float).flatten()
    n = len(s)
    if n < 32:
        return np.nan
    spec = np.abs(np.fft.rfft(s - np.mean(s))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = freqs > 0
    if np.sum(mask) < 4:
        return np.nan
    slope, _, r_value, _, _ = stats.linregress(np.log10(freqs[mask]), np.log10(spec[mask] + 1e-18))
    if not np.isfinite(slope):
        return np.nan
    if r_value ** 2 < 0.15 and abs(slope) > 0.35:
        return np.nan
    return float(-slope)


def _embedding_tau(s):
    s = np.asarray(s, dtype=float)
    n = len(s)
    sd = np.std(s, ddof=1)
    if sd < 1e-12:
        return 1
    centered = (s - np.mean(s)) / sd
    max_lag = min(12, n // 8)
    prev = 1.0
    for lag in range(1, max_lag + 1):
        corr = float(np.mean(centered[:-lag] * centered[lag:]))
        if prev > 0 and corr <= 1.0 / np.e:
            return lag
        if corr < 0:
            return max(1, lag)
        prev = corr
    return 1


def _delay_embed(s, m, tau, cap=110):
    n_vec = len(s) - (m - 1) * tau
    if n_vec < 12:
        return np.empty((0, m)), np.empty(0, dtype=int)
    stride = max(1, int(np.ceil(n_vec / cap)))
    idx = np.arange(0, n_vec, stride)
    mat = np.column_stack([s[idx + k * tau] for k in range(m)])
    return mat, idx


def _cloud_theiler(times, tau, m=1):
    """Theiler au moins égal au pas du nuage et au recouvrement du plongement.

    Sans cela, deux points voisins dans le temps restent les plus proches
    voisins géométriques et le rapport d2/d1 tombe vers 1.
    """
    tau = max(int(tau) if np.isfinite(tau) else 1, 1)
    overlap = max(tau, (max(int(m), 1) - 1) * tau)
    if times is None or len(times) < 3:
        return overlap
    step = int(np.median(np.diff(np.asarray(times, dtype=float))))
    return max(overlap, max(step, 1))


def _knn_distances(mat, times, k, theiler):
    diff = mat[:, None, :] - mat[None, :, :]
    dist = np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=2)))
    np.fill_diagonal(dist, np.inf)
    if theiler > 0:
        too_close = np.abs(times[:, None] - times[None, :]) <= theiler
        dist[too_close] = np.inf
    dist.sort(axis=1)
    return dist[:, :k]


def two_nn_dimension(s):
    z = _zscore_series(s)
    if len(z) < 30:
        return np.nan
    tau = _embedding_tau(z)
    m = int(min(8, max(3, len(z) // 30)))
    mat, times = _delay_embed(z, m, tau, 110)
    if len(mat) < 20:
        return np.nan
    dists = _knn_distances(mat, times, 2, _cloud_theiler(times, tau))
    d1 = dists[:, 0]
    d2 = dists[:, 1]
    ok = np.isfinite(d1) & np.isfinite(d2) & (d1 > 1e-12) & (d2 > d1)
    if int(np.sum(ok)) < 12:
        return np.nan
    d = float((np.sum(ok) - 1) / np.sum(np.log(d2[ok] / d1[ok])))
    median_ratio = float(np.median(d2[ok] / d1[ok]))
    # d2≈d1 : les deux voisins sont équidistants (courbe régulière), pas un
    # nuage isotrope. Au-dessus du plongement, l'estimateur n'est plus identifiable.
    if not np.isfinite(d) or d <= 0 or median_ratio < 1.02 or d > m:
        return np.nan
    return d


def levina_bickel(s, k=8):
    z = _zscore_series(s)
    if len(z) < 40:
        return np.nan
    tau = _embedding_tau(z)
    m = int(min(10, max(4, len(z) // 24)))
    mat, times = _delay_embed(z, m, tau, 100)
    if len(mat) < k + 8:
        return np.nan
    dists = _knn_distances(mat, times, k, _cloud_theiler(times, tau))
    locals_ = []
    for row in dists:
        if not np.isfinite(row[-1]) or row[0] <= 1e-12:
            continue
        acc = 0.0
        used = 0
        for j in range(k - 1):
            if row[j] > 1e-12 and np.isfinite(row[j]):
                acc += np.log(row[-1] / row[j])
                used += 1
        if used > 0 and acc > 0:
            locals_.append(used / acc)
    if len(locals_) < 12:
        return np.nan
    d = float(np.mean(locals_))
    if not np.isfinite(d) or d <= 0 or d > 2.0 * m:
        return np.nan
    return d


def pca_participation(s, emb=8):
    z = _zscore_series(s)
    tau = max(1, min(_embedding_tau(z), 3))
    mat, _ = _delay_embed(z, emb, tau, 160)
    if len(mat) < emb + 2:
        return np.nan, np.nan
    cov = np.cov(mat, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    pos = np.sort(eig[eig > 1e-10])[::-1]
    if len(pos) == 0:
        return np.nan, np.nan
    total = float(np.sum(pos))
    participation = float(total ** 2 / np.sum(pos ** 2))
    cum = np.cumsum(pos) / total
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    return participation, n95


def lyapunov_rosenstein(s):
    z = _zscore_series(s)
    tau = _embedding_tau(z)
    mat, times = _delay_embed(z, 3, tau, 90)
    n = len(mat)
    if n < 24:
        return np.nan
    dist = np.sqrt(np.maximum(0.0, np.sum((mat[:, None, :] - mat[None, :, :]) ** 2, axis=2)))
    np.fill_diagonal(dist, np.inf)
    theiler = _cloud_theiler(times, tau)
    dist[np.abs(times[:, None] - times[None, :]) <= theiler] = np.inf
    nn = np.argmin(dist, axis=1)
    horizon = min(8, n // 6)
    if horizon < 3:
        return np.nan
    curves = []
    for i, j in enumerate(nn):
        if not np.isfinite(dist[i, j]):
            continue
        if i + horizon < n and j + horizon < n:
            curves.append(np.linalg.norm(mat[i:i + horizon] - mat[j:j + horizon], axis=1))
    if len(curves) < 8:
        return np.nan
    stack = np.vstack(curves)
    ys = np.mean(np.log(stack + 1e-9), axis=0)
    slope, _, r_value, _, _ = stats.linregress(np.arange(horizon), ys)
    if r_value ** 2 < 0.3 or not np.isfinite(slope):
        return np.nan
    return float(slope)


def _safe_call(fn, *args):
    """Appelle un estimateur sans laisser une exception vider le spectre."""
    try:
        value = fn(*args)
    except Exception:
        return np.nan
    if isinstance(value, tuple):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _embed_cloud(series, m=None, cap=100, m_min=3, m_max=8):
    z = _zscore_series(series)
    if len(z) < 20:
        return z, np.empty((0, 1)), np.empty(0, dtype=int), 1
    tau = _embedding_tau(z)
    if m is None:
        m = int(min(m_max, max(m_min, len(z) // 30)))
    mat, times = _delay_embed(z, int(m), tau, cap)
    return z, mat, times, tau


def _pair_distances(mat, times=None, theiler=0):
    diff = mat[:, None, :] - mat[None, :, :]
    dist = np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=2)))
    np.fill_diagonal(dist, np.inf)
    if theiler and times is not None:
        dist[np.abs(times[:, None] - times[None, :]) <= theiler] = np.inf
    return dist


def _scaling_slope(log_x, log_y, min_r2=0.3, positive=True):
    if len(log_x) < 3:
        return np.nan
    slope, _, r_value, _, _ = stats.linregress(log_x, log_y)
    if not np.isfinite(slope) or r_value ** 2 < min_r2:
        return np.nan
    if positive and slope <= 0:
        return np.nan
    return float(slope)


def correlation_dimension_d2(s, theiler_on=False):
    """Dimension de corrélation D2. Avec fenêtre de Theiler : Grassberger-Procaccia."""
    z, mat, times, tau = _embed_cloud(s, cap=90, m_max=6)
    n = len(mat)
    if n < 25:
        return np.nan
    theiler = _cloud_theiler(times, tau) if theiler_on else 0
    dist = _pair_distances(mat, times, theiler)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = d[np.isfinite(d) & (d > 1e-12)]
    if len(d) < 30:
        return np.nan
    qs = np.unique(np.quantile(d, [0.08, 0.14, 0.20, 0.28, 0.36, 0.46]))
    log_r, log_c = [], []
    for r in qs:
        c = float(np.mean(d < r))
        if c > 1e-6 and r > 0:
            log_r.append(np.log(r))
            log_c.append(np.log(c))
    slope = _scaling_slope(log_r, log_c, min_r2=0.35)
    if not np.isfinite(slope):
        return np.nan
    return float(min(max(slope, 0.0), mat.shape[1]))


def grassberger_procaccia(s):
    """D2 avec correction de Theiler : les voisins temporels ne comptent pas."""
    return correlation_dimension_d2(s, theiler_on=True)


def information_dimension_d1(s):
    """Dimension d'information D1 : pente de l'entropie de Shannon contre log(1/ε)."""
    _z, mat, _times, _tau = _embed_cloud(s, cap=120, m_max=3, m_min=2)
    if len(mat) < 25:
        return np.nan
    span = np.ptp(mat, axis=0)
    span[span < 1e-12] = 1.0
    normed = (mat - np.min(mat, axis=0)) / span
    xs, ys = [], []
    for g in (3, 4, 5, 6, 8):
        idx = np.minimum(g - 1, np.floor(np.clip(normed, 0.0, 1.0) * g).astype(int))
        keys = {}
        for row in idx:
            key = tuple(row.tolist())
            keys[key] = keys.get(key, 0) + 1
        probs = np.asarray(list(keys.values()), dtype=float)
        probs /= probs.sum()
        info = float(-np.sum(probs * np.log(probs)))
        xs.append(np.log(g))
        ys.append(info)
    slope = _scaling_slope(xs, ys, min_r2=0.25)
    if not np.isfinite(slope):
        return np.nan
    return float(min(slope, mat.shape[1]))


def visibility_graph_dimension(s):
    """Dimension via le graphe de visibilité naturelle (Lacasa : α = 3 - 2H, d = 2 - H)."""
    z = _zscore_series(s)
    n = len(z)
    if n < 40:
        return np.nan
    if n > 180:
        z = z[np.linspace(0, n - 1, 180).astype(int)]
        n = len(z)
    degree = np.zeros(n, dtype=int)
    for i in range(n - 1):
        max_slope = -np.inf
        yi = z[i]
        for j in range(i + 1, n):
            slope = (z[j] - yi) / (j - i)
            if slope > max_slope:
                max_slope = slope
                degree[i] += 1
                degree[j] += 1
    max_k = int(degree.max()) if n else 0
    if max_k < 4:
        return np.nan
    ks = np.arange(1, max_k + 1)
    ccdf = np.array([np.mean(degree >= k) for k in ks], dtype=float)
    mask = (ks >= 2) & (ccdf > 0)
    if int(np.sum(mask)) < 4:
        return np.nan
    slope = _scaling_slope(np.log(ks[mask]), np.log(ccdf[mask]), min_r2=0.45, positive=False)
    if not np.isfinite(slope):
        return np.nan
    gamma = 1.0 - slope
    if not np.isfinite(gamma) or gamma <= 1.05 or gamma > 8:
        return np.nan
    # P(k) ~ k^{-γ} et, pour un fBm, γ = 3 - 2H donc d = 2 - H = (1 + γ) / 2
    d = (1.0 + gamma) / 2.0
    if d <= 0.5 or d > 6:
        return np.nan
    return float(d)


def danco_dimension(s, k=6):
    """DANCo allégé : MLE de Levina-Bickel corrigé par l'angle des voisins.

    d_angle = 1 / moyenne(cos² θ). Si l'angle voit une dimension plus basse
    que le MLE, la moyenne harmonique corrige le biais de courbure. Le
    résultat est plafonné par la dimension du plongement. Ce n'est pas la
    table de divergence KL de Ceruti et al.
    """
    _z, mat, times, tau = _embed_cloud(s, cap=90, m_min=4, m_max=8)
    n = len(mat)
    m_emb = mat.shape[1] if n else 0
    if n < k + 12 or m_emb < 2:
        return np.nan
    th = _cloud_theiler(times, tau, m_emb)
    dist = _pair_distances(mat, times, th)
    kk = min(k, n - 1)
    nn_idx = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    nd = dist[row, nn_idx]
    order = np.argsort(nd, axis=1)
    nn_idx = nn_idx[row, order]
    nd = nd[row, order]
    locals_ = []
    cos2 = []
    for i in range(n):
        di = nd[i]
        if np.isfinite(di[-1]) and di[0] > 1e-12:
            acc = 0.0
            used = 0
            for j in range(kk - 1):
                if di[j] > 1e-12 and np.isfinite(di[j]):
                    acc += np.log(di[-1] / di[j])
                    used += 1
            if used > 0 and acc > 0:
                locals_.append(used / acc)
        if np.isfinite(nd[i, 0]) and np.isfinite(nd[i, 1]):
            v1 = mat[nn_idx[i, 0]] - mat[i]
            v2 = mat[nn_idx[i, 1]] - mat[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-12 and n2 > 1e-12:
                c = float(np.dot(v1, v2) / (n1 * n2))
                cos2.append(c * c)
    if len(locals_) < 12 or len(cos2) < 12:
        return np.nan
    d_mle = float(np.mean(locals_))
    mean_cos2 = float(np.mean(cos2))
    if not np.isfinite(d_mle) or d_mle <= 0:
        return np.nan
    if mean_cos2 < 1e-8:
        d = d_mle
    else:
        d_angle = 1.0 / mean_cos2
        d = 2.0 / (1.0 / d_mle + 1.0 / d_angle) if d_angle < d_mle else d_mle
    if not np.isfinite(d) or d <= 0:
        return np.nan
    return float(min(d, float(m_emb)))


def fisher_dimension(s):
    """Participation de la métrique de Fisher Σ^{-1}.

    Les valeurs propres de Σ inférieures à 1 % de la plus grande sont
    écartées : ce plancher retire les directions de bruit numérique avant
    inversion. Le résultat est une dimension, pas l'information de Frieden.
    """
    _z, mat, _times, _tau = _embed_cloud(s, cap=140, m_min=4, m_max=8)
    m = mat.shape[1] if len(mat) else 0
    if len(mat) < m + 5 or m < 2:
        return np.nan
    cov = np.cov(mat, rowvar=False) + np.eye(m) * 1e-6
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-10]
    if len(eig) < 2:
        return np.nan
    kept = eig[eig >= 0.01 * np.max(eig)]
    if len(kept) < 1:
        return np.nan
    inv = 1.0 / kept
    return float((np.sum(inv) ** 2) / np.sum(inv ** 2))


def local_pca_dimension(s, k=12):
    """PCA locale : moyenne du nombre de composantes à 95 % dans chaque voisinage."""
    _z, mat, times, tau = _embed_cloud(s, cap=90, m_min=4, m_max=10)
    n = len(mat)
    m = mat.shape[1] if n else 0
    if n < k + 8 or m < 2:
        return np.nan
    dist = _pair_distances(mat, times, max(int(tau), 1))
    kk = min(k, n - 2)
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    dims = []
    for i in range(n):
        if not np.isfinite(dist[i, nn[i]]).all():
            continue
        neigh = mat[nn[i]]
        neigh = neigh - neigh.mean(axis=0)
        cov = np.cov(neigh, rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        pos = eig[eig > 1e-10]
        if len(pos) == 0:
            continue
        cum = np.cumsum(pos) / np.sum(pos)
        dims.append(int(np.searchsorted(cum, 0.95) + 1))
    if len(dims) < 12:
        return np.nan
    return float(np.mean(dims))


def graph_cluster_dimension(s):
    """Dimension de graphe par croissance de clusters (sauts sur un k-NN)."""
    _z, mat, times, tau = _embed_cloud(s, cap=70, m_max=6)
    n = len(mat)
    if n < 25:
        return np.nan
    dist = _pair_distances(mat, times, 0)
    k = min(8, max(4, n // 6), n - 1)
    nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in nn[i]:
            j = int(j)
            if np.isfinite(dist[i, j]):
                adj[i].add(j)
                adj[j].add(i)
    rng = np.random.default_rng(0)
    seeds = rng.choice(n, size=min(6, n), replace=False)
    slopes = []
    for seed in seeds:
        hop = -np.ones(n, dtype=int)
        hop[int(seed)] = 0
        queue = [int(seed)]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if hop[v] < 0:
                    hop[v] = hop[u] + 1
                    queue.append(v)
        max_h = int(hop.max())
        if max_h < 3:
            continue
        radii, counts = [], []
        for r in range(1, max_h + 1):
            c = int(np.sum((hop >= 0) & (hop <= r)))
            if c >= 3:
                radii.append(r)
                counts.append(c)
        if len(radii) < 3:
            continue
        # Le dernier rayon sature sur la composante : on le laisse de côté s'il y a assez de points.
        if len(radii) > 4:
            radii, counts = radii[:-1], counts[:-1]
        slope = _scaling_slope(np.log(radii), np.log(counts), min_r2=0.5)
        if np.isfinite(slope):
            slopes.append(slope)
    if len(slopes) < 2:
        return np.nan
    return float(np.median(slopes))


def persistent_homology_dimension(s):
    """Dimension par la mise à l'échelle des composantes H0 (union-find).

    N(ε) ~ ε^{-d} sur les échelles où les composantes connexes meurent.
    Implémentation autonome : pas de Ripser ni de Gudhi.
    """
    _z, mat, _times, _tau = _embed_cloud(s, m=3, cap=64, m_min=3, m_max=3)
    n = len(mat)
    if n < 20:
        return np.nan
    dist = _pair_distances(mat, None, 0)
    iu = np.triu_indices(n, k=1)
    dvec = dist[iu]
    finite = dvec[np.isfinite(dvec) & (dvec > 0)]
    if len(finite) < 20:
        return np.nan
    nn = np.min(dist, axis=1)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) < 8:
        return np.nan
    r_min = float(np.quantile(nn, 0.25))
    r_max = float(np.quantile(finite, 0.12))
    if not np.isfinite(r_min) or r_min <= 0:
        return np.nan
    if r_max <= r_min * 1.05:
        r_max = float(np.quantile(finite, 0.25))
    if r_max <= r_min:
        return np.nan
    radii = np.unique(np.geomspace(r_min, r_max, 7))
    order = np.argsort(dvec)
    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    comp = n
    ei = 0
    edges_i = iu[0][order]
    edges_j = iu[1][order]
    edges_d = dvec[order]
    n_edges = len(edges_d)
    xs, ys = [], []
    for r in radii:
        while ei < n_edges and edges_d[ei] <= r:
            if np.isfinite(edges_d[ei]) and union(int(edges_i[ei]), int(edges_j[ei])):
                comp -= 1
            ei += 1
        if r > 1e-12 and comp >= 2:
            xs.append(np.log(1.0 / r))
            ys.append(np.log(comp))
    slope = _scaling_slope(xs, ys, min_r2=0.2)
    if not np.isfinite(slope):
        return np.nan
    return float(min(slope, 12.0))


def mfa_dimension(s, n_comp=3):
    """Mélange d'analyseurs de facteurs, version PCA par composante.

    k-means puis PCA à 95 % dans chaque groupe. La lecture de divergence est
    le maximum local (le sous-espace le plus riche), pas la moyenne des groupes.
    """
    _z, mat, _times, _tau = _embed_cloud(s, cap=90, m_min=3, m_max=8)
    n = len(mat)
    m = mat.shape[1] if n else 0
    if n < n_comp * 8 or m < 2:
        return np.nan
    rng = np.random.default_rng(0)
    centroids = mat[rng.choice(n, size=n_comp, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(12):
        d2 = np.sum((mat[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        for c in range(n_comp):
            members = mat[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    local_dims = []
    for c in range(n_comp):
        members = mat[labels == c]
        if len(members) < m + 2:
            continue
        cov = np.cov(members, rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        pos = eig[eig > 1e-10]
        if len(pos) == 0:
            continue
        cum = np.cumsum(pos) / np.sum(pos)
        local_dims.append(int(np.searchsorted(cum, 0.95) + 1))
    if not local_dims:
        return np.nan
    return float(np.max(local_dims))


def kolmogorov_lz_dimension(s):
    """Proxy de dimension algorithmique : nombre de productions LZ76.

    Incalculable au sens strict de Kolmogorov. LZ76 sur un alphabet à 4
    symboles est le substitut classique. Il n'est pas borné et ce n'est pas
    une dimension de Hausdorff : il ne doit pas être moyenné avec les autres.
    """
    z = _zscore_series(s)
    n = len(z)
    if n < 32:
        return np.nan
    if n > 500:
        z = z[:500]
        n = 500
    qs = np.quantile(z, [0.25, 0.5, 0.75])
    seq = np.digitize(z, qs).tolist()
    c = 1
    i = 0
    k = 1
    l = 1
    kmax = 1
    while True:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l = l + kmax
                if l + 1 > n:
                    break
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1
    return float(c)


def mfdfa_spectrum(s):
    """MF-DFA court. Retourne Δα, la liste α, la liste f(α) et max f(α)."""
    z = _zscore_series(s)
    n = len(z)
    empty = (np.nan, [], [], np.nan)
    if n < 64:
        return empty
    y = np.cumsum(z - np.mean(z))
    scales = np.unique(np.logspace(np.log10(8), np.log10(max(9, n // 4)), num=8).astype(int))
    scales = scales[(scales >= 8) & (scales <= n // 4)]
    if len(scales) < 4:
        return empty
    hq, qs_used = [], []
    for q in (-3, -1, 0, 1, 2, 3):
        log_s, log_f = [], []
        for sc in scales:
            segments = n // int(sc)
            if segments < 2:
                continue
            y_trunc = y[:segments * int(sc)].reshape(segments, int(sc))
            x = np.arange(int(sc))
            x_mean = (int(sc) - 1) / 2.0
            x_var = np.var(x)
            if x_var < 1e-12:
                continue
            y_m = np.mean(y_trunc, axis=1, keepdims=True)
            cov = np.mean((y_trunc - y_m) * (x - x_mean), axis=1)
            slope = cov / x_var
            intercept = y_m.squeeze() - slope * x_mean
            trend = slope[:, None] * x + intercept[:, None]
            var = np.maximum(np.mean((y_trunc - trend) ** 2, axis=1), 1e-18)
            if abs(q) < 1e-9:
                fq = float(np.exp(0.5 * np.mean(np.log(var))))
            else:
                fq = float(np.mean(var ** (q / 2.0)) ** (1.0 / q))
            if fq > 0 and np.isfinite(fq):
                log_s.append(np.log(sc))
                log_f.append(np.log(fq))
        h = _scaling_slope(log_s, log_f, min_r2=0.3, positive=False)
        if np.isfinite(h):
            hq.append(h)
            qs_used.append(q)
    if len(hq) < 4:
        return empty
    qs = np.asarray(qs_used, dtype=float)
    hq = np.asarray(hq, dtype=float)
    order = np.argsort(qs)
    qs, hq = qs[order], hq[order]
    tau = qs * hq - 1.0
    alpha = np.gradient(tau, qs)
    falpha = qs * alpha - tau
    ok = np.isfinite(alpha) & np.isfinite(falpha)
    alpha, falpha = alpha[ok], falpha[ok]
    if len(alpha) < 3:
        return empty
    delta = float(np.max(alpha) - np.min(alpha))
    fmax = float(np.max(falpha))
    if not np.isfinite(delta) or delta < 0:
        delta = np.nan
    if not np.isfinite(fmax) or fmax < -0.5 or fmax > 3.5:
        fmax = np.nan
    return delta, [float(a) for a in alpha], [float(v) for v in falpha], fmax


def cohen_d(g1, g2):
    g1, g2 = np.asarray(g1), np.asarray(g2)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt((v1 + v2) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)

def cliff_delta(g1, g2, min_grp=3):
    g1, g2 = np.asarray(g1), np.asarray(g2)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < min_grp or n2 < min_grp:
        return np.nan
    try:
        U, _ = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        return (2 * U - n1 * n2) / (n1 * n2)
    except Exception:
        return np.nan

def separation_score(g1, g2, stat_test='cohen_d', min_grp=3):
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    _ds_rng = np.random.default_rng(0)
    if len(g1) > 1000:
        g1 = _ds_rng.choice(g1, size=1000, replace=False)
    if len(g2) > 1000:
        g2 = _ds_rng.choice(g2, size=1000, replace=False)
        
    d = cohen_d(g1, g2) if stat_test == 'cohen_d' else cliff_delta(g1, g2, min_grp)
    return abs(d) if not np.isnan(d) else np.nan

# =============================================================================
# SCAN DU SEUIL CRITIQUE
# =============================================================================
def _scan_critical_threshold_legacy(fluxes, control_series, mode_cfg, n_candidates=None, resolution=None, w_size=None, min_grp=None):
    """
    Scan du seuil critique avec paramètres personnalisables.
    
    Paramètres:
    - n_candidates: Nombre de seuils candidats (défaut: 1001)
    - resolution: Résolution en pourcentage (défaut: 1)
    - w_size: Taille de la fenêtre (défaut: 80 depuis mode_cfg)
    - min_grp: Groupe minimum (défaut: 3 depuis mode_cfg)
    """
    n_pts = len(control_series)
    if w_size is None:
        w_size = mode_cfg.get('w_size', 80)
    w_size = max(2, min(w_size, n_pts // 2))
    if min_grp is None:
        min_grp = mode_cfg.get('min_grp', 3)
    stat_test = mode_cfg.get('test', 'cohen_d')
    
    n_windows = n_pts - w_size + 1
    if n_windows < 1:
        return np.nan, []
    
    # Utiliser les paramètres personnalisés ou les valeurs par défaut
    # Priorité : resolution > n_candidates
    if resolution is not None and resolution > 0:
        # Générer les candidats avec un pas fixe basé sur la résolution
        candidats = np.arange(0, 100 + resolution, resolution)
        candidats = np.clip(candidats, 0, 100)  # S'assurer qu'on reste dans [0, 100]
    elif n_candidates is not None:
        # Utiliser n_candidates avec linspace
        candidats = np.linspace(0, 100, int(max(5, n_candidates)))
    else:
        # Plus de grille magique à 1001. Si le Monte Carlo a déjà retenu une
        # loupe, on la réutilise (stabilité). Sinon, milieu de la plage
        # autorisée par le nombre de fenêtres.
        tuned = mode_cfg.get('loupe_candidates')
        if tuned:
            n_use = int(tuned)
        else:
            lo, hi = loupe_candidate_bounds(n_windows)
            n_use = int(round((lo + hi) / 2.0))
        candidats = np.linspace(0, 100, int(max(5, n_use)))
    sort_idx = np.argsort(control_series)
    prog_s = control_series[sort_idx]
    flux_s = {k: v[sort_idx] for k, v in fluxes.items()}
    
    prog_cumsum = np.cumsum(np.concatenate(([0.0], prog_s)))
    win_progress = (prog_cumsum[w_size:] - prog_cumsum[:n_windows]) / w_size
    
    win_order = np.argsort(win_progress)
    win_progress_sorted = win_progress[win_order]
    
    scores = np.full(len(candidats), np.nan)  # abs Cohen d (separation score)
    for ci, s in enumerate(candidats):
        split = np.searchsorted(win_progress_sorted, s, side='left')
        n_low = split
        n_high = n_windows - split
        
        if n_low < min_grp or n_high < min_grp:
            continue
            
        low_win_ids = win_order[:split]
        high_win_ids = win_order[split:]
        
        sep = []
        for k, flux_arr in flux_s.items():
            v_low = np.concatenate([flux_arr[i:i+w_size] for i in low_win_ids])
            v_high = np.concatenate([flux_arr[i:i+w_size] for i in high_win_ids])
            d = separation_score(v_low, v_high, stat_test, min_grp)
            if not np.isnan(d):
                sep.append(d)
        scores[ci] = np.nanmean(sep) if sep else np.nan  # MEAN SEPARATION SCORE (abs Cohen's d)
        
    if np.all(np.isnan(scores)):
        return np.nan, []
        
    opt_threshold = candidats[np.nanargmax(scores)]
    
    scan_curve = []
    for cand, score in zip(candidats, scores):
        scan_curve.append({
            'candidate': float(cand),
            'score': float(score) if not np.isnan(score) else None
        })
        
    return float(opt_threshold), scan_curve


# =============================================================================
# SCAN RAPIDE v3.4-H3 — MÊME RÉSULTAT, SANS LES GROUPES CONCATÉNÉS
# =============================================================================
# Rappel du goulot v3.3 : pour chaque candidat et chaque flux, le scan
# legacy concaténait toutes les fenêtres glissantes (n_windows * w_size
# valeurs), alors que separation_score() ne gardait ensuite que 1 000
# valeurs tirées avec default_rng(0) (ou tout le groupe si <= 1 000).
# La version rapide ne matérialise jamais ces groupes : elle retrouve
# chaque valeur tirée par arithmétique d'indices. Le tirage est reproduit
# exactement, y compris le fait que separation_score() crée UN seul
# rng(0) par appel et tire g1 puis g2 avec ce même générateur (état
# avancé) — d'où le rng partagé ci-dessous. Cas NaN : on passe par le
# tableau des valeurs valides (ordre conservé) et les cumuls de comptes
# valides par fenêtre dans l'ordre win_order ; le préfixe d'un groupe
# est alors une simple tranche du cumul précalculé.
# Journal de repli : si le scan rapide lève une exception, le wrapper
# ci-dessous replie sur le scan legacy ET journalise (voir point 2 du
# changelog v3.4). Un repli silencieux est interdit dans ce fichier.
SCAN_FAST_FALLBACK_LOG = []


def _scan_critical_threshold_fast(fluxes, control_series, mode_cfg,
                                  n_candidates=None, resolution=None,
                                  w_size=None, min_grp=None):
    n_pts = len(control_series)
    if w_size is None:
        w_size = mode_cfg.get('w_size', 80)
    w_size = max(2, min(w_size, n_pts // 2))
    if min_grp is None:
        min_grp = mode_cfg.get('min_grp', 3)
    n_windows = n_pts - w_size + 1
    if n_windows < 1:
        return np.nan, []
    if resolution is not None and resolution > 0:
        candidats = np.clip(np.arange(0, 100 + resolution, resolution), 0, 100)
    elif n_candidates is not None:
        candidats = np.linspace(0, 100, int(max(5, n_candidates)))
    else:
        tuned = mode_cfg.get('loupe_candidates')
        if tuned:
            n_use = int(tuned)
        else:
            lo, hi = loupe_candidate_bounds(n_windows)
            n_use = int(round((lo + hi) / 2.0))
        candidats = np.linspace(0, 100, int(max(5, n_use)))
    # Tri, progression par fenetre et ordre : IDENTIQUES au scan legacy
    # (meme np.argsort par defaut, donc meme ordre, y compris entre ex aequo).
    sort_idx = np.argsort(control_series)
    prog_s = control_series[sort_idx]
    flux_s = {k: np.asarray(v, dtype=float)[sort_idx] for k, v in fluxes.items()}
    prog_cumsum = np.cumsum(np.concatenate(([0.0], prog_s)))
    win_progress = (prog_cumsum[w_size:] - prog_cumsum[:n_windows]) / w_size
    win_order = np.argsort(win_progress)
    win_progress_sorted = win_progress[win_order]
    # Precalculs par flux, une fois par scan (pas par candidat).
    infos = {}
    for k, arr in flux_s.items():
        valid = ~np.isnan(arr)
        valid_vals = arr[valid]
        # CORRECTION v3.4 (26 oct., revelee par le scan brut) : le debut
        # valide d'une fenetre est le PREMIER valide >= debut de fenetre,
        # pas valid_pos[debut] (qui vaut -1 si la fenetre COMMENCE par un
        # NaN, decalant alors tout l'echantillon de cette fenetre).
        cs = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
        win_valid = cs[w_size:] - cs[:n_windows]
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            win_start = np.full(n_windows, -1, dtype=np.int64)
        else:
            pos_in_valid = np.searchsorted(valid_indices, np.arange(n_windows))
            first_pos = valid_indices[np.clip(pos_in_valid, 0, len(valid_indices) - 1)]
            within = (pos_in_valid < len(valid_indices)) & (first_pos < np.arange(n_windows) + w_size)
            win_start = np.where((win_valid > 0) & within, pos_in_valid, -1)
        cum_win_order = np.concatenate(
            ([0], np.cumsum(win_valid[win_order].astype(np.int64))))
        infos[k] = {'arr': arr, 'valid_vals': valid_vals,
                    'win_start': win_start, 'cum_win_order': cum_win_order,
                    'no_nan': bool(valid.all())}
    scores = np.full(len(candidats), np.nan)
    for ci, s in enumerate(candidats):
        split = int(np.searchsorted(win_progress_sorted, s, side='left'))
        n_low = split
        n_high = n_windows - split
        if n_low < min_grp or n_high < min_grp:
            continue
        low_ids = win_order[:split]
        high_ids = win_order[split:]
        sep = []
        for k, info in infos.items():
            L_low = int(info['cum_win_order'][split])
            L_high = int(info['cum_win_order'][-1] - info['cum_win_order'][split])
            # UN seul rng(0) par appel, comme separation_score() : g1
            # avance l'etat, g2 tire ensuite avec ce meme generateur.
            rng = np.random.default_rng(0)
            g1 = _fast_group_sample(info, low_ids, split, 0, L_low, w_size, rng)
            g2 = _fast_group_sample(info, high_ids, split, 1, L_high, w_size, rng)
            d = cohen_d(g1, g2)
            if not np.isnan(d):
                sep.append(abs(d))
        scores[ci] = np.nanmean(sep) if sep else np.nan
    if np.all(np.isnan(scores)):
        return np.nan, []
    opt_threshold = float(candidats[np.nanargmax(scores)])
    scan_curve = [{'candidate': float(c),
                   'score': float(sc) if not np.isnan(sc) else None}
                  for c, sc in zip(candidats, scores)]
    return opt_threshold, scan_curve


def _fast_group_sample(info, group_ids, split, which, L, w_size, rng):
    """Valeurs du groupe apres filtrage NaN et tirage a 1 000 de
    separation_score(). which=0 : fenetres win_order[:split] ;
    which=1 : fenetres win_order[split:]."""
    if L == 0:
        return np.empty(0, dtype=float)
    cum = info['cum_win_order']
    if which == 0:
        prefix = cum[:split + 1]
    else:
        prefix = cum[split:] - cum[split]
    if L <= 1000:
        idx = np.arange(L, dtype=np.int64)
    else:
        idx = rng.choice(L, size=1000, replace=False)
    if info['no_nan']:
        win = idx // w_size
        off = idx % w_size
        return info['arr'][group_ids[win] + off]
    win = np.clip(np.searchsorted(prefix, idx, side='right') - 1,
                  0, len(group_ids) - 1)
    off = idx - prefix[win]
    starts = info['win_start'][group_ids[win]]
    return info['valid_vals'][starts + off]


def scan_critical_threshold(fluxes, control_series, mode_cfg, n_candidates=None,
                            resolution=None, w_size=None, min_grp=None):
    """Scan du seuil critique (v3.4) : version rapide par defaut, repli
    JOURNALISE sur le scan legacy en cas d'exception. Pour un test autre
    que cohen_d, le scan legacy est utilise directement (le scan rapide
    ne reproduit que le chemin cohen_d, seul chemin du protocole gelé)."""
    if mode_cfg.get('test', 'cohen_d') != 'cohen_d':
        return _scan_critical_threshold_legacy(
            fluxes, control_series, mode_cfg, n_candidates, resolution,
            w_size, min_grp)
    try:
        return _scan_critical_threshold_fast(
            fluxes, control_series, mode_cfg, n_candidates, resolution,
            w_size, min_grp)
    except Exception as exc:  # repli explicite, jamais silencieux
        msg = (f"SCAN FAST echoue ({type(exc).__name__}: {exc}) ; "
               f"repli sur scan legacy (w_size={w_size}, "
               f"n_candidates={n_candidates})")
        SCAN_FAST_FALLBACK_LOG.append(msg)
        print(f"⚠ {msg}")
        print(f"⚠ {msg}", file=sys.stderr)
        logging.warning(msg)
        return _scan_critical_threshold_legacy(
            fluxes, control_series, mode_cfg, n_candidates, resolution,
            w_size, min_grp)


def evaluate_scan_quality(scan_curve, threshold, w_size, n_pts):
    """
    Évalue la qualité du scan sans biais de résolution ou de bruit bimodal.

    scan_curve contient des SCORES DE SÉPARATION = abs(Cohen's d)
    entre groupes pré-seuil et post-seuil.
    Ce ne sont PAS les valeurs absolues des estimateurs de complexité
    (Higuchi, DFA, Hurst, etc.).
    """
    if not scan_curve or np.isnan(threshold):
        return {'max_separation': 0.0, 's_norm': 0.0, 'stability': 0.0, 'robustness': 0.0, 'sharpness': 0.0}
        
    scores = [s['score'] for s in scan_curve if s['score'] is not None]
    candidates = [s['candidate'] for s in scan_curve if s['score'] is not None]
    
    if len(scores) < 3:
        return {'max_separation': 0.0, 's_norm': 0.0, 'stability': 0.0, 'robustness': 0.0, 'sharpness': 0.0}
        
    max_idx = int(np.argmax(scores))
    max_separation = scores[max_idx]
    
    # 1. Normalisation Bornée de S_max
    s_norm = float(np.tanh(max_separation / 2.0))
    
    # 2. Stabilité Locale autour du pic
    window_size = max(3, len(scores) // 20)
    start_idx = max(0, max_idx - window_size)
    end_idx = min(len(scores), max_idx + window_size)
    local_scores = scores[start_idx:end_idx]
    
    stability = 1.0 / (1.0 + np.std(local_scores)) if len(local_scores) > 1 else 0.0
        
    # 3. FWHM Contiguë autour du sommet (Robuste au bruit distant)
    half_max = max_separation / 2.0
    
    # Extension vers la gauche
    left_idx = max_idx
    while left_idx > 0 and scores[left_idx] >= half_max:
        left_idx -= 1
        
    # Extension vers la droite
    right_idx = max_idx
    while right_idx < len(scores) - 1 and scores[right_idx] >= half_max:
        right_idx += 1
        
    # FWHM mesurée en % de la plage totale d'analyse
    fwhm_range = candidates[right_idx] - candidates[left_idx]
    
    # Pénalité Gaussienne (préfère une largeur entre 5% et 20%)
    ideal_fwhm = 10.0  # 10% de la plage
    robustness = float(np.exp(-((fwhm_range - ideal_fwhm)**2) / (2 * 5.0**2)))
        
    # 4. Acuité (Vraie Dérivée dS/dx)
    dx = np.diff(candidates)
    ds = np.diff(scores)
    
    # Éviter la division par zéro si deux candidats sont identiques
    valid_dx = dx > 1e-6
    if np.any(valid_dx):
        derivatives = np.abs(ds[valid_dx] / dx[valid_dx])
        # Recherche du gradient max autour du seuil
        local_derivs = derivatives[max(0, max_idx - window_size):min(len(derivatives), max_idx + window_size)]
        sharpness_val = float(np.max(local_derivs)) if len(local_derivs) > 0 else 0.0
    else:
        sharpness_val = 0.0
        
    # Normalisation douce avec tanh (évite le découpage brutal min(x/0.2, 1.0))
    sharpness_norm = float(np.tanh(sharpness_val / 0.5))
    
    return {
        'max_separation': float(max_separation),
        's_norm': s_norm,
        'stability': float(stability),
        'robustness': float(robustness),
        'sharpness': sharpness_norm
    }

def _sample_harmonic_guided(rng, n_pts, max_w_size, harmonic_prob):
    """Tirage guidé (φ, π, e, √2, √3) ou uniforme selon harmonic_prob.

    Retourne (w_size, min_grp, n_candidates, rule_used).
    """
    use_harmonic = rng.random() < float(harmonic_prob)
    rule_used = 'uniform'

    if use_harmonic:
        rule = rng.choice(['phi', 'pi', 'e', 'sqrt2', 'sqrt3'])
        rule_used = rule
        if rule == 'phi':
            k = int(rng.integers(1, 4))
            w_size = int(np.clip(n_pts / (PHI ** k), 10, max_w_size))
            n_windows = n_pts - w_size + 1
            lo, hi = loupe_candidate_bounds(n_windows)
            n_candidates = int(np.clip(round(lo * PHI), lo, hi))
        elif rule == 'pi':
            factor = float(rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
            w_size = int(np.clip(n_pts / (PI * factor), 10, max_w_size))
            n_windows = n_pts - w_size + 1
            lo, hi = loupe_candidate_bounds(n_windows)
            n_candidates = int(np.clip(round(n_windows / PI), lo, hi))
        elif rule == 'e':
            factor = float(rng.uniform(0.8, 2.2))
            w_size = int(np.clip(n_pts / (E * factor), 10, max_w_size))
            n_windows = n_pts - w_size + 1
            lo, hi = loupe_candidate_bounds(n_windows)
            n_candidates = int(np.clip(round(lo * E), lo, hi))
        elif rule == 'sqrt2':
            k = int(rng.integers(1, 5))
            w_size = int(np.clip(n_pts / (SQRT2 ** k), 10, max_w_size))
            n_windows = n_pts - w_size + 1
            lo, hi = loupe_candidate_bounds(n_windows)
            n_candidates = int(np.clip(round((lo + hi) / SQRT2), lo, hi))
        else:  # sqrt3
            k = int(rng.integers(1, 4))
            w_size = int(np.clip(n_pts / (SQRT3 ** k), 10, max_w_size))
            n_windows = n_pts - w_size + 1
            lo, hi = loupe_candidate_bounds(n_windows)
            n_candidates = int(np.clip(round(lo * SQRT3), lo, hi))
    else:
        w_size = int(rng.integers(10, min(200, max_w_size) + 1))
        n_windows = n_pts - w_size + 1
        lo, hi = loupe_candidate_bounds(n_windows)
        n_candidates = int(rng.integers(lo, hi + 1))

    min_grp = int(rng.integers(2, 11))

    # Sécurité longueur
    n_windows = n_pts - w_size + 1
    if n_windows < 2 * min_grp:
        w_size = max(10, n_pts - 2 * min_grp + 1)
        n_windows = n_pts - w_size + 1
        if w_size > max_w_size or n_windows < 2 * min_grp:
            return None
        lo, hi = loupe_candidate_bounds(max(n_windows, 2))
        n_candidates = int(np.clip(n_candidates, lo, hi))

    return w_size, min_grp, n_candidates, rule_used


def _monte_carlo_worker(iteration, fluxes, control_series, mode_cfg, n_pts, max_w_size, weights):
    """Évalue une configuration Monte Carlo reproductible (mode GUIDÉ v1).

    Chaque itération reçoit une graine dérivée de la graine globale.
    Avec probabilité HARMONIC_PROB, les paramètres sont dérivés de φ, π, e, √2, √3.
    Sinon, tirage uniforme classique (comportement v3.3).
    """
    weight_separation, weight_stability, weight_robustness, weight_sharpness = weights
    rng = np.random.default_rng(int(mode_cfg.get('random_seed', RANDOM_SEED)) + iteration)

    harmonic_prob = float(mode_cfg.get('harmonic_prob', HARMONIC_PROB))
    sampled = _sample_harmonic_guided(rng, n_pts, max_w_size, harmonic_prob)
    if sampled is None:
        return None
    w_size, min_grp, n_candidates, rule_used = sampled

    resolution = 100.0 / max(n_candidates - 1, 1)

    # Exécuter le scan avec ces paramètres
    threshold, scan_curve = scan_critical_threshold(
        fluxes, control_series, mode_cfg, n_candidates, None, w_size, min_grp
    )

    # Calculer les métriques de qualité
    quality_metrics = evaluate_scan_quality(scan_curve, threshold, w_size, n_pts)

    # Score composite
    composite_score = (
        weight_separation * quality_metrics['s_norm'] +
        weight_stability * quality_metrics['stability'] +
        weight_robustness * quality_metrics['robustness'] +
        weight_sharpness * quality_metrics['sharpness']
    )

    # Pénalité pour les fenêtres trop petites
    if w_size < n_pts // 10:
        composite_score *= 0.8

    return {
        'iteration': iteration,
        'rule': rule_used,
        'n_candidates': n_candidates,
        'resolution': resolution,
        'w_size': w_size,
        'min_grp': min_grp,
        'threshold': threshold,
        'score': composite_score,
        'metrics': quality_metrics,
        'scan_curve': scan_curve
    }

def optimize_threshold_scan_parameters_monte_carlo(fluxes, control_series, mode_cfg, n_iterations=100):
    """
    Optimisation Monte Carlo des paramètres de scan de seuil (parallélisée).
    
    Génère ``n_iterations`` combinaisons reproductibles de
    ``(n_candidates, w_size, min_grp)``, évalue chaque combinaison en parallèle
    et retourne la meilleure configuration avec sa courbe de scan.
    v3.4 : retourne AUSSI un 4e élément, le diagnostic d'optimisation
    (nombre d'itérations échouées et leurs numéros). Une itération échoue
    quand _sample_harmonic_guided() retourne None, c'est-à-dire quand
    aucune fenêtre valide ne respecte w_size <= max_w_size et
    n_windows >= 2 * min_grp après correction de sécurité — ce n'est
    pas une erreur de calcul, c'est un tirage invalide, mais il doit
    être compté et journalisé, pas filtré en silence (v3.3 : continue
    sans compter). Si TOUTES les itérations échouent, best_config est
    None et c'est l'appelant (run_pipeline) qui déclenche le repli
    JOURNALISÉ vers le seuil par défaut.
    """
    
    n_pts = len(control_series)
    max_w_size = n_pts // 2
    
    # Poids paramétrables du score composite
    weight_separation = mode_cfg.get('weight_separation', 0.3)
    weight_stability = mode_cfg.get('weight_stability', 0.3)
    weight_robustness = mode_cfg.get('weight_robustness', 0.2)
    weight_sharpness = mode_cfg.get('weight_sharpness', 0.2)
    weights = (weight_separation, weight_stability, weight_robustness, weight_sharpness)
    
    # Déterminer le nombre de workers (max 4 pour éviter surcharge)
    n_workers = min(4, mp.cpu_count(), n_iterations)
    use_parallel = n_workers > 1 and n_iterations >= 20
    
    print(f"Optimisation Monte Carlo HARMONIQUE GUIDÉ v1 ({n_iterations} itérations, P_harmonique={mode_cfg.get('harmonic_prob', HARMONIC_PROB)})...")
    if use_parallel:
        print(f"  Parallélisation: {n_workers} workers")
    
    best_config = None
    best_score = -np.inf
    best_scan_curve = None
    results = []
    failed_iterations = []

    if use_parallel:
        # Version parallélisée avec multiprocessing
        with mp.Pool(processes=n_workers) as pool:
            worker_args = partial(_monte_carlo_worker,
                                fluxes=fluxes,
                                control_series=control_series,
                                mode_cfg=mode_cfg,
                                n_pts=n_pts,
                                max_w_size=max_w_size,
                                weights=weights)

            worker_results = pool.map(worker_args, range(n_iterations))

        # Filtrer les résultats None et traiter — en COMPTANT les échecs
        for _it, res in enumerate(worker_results):
            if res is None:
                failed_iterations.append(_it)
                continue
            results.append(res)
            
            if res['score'] > best_score:
                best_score = res['score']
                best_config = {
                    'rule': res.get('rule', 'unknown'),
                    'n_candidates': res['n_candidates'],
                    'resolution': res['resolution'],
                    'w_size': res['w_size'],
                    'min_grp': res['min_grp'],
                    'threshold': res['threshold'],
                    'score': res['score'],
                    'metrics': res['metrics']
                }
                best_scan_curve = res['scan_curve']
    else:
        # Version séquentielle (fallback)
        for iteration in range(n_iterations):
            res = _monte_carlo_worker(iteration, fluxes, control_series, mode_cfg, n_pts, max_w_size, weights)
            if res is None:
                failed_iterations.append(iteration)
                continue
            results.append(res)
            
            if res['score'] > best_score:
                best_score = res['score']
                best_config = {
                    'rule': res.get('rule', 'unknown'),
                    'n_candidates': res['n_candidates'],
                    'resolution': res['resolution'],
                    'w_size': res['w_size'],
                    'min_grp': res['min_grp'],
                    'threshold': res['threshold'],
                    'score': res['score'],
                    'metrics': res['metrics']
                }
                best_scan_curve = res['scan_curve']
    
    diagnostics = {
        'n_iterations': int(n_iterations),
        'n_success': int(len(results)),
        'n_failed': int(len(failed_iterations)),
        'failed_iterations': list(failed_iterations),
    }
    if failed_iterations:
        _msg = (f"Monte Carlo : {len(failed_iterations)}/{n_iterations} "
                f"iteration(s) echouee(s) (tirages invalides) : "
                f"{failed_iterations}")
        print(f"⚠ {_msg}")
        print(f"⚠ {_msg}", file=sys.stderr)
        logging.warning(_msg)
    print(f"✓ Optimisation terminée - Meilleur score: {best_score:.3f}")
    if best_config:
        print(f"  règle: {best_config.get('rule', '?')}")
        print(f"  n_candidates: {best_config['n_candidates']}, resolution: {best_config['resolution']:.3f}%")
        print(f"  w_size: {best_config['w_size']}, min_grp: {best_config['min_grp']}")
    else:
        _msg = ("Monte Carlo : AUCUNE configuration valide sur "
                f"{n_iterations} iterations ; l'appelant doit appliquer "
                "le repli JOURNALISE vers le seuil par defaut.")
        print(f"⚠ {_msg}")
        print(f"⚠ {_msg}", file=sys.stderr)
        logging.warning(_msg)

    return best_config, best_scan_curve, results, diagnostics

# =============================================================================
# STABILITÉ DU SEUIL
# =============================================================================
def compute_threshold_stability(fluxes, control_series, window_scan, mode_cfg):
    thresholds = []
    for w in window_scan:
        cfg = mode_cfg.copy()
        cfg['w_size'] = w
        th, _ = scan_critical_threshold(fluxes, control_series, cfg)
        if not np.isnan(th):
            thresholds.append(th)
            
    if not thresholds:
        return {'list': [], 'mean': None, 'median': None, 'std': None, 'iqr': None, 'min': None, 'max': None, 'rating': 'UNKNOWN'}
        
    thresholds = np.array(thresholds)
    q75, q25 = np.percentile(thresholds, [75, 25])
    iqr = q75 - q25
    std_val = np.std(thresholds, ddof=1) if len(thresholds) > 1 else 0.0
    
    rating = "VERY STABLE"
    if std_val < 3.0 and iqr < 4.0:
        rating = "VERY STABLE"
    elif std_val < 6.0 and iqr < 8.0:
        rating = "MODERATELY STABLE"
    else:
        rating = "UNSTABLE"
        
    return {
        'list': thresholds.tolist(),
        'mean': float(np.mean(thresholds)),
        'median': float(np.median(thresholds)),
        'std': float(std_val),
        'iqr': float(iqr),
        'min': float(np.min(thresholds)),
        'max': float(np.max(thresholds)),
        'rating': rating
    }

# =============================================================================
# DIAGNOSTIC DE BRUIT (4 ÉTAPES) - Version Multi-Flux
# =============================================================================
_WN_REF_CACHE = {}
_WN_REF_SEED = 1234
_WN_REF_REPS = 8


def _white_noise_reference(length, reps=_WN_REF_REPS):
    """Références bruit blanc empiriques par estimateur, pour une longueur.

    R réplications gaussiennes à graine fixe, mêmes paramètres
    d'appel qu'à l'étape 3 du diagnostic de bruit. Mise en cache par
    longueur. Aucune constante supposée : la référence est mesurée.
    """
    length = int(max(30, length))
    if length in _WN_REF_CACHE:
        return _WN_REF_CACHE[length]
    vals = {m: [] for m in ESTIMATOR_NAMES}
    for r in range(int(reps)):
        rng = np.random.default_rng(_WN_REF_SEED + 7919 * length + r)
        wn = rng.standard_normal(length)
        try:
            est = apply_complexity(wn, mode_cfg={'min_len': 10})
        except Exception:
            est = {}
        for m in ESTIMATOR_NAMES:
            try:
                v = float(est.get(m, np.nan))
            except (TypeError, ValueError):
                v = np.nan
            vals[m].append(v if np.isfinite(v) else np.nan)
    ref = {}
    for m in ESTIMATOR_NAMES:
        arr = np.asarray(vals[m], dtype=float)
        ok = ~np.isnan(arr)
        ref[m] = (float(np.mean(arr[ok])) if ok.any() else None,
                  float(np.std(arr[ok])) if ok.any() else None,
                  float(np.mean(~ok)))
    _WN_REF_CACHE[length] = ref
    return ref


def compute_single_flux_noise_diagnostic(flux_name, flux_series, progress_arr, other_fluxes, opt_threshold):
    """Calcule le diagnostic de bruit pour un seul flux."""
    try:
        noise_series = np.asarray(flux_series, dtype=float)
        n = len(noise_series)
        if n < 20:
            return None

        valid_mask = ~np.isnan(noise_series) & ~np.isnan(progress_arr)
        if np.sum(valid_mask) < 20:
            return None

        noise_clean = noise_series[valid_mask]
        prog_clean = progress_arr[valid_mask]
        m = len(noise_clean)

        opt_thresh_val = float(opt_threshold) if opt_threshold is not None and not np.isnan(opt_threshold) else 52.5
        post_mask = prog_clean >= opt_thresh_val
        pre_mask = prog_clean < opt_thresh_val
        
        # Step 1: True White Noise Replication
        try:
            _wn_rng = np.random.default_rng(42)
            true_white_noise = _wn_rng.standard_normal(m)
            
            if np.sum(post_mask) > 3 and np.sum(pre_mask) > 3:
                cand_pre_mean, cand_post_mean = np.mean(noise_clean[pre_mask]), np.mean(noise_clean[post_mask])
                cand_pre_std, cand_post_std = np.std(noise_clean[pre_mask]), np.std(noise_clean[post_mask])
                pooled_s = np.sqrt((cand_pre_std**2 + cand_post_std**2) / 2.0)
                cand_q5_score = float(abs(cand_post_mean - cand_pre_mean) / (pooled_s + 1e-9) * (1.0 + cand_post_std / (cand_pre_std + 1e-9)))
                
                white_pre_mean, white_post_mean = np.mean(true_white_noise[pre_mask]), np.mean(true_white_noise[post_mask])
                white_pre_std, white_post_std = np.std(true_white_noise[pre_mask]), np.std(true_white_noise[post_mask])
                pooled_w = np.sqrt((white_pre_std**2 + white_post_std**2) / 2.0)
                white_q5_score = float(abs(white_post_mean - white_pre_mean) / (pooled_w + 1e-9) * (1.0 + white_post_std / (white_pre_std + 1e-9)))
            else:
                cand_q5_score = 0.0
                white_q5_score = 0.0

            ratio_val = float(cand_q5_score / max(white_q5_score, 0.01))
        except Exception:
            cand_q5_score, white_q5_score, ratio_val = 0.0, 0.0, 0.0

        # Step 2: Correlation with Local Volatility
        try:
            window = max(5, m // 20)
            # Aligner les autres flux sur le même masque de validité que le
            # flux étudié afin d'éviter des vecteurs de tailles différentes.
            aligned_other_fluxes = []
            for other_flux in other_fluxes:
                other_arr = np.asarray(other_flux, dtype=float)
                if len(other_arr) != n:
                    continue
                other_clean = other_arr[valid_mask]
                if np.all(np.isnan(other_clean)):
                    continue
                if np.any(np.isnan(other_clean)):
                    other_clean = np.where(
                        np.isnan(other_clean), np.nanmean(other_clean), other_clean
                    )
                aligned_other_fluxes.append(other_clean)

            if aligned_other_fluxes:
                avg_system = np.mean(aligned_other_fluxes, axis=0)
                system_vol = pd.Series(avg_system).rolling(window=window, min_periods=2).std().fillna(0).values
            else:
                system_vol = pd.Series(noise_clean).rolling(window=window, min_periods=2).std().fillna(0).values
                
            noise_var = pd.Series(noise_clean).rolling(window=window, min_periods=2).var().fillna(0).values
            
            if np.sum(pre_mask) > 5 and np.sum(post_mask) > 5:
                corr_before, _ = stats.spearmanr(system_vol[pre_mask], noise_var[pre_mask])
                corr_after, _ = stats.spearmanr(system_vol[post_mask], noise_var[post_mask])
                corr_before = 0.0 if np.isnan(corr_before) else float(corr_before)
                corr_after = 0.0 if np.isnan(corr_after) else float(corr_after)
            else:
                corr_before, corr_after = 0.0, 0.0

            delta_corr = corr_after - corr_before
        except Exception:
            corr_before, corr_after, delta_corr = 0.0, 0.0, 0.0

        # Step 3: Hurst Exponent Drifting
        hurst_computed = True
        try:
            n_bins = 5
            prog_min, prog_max = np.min(prog_clean), np.max(prog_clean)
            if prog_max <= prog_min:
                prog_min, prog_max = 0.0, 100.0
            bins = np.linspace(prog_min, prog_max, n_bins + 1)
            hurst_bins = []
            insufficient_data_bins = 0
            bin_ests = []
            bin_lengths = []
            
            for i in range(n_bins):
                b_min, b_max = bins[i], bins[i+1]
                bin_mask = (prog_clean >= b_min) & (prog_clean <= b_max)
                bin_noise = noise_clean[bin_mask]
                if len(bin_noise) >= 15:
                    mode_cfg = {'min_len': 10}
                    # Z-score normalization pour éviter l'instabilité des estimateurs
                    # (ex: Higuchi=58 avec des valeurs brutes non normalisées)
                    bin_std = np.std(bin_noise)
                    if bin_std > 1e-10:
                        bin_noise = (bin_noise - np.mean(bin_noise)) / bin_std
                    res_est = apply_complexity(bin_noise, mode_cfg=mode_cfg)
                    bin_ests.append(dict(res_est))
                    bin_lengths.append(int(len(bin_noise)))
                    h_val = res_est.get('Hurst', np.nan)
                    if np.isnan(h_val) or h_val is None:
                        hurst_computed = False
                        h_val = np.nan
                else:
                    insufficient_data_bins += 1
                    hurst_computed = False
                    h_val = np.nan
                    bin_ests.append(None)
                    bin_lengths.append(int(len(bin_noise)))
                hurst_bins.append(h_val)

            if insufficient_data_bins > 2:
                hurst_computed = False
                
            if hurst_computed and hurst_bins:
                valid_hurst = [h for h in hurst_bins if not np.isnan(h)]
                if valid_hurst:
                    max_h = max(valid_hurst)
                    avg_h = np.mean(valid_hurst)
                else:
                    hurst_computed = False
                    max_h = np.nan
                    avg_h = np.nan
            else:
                max_h = np.nan
                avg_h = np.nan
        except Exception:
            hurst_computed = False
            hurst_bins = [np.nan] * 5
            max_h = np.nan
            avg_h = np.nan


        # Dérive par estimateur (v3.5.2, ADDITIF) : les 29 valeurs déjà
        # calculées ci-dessus sont comparées à des références bruit blanc
        # empiriques (même longueur, graine fixe). Le score historique
        # (Hurst seul) et ses composantes sont inchangés.
        estimator_drift = {}
        try:
            _refs = {}

            def _get_ref(_L):
                _key = max(30, int(round(_L / 10.0)) * 10)
                if _key not in _refs:
                    _refs[_key] = _white_noise_reference(_key, _WN_REF_REPS)
                return _refs[_key], _key

            for _metric in ESTIMATOR_NAMES:
                _devs = []
                _signed = []
                _n_nan = 0
                _ref_shown = None
                _ref_key = None
                for _L, _est in zip(bin_lengths, bin_ests):
                    if _est is None:
                        _n_nan += 1
                        continue
                    _v = _est.get(_metric, np.nan)
                    if _v is None:
                        _n_nan += 1
                        continue
                    try:
                        _v = float(_v)
                    except (TypeError, ValueError):
                        _n_nan += 1
                        continue
                    if not np.isfinite(_v):
                        _n_nan += 1
                        continue
                    _ref, _ref_key = _get_ref(_L)
                    _mean, _std, _nan_rate = _ref[_metric]
                    if _ref_shown is None:
                        _ref_shown = (_mean, _std, _nan_rate)
                    if _mean is None or _std is None or not np.isfinite(_mean) or not np.isfinite(_std):
                        _n_nan += 1
                        continue
                    _devs.append(abs(_v - _mean) / max(_std, 1e-3))
                    _signed.append(_v - _mean)
                if _devs:
                    estimator_drift[_metric] = {
                        'drift_max': float(max(_devs)),
                        'drift_mean': float(np.mean(_devs)),
                        'signed_deviation_max_abs': float(max(_signed, key=abs)) if _signed else None,
                        'n_nan_bins': int(_n_nan),
                        'ref_length': int(_ref_key) if _ref_key is not None else None,
                        'ref_mean': float(_ref_shown[0]) if _ref_shown[0] is not None and np.isfinite(_ref_shown[0]) else None,
                        'ref_std': float(_ref_shown[1]) if _ref_shown[1] is not None and np.isfinite(_ref_shown[1]) else None,
                        'ref_nan_rate': float(_ref_shown[2]) if _ref_shown[2] is not None else None,
                        'ref_degenerate': bool(_ref_shown[1] is not None and _ref_shown[1] < 1e-3),
                    }
                else:
                    estimator_drift[_metric] = {
                        'drift_max': None, 'drift_mean': None,
                        'signed_deviation_max_abs': None,
                        'n_nan_bins': int(_n_nan), 'ref_length': None,
                        'ref_mean': None, 'ref_std': None, 'ref_nan_rate': None,
                        'ref_degenerate': None,
                    }
        except Exception:
            estimator_drift = {}

        # Step 4: Spectral FFT Density Analysis
        spectral_computed = True
        try:
            # Le seuil est défini sur l'axe de progression. Le flux doit donc
            # être trié par progression avant de séparer les deux segments FFT.
            order = np.argsort(prog_clean)
            prog_sorted = prog_clean[order]
            noise_sorted = noise_clean[order]
            seuil_idx = int(np.searchsorted(prog_sorted, opt_thresh_val, side='left'))
            seuil_idx = max(10, min(m - 10, seuil_idx))
            seg_len = min(200, seuil_idx, m - seuil_idx)
            
            if seg_len >= 10:
                seg_before = noise_sorted[seuil_idx - seg_len : seuil_idx]
                seg_after = noise_sorted[seuil_idx : seuil_idx + seg_len]
                
                fft_bef = np.abs(np.fft.rfft(seg_before))**2
                fft_aft = np.abs(np.fft.rfft(seg_after))**2
                freqs = np.fft.rfftfreq(len(seg_before), d=1.0)
                
                pos_m = freqs > 0
                freqs_pos = freqs[pos_m]
                fft_bef_pos = fft_bef[pos_m]
                fft_aft_pos = fft_aft[pos_m]
                
                if len(freqs_pos) > 3:
                    slope_bef, _, _, _, _ = stats.linregress(np.log10(freqs_pos), np.log10(fft_bef_pos + 1e-12))
                    slope_aft, _, _, _, _ = stats.linregress(np.log10(freqs_pos), np.log10(fft_aft_pos + 1e-12))
                else:
                    spectral_computed = False
                    slope_bef = np.nan
                    slope_aft = np.nan
            else:
                spectral_computed = False
                slope_bef = np.nan
                slope_aft = np.nan
        except Exception:
            spectral_computed = False
            slope_bef = np.nan
            slope_aft = np.nan

        # Calcul du score nuancé de proxy de variance
        # Score entre 0 (bruit blanc pur) et 1 (proxy CSD fort)
        variance_proxy_score = 0.0
        computed_status = True
        
        # Composante Q5 (asymétrie de variance)
        if white_q5_score > 0:
            q5_component = min(cand_q5_score / (3 * white_q5_score), 1.0)
            variance_proxy_score += 0.3 * q5_component
        
        # Composante corrélation volatilité
        if delta_corr > 0:
            corr_component = min(delta_corr / 0.4, 1.0)
            variance_proxy_score += 0.25 * corr_component
        
        # Composante mémoire (Hurst)
        if hurst_computed and not np.isnan(max_h) and max_h > 0.5:
            memory_component = min((max_h - 0.5) / 0.2, 1.0)
            variance_proxy_score += 0.25 * memory_component
        elif not hurst_computed:
            computed_status = False
        
        # Composante spectrale (pente FFT)
        if spectral_computed and not np.isnan(slope_aft) and slope_aft < -0.15:
            spectral_component = min(abs(slope_aft) / 0.5, 1.0)
            variance_proxy_score += 0.2 * spectral_component
        elif not spectral_computed:
            computed_status = False

        return {
            'flux_name': flux_name,
            'variance_proxy_score': float(np.round(variance_proxy_score, 3)) if computed_status else np.nan,
            'cand_q5_score': float(np.round(cand_q5_score, 2)),
            'white_q5_score': float(np.round(white_q5_score, 2)),
            'delta_corr': float(np.round(delta_corr, 3)),
            'max_hurst': float(np.round(max_h, 3)) if hurst_computed else np.nan,
            'avg_hurst': float(np.round(avg_h, 3)) if hurst_computed else np.nan,
            'slope_after': float(np.round(slope_aft, 3)) if spectral_computed else np.nan,
            'computed': computed_status,
            'hurst_computed': hurst_computed,
            'spectral_computed': spectral_computed,
            'estimator_drift': estimator_drift,
            'drift_reference_seed': _WN_REF_SEED,
            'drift_reference_reps': _WN_REF_REPS,
            'status': 'computed' if computed_status else 'insufficient_data'
        }
    except Exception:
        return None

def compute_noise_diagnostic(df, progress, fluxes, opt_threshold, active_names=None, receptive_names=None):
    """Calcule le diagnostic de bruit pour tous les flux et estimateurs."""
    try:
        if not fluxes:
            return None

        progress_arr = np.asarray(progress, dtype=float)
        opt_thresh_val = float(opt_threshold) if opt_threshold is not None and not np.isnan(opt_threshold) else 52.5
        
        # Calculer le diagnostic pour chaque flux
        all_diagnostics = []
        for flux_name in fluxes.keys():
            other_fluxes = [np.asarray(fluxes[fn], dtype=float) for fn in fluxes.keys() if fn != flux_name]
            diag = compute_single_flux_noise_diagnostic(
                flux_name, fluxes[flux_name], progress_arr, other_fluxes, opt_thresh_val
            )
            if diag:
                all_diagnostics.append(diag)
        
        if not all_diagnostics:
            return None

        # Analyser les résultats globaux
        scores = [d['variance_proxy_score'] for d in all_diagnostics]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # Verdict nuancé basé sur la distribution des scores
        if avg_score < 0.2:
            diagnostic_label = "Bruit blanc dominant - Tous les flux présentent des caractéristiques de bruit blanc"
            verdict_level = "LOW"
        elif avg_score < 0.4:
            diagnostic_label = "Bruit blanc avec signaux faibles - Quelques flux montrent des signatures de proxy"
            verdict_level = "LOW_MEDIUM"
        elif avg_score < 0.6:
            diagnostic_label = "Mixte - Coexistence de bruit blanc et de proxies de variance résiduelle"
            verdict_level = "MEDIUM"
        elif avg_score < 0.8:
            diagnostic_label = "Proxy de variance modéré - Plusieurs flux agissent comme proxies de variance résiduelle"
            verdict_level = "MEDIUM_HIGH"
        else:
            diagnostic_label = "Proxy de variance résiduelle colorée (CSD) - Forte signature de mémoire et de variance"
            verdict_level = "HIGH"

        # Identifier les flux les plus problématiques
        sorted_diagnostics = sorted(all_diagnostics, key=lambda x: x['variance_proxy_score'], reverse=True)
        top_proxies = sorted_diagnostics[:3]
        bottom_noise = sorted_diagnostics[-3:]
        
        # Générer le résumé détaillé
        summary_text = (
            f"DIAGNOSTIC MULTI-FLUX - Analyse sur {len(all_diagnostics)} flux\n\n"
            f"Score moyen de proxy de variance: {avg_score:.3f} (échelle 0-1)\n"
            f"Score maximum: {max_score:.3f}, Score minimum: {min_score:.3f}\n\n"
            f"TOP 3 FLUX AVEC SIGNATURE CSD:\n"
        )
        
        for i, diag in enumerate(top_proxies, 1):
            summary_text += (
                f"  {i}. {diag['flux_name']} (score: {diag['variance_proxy_score']:.3f})\n"
                f"     - Q5 ratio: {diag['cand_q5_score']:.2f} vs bruit blanc {diag['white_q5_score']:.2f}\n"
                f"     - Hurst max: {diag['max_hurst']:.3f}, Pente FFT: {diag['slope_after']:.3f}\n"
            )
        
        summary_text += f"\nBOTTOM 3 FLUX (BRUIT BLANC):\n"
        for i, diag in enumerate(bottom_noise, 1):
            summary_text += (
                f"  {i}. {diag['flux_name']} (score: {diag['variance_proxy_score']:.3f})\n"
                f"     - Hurst moyen: {diag['avg_hurst']:.3f}\n"
            )

        # Conserver la compatibilité avec l'ancien format en utilisant le flux le plus problématique
        worst_flux = top_proxies[0] if top_proxies else all_diagnostics[0]
        
        return {
            'is_variance_proxy': avg_score > 0.4,
            'noise_col': worst_flux['flux_name'],
            'opt_threshold': opt_thresh_val,
            'summary_text': summary_text,
            'diagnostic_label': diagnostic_label,
            'verdict_level': verdict_level,
            'avg_variance_proxy_score': float(avg_score),
            'max_variance_proxy_score': float(max_score),
            'min_variance_proxy_score': float(min_score),
            'all_flux_diagnostics': all_diagnostics,
            'top_proxies': top_proxies,
            'bottom_noise': bottom_noise,
            # Compatibilité avec l'ancien format
            'step1': {
                'noise_column': worst_flux['flux_name'],
                'candidate_q5_score': worst_flux['cand_q5_score'],
                'true_white_noise_q5_score': worst_flux['white_q5_score'],
                'ratio': float(worst_flux['cand_q5_score'] / max(worst_flux['white_q5_score'], 0.01)),
                'interpretation': f"Score Q5 de {worst_flux['flux_name']}: {worst_flux['cand_q5_score']:.2f} vs bruit blanc {worst_flux['white_q5_score']:.2f}"
            },
            'step2': {
                'corr_before_threshold': 0.0,
                'corr_after_threshold': worst_flux['delta_corr'],
                'delta_corr': worst_flux['delta_corr'],
                'interpretation': f"Delta corrélation volatilité: {worst_flux['delta_corr']:.3f}"
            },
            'step3': {
                'hurst_by_bin': [{'bin': f"{i*20}% - {(i+1)*20}%", 'hurst': worst_flux['avg_hurst']} for i in range(5)],
                'interpretation': f"Hurst moyen: {worst_flux['avg_hurst']:.3f}, Max: {worst_flux['max_hurst']:.3f}"
            },
            'step4': {
                'spectral_slope_before': -0.1,
                'spectral_slope_after': worst_flux['slope_after'],
                'psd_chart': [],
                'interpretation': f"Pente spectrale post-seuil: {worst_flux['slope_after']:.3f}"
            }
        }
    except Exception as err:
        return {
            'is_variance_proxy': False,
            'noise_col': 'Erreur',
            'opt_threshold': float(opt_threshold) if opt_threshold else 52.5,
            'summary_text': f"Erreur lors du calcul du diagnostic bruit: {str(err)}",
            'diagnostic_label': "❌ Erreur de Diagnostic",
            'error': str(err),
            'all_flux_diagnostics': [],
            'step1': {'candidate_q5_score': 0, 'true_white_noise_q5_score': 0, 'ratio': 0, 'interpretation': f"Erreur: {str(err)}"},
            'step2': {'corr_before_threshold': 0, 'corr_after_threshold': 0, 'delta_corr': 0, 'interpretation': "Non disponible"},
            'step3': {'hurst_by_bin': [], 'interpretation': "Non disponible"},
            'step4': {'spectral_slope_before': 0, 'spectral_slope_after': 0, 'psd_chart': [], 'interpretation': "Non disponible"}
        }

# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================
def plot_complexity_landscape(df_feat, fluxes, main_threshold, output_path):
    """Génère le graphique du paysage de complexité avec les 5 estimateurs."""
    metrics = ['Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Paysage de Complexité - Évolution des Estimateurs', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot each flux
        for flux_name in fluxes.keys():
            col_name = f"{flux_name}_{metric}_mean"
            if col_name in df_feat.columns:
                data = df_feat[['Progress', col_name]].dropna()
                if len(data) > 0:
                    ax.plot(data['Progress'], data[col_name], label=flux_name, alpha=0.7, linewidth=2)
        
        ax.axvline(x=main_threshold, color='red', linestyle='--', linewidth=2, label='Seuil Critique')
        ax.set_xlabel('Progression (%)')
        ax.set_ylabel(f'{metric} (Z-score)')
        ax.set_title(f'Dimension Fractale de {metric}' if metric in ['Higuchi', 'Katz', 'DFA'] else 
                     f'Exposant de {metric}' if metric == 'Hurst' else f'Entropie d\'Échantillon ({metric})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if odd number of metrics
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique de complexité sauvegardé: {output_path}")

def plot_threshold_scan(scan_curve, main_threshold, output_path):
    """Génère la courbe de scan du seuil critique."""
    if not scan_curve:
        print("⚠ Aucune donnée de scan disponible")
        return
    
    candidates = [c['candidate'] for c in scan_curve if c['score'] is not None]
    scores = [c['score'] for c in scan_curve if c['score'] is not None]
    
    if not candidates:
        print("⚠ Aucune donnée valide pour la courbe de scan")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(candidates, scores, linewidth=2, color='steelblue')
    ax.axvline(x=main_threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil Optimal: {main_threshold:.1f}%')
    ax.axhline(y=max(scores), color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Seuil Candidat (%)')
    ax.set_ylabel('Score de Séparation Moyen')
    ax.set_title('Courbe d\'Émergence du Seuil Critique', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Courbe de scan sauvegardée: {output_path}")

def plot_raw_fluxes(distribution_samples, fluxes_ui, main_threshold, output_path):
    """Génère le graphique de distribution des flux bruts."""
    if not distribution_samples:
        print("⚠ Aucune donnée de distribution disponible")
        return
    
    df_dist = pd.DataFrame(distribution_samples)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    flux_cols = [col for col in df_dist.columns if col != 'progress']
    
    for flux_col in flux_cols:
        ax.plot(df_dist['progress'], df_dist[flux_col], label=flux_col, alpha=0.7, linewidth=2)
    
    ax.axvline(x=main_threshold, color='red', linestyle='--', linewidth=2, label='Seuil Critique')
    ax.set_xlabel('Progression (%)')
    ax.set_ylabel('Valeur Normalisée [0,1]')
    ax.set_title('Profil des Flux Bruts - Distribution Normalisée', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique de distribution sauvegardé: {output_path}")

def plot_separation_heatmap(scores, output_path):
    """Génère la heatmap de séparation pour les questions Q1-Q5."""
    # Prepare data for heatmap
    questions = ['Q1_Calme_vs_Chaos', 'Q2_Tension_Seuil', 'Q3_Agressivite', 'Q4_Persistence', 'Q5_Actif_Receptif']
    
    # Get all unique features
    all_features = set()
    for q_name in questions:
        if q_name in scores:
            all_features.update(scores[q_name].keys())
    all_features = sorted(list(all_features))
    
    if not all_features:
        print("⚠ Aucune donnée de séparation disponible")
        return
    
    # Create matrix
    matrix_data = []
    for feature in all_features:
        row = []
        for q_name in questions:
            if q_name in scores and feature in scores[q_name]:
                row.append(scores[q_name][feature])
            else:
                row.append(0.0)
        matrix_data.append(row)
    
    matrix_data = np.array(matrix_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=np.max(matrix_data))
    
    # Set ticks
    ax.set_xticks(np.arange(len(questions)))
    ax.set_yticks(np.arange(len(all_features)))
    ax.set_xticklabels(['Q1\nCalme vs\nChaos', 'Q2\nTension\nSeuil', 'Q3\nAgressivité', 'Q4\nPersistance', 'Q5\nActif vs\nRéceptif'])
    ax.set_yticklabels(all_features, fontsize=8)
    
    # Add text annotations
    for i in range(len(all_features)):
        for j in range(len(questions)):
            text = ax.text(j, i, f'{matrix_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Matrice de Séparation de Régime', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score de Séparation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap de séparation sauvegardée: {output_path}")

def plot_stability_analysis(stability_report, output_path):
    """Génère le graphique d'analyse de stabilité du seuil."""
    if not stability_report or not stability_report.get('list'):
        print("⚠ Aucune donnée de stabilité disponible")
        return
    
    window_sizes = [50, 80, 100, 120, 150, 180, 220]
    thresholds = stability_report['list']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(thresholds)), thresholds, color='steelblue', alpha=0.7)
    ax.axhline(y=stability_report['mean'], color='red', linestyle='--', linewidth=2, label=f'Moyenne: {stability_report["mean"]:.1f}%')
    ax.axhline(y=stability_report['median'], color='green', linestyle=':', linewidth=2, label=f'Médiane: {stability_report["median"]:.1f}%')
    
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(w) for w in window_sizes[:len(thresholds)]])
    ax.set_xlabel('Taille de Fenêtre')
    ax.set_ylabel('Seuil Critique Détecté (%)')
    ax.set_title(f'Convergence des Seuils - Stabilité: {stability_report["rating"]}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique de stabilité sauvegardé: {output_path}")

def plot_post_threshold_analysis(transition_analysis, output_path):
    """Génère l'analyse post-seuil avec dérivée première et seconde."""
    if not transition_analysis:
        print("⚠ Aucune donnée post-seuil disponible")
        return
    
    metrics = ['Higuchi', 'DFA', 'Katz', 'Hurst', 'SampleEn']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Analyse Post-Critique - Dérivée Première (Transition Brutale)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        if metric in transition_analysis:
            flux_data = transition_analysis[metric]
            flux_labels = []
            first_derivs = []
            second_derivs = []
            signs = []
            
            for flux_label, data in flux_data.items():
                if data.get('status') == 'success':
                    flux_labels.append(flux_label)
                    first_derivs.append(data['first_derivative'])
                    second_derivs.append(data['second_derivative'])
                    signs.append(data['sign'])
            
            if flux_labels:
                # Afficher la dérivée première comme métrique principale
                colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'gray' for s in signs]
                bars = ax.bar(range(len(flux_labels)), first_derivs, color=colors, alpha=0.7)
                ax.set_xticks(range(len(flux_labels)))
                ax.set_xticklabels(flux_labels, rotation=45, ha='right', fontsize=8)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # Ajouter les valeurs de dérivée seconde comme annotations
                for i, (bar, second_deriv) in enumerate(zip(bars, second_derivs)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'f\'\'={second_deriv:.3f}',
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=7, color='blue')
            else:
                ax.text(0.5, 0.5, 'Données insuffisantes', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_ylabel("Dérivée Première (f')")
        ax.set_title(f'{metric}')
        ax.grid(True, alpha=0.3, axis='y')
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique post-seuil sauvegardé: {output_path}")

def plot_noise_diagnostic(noise_diag_res, output_path):
    """Génère le graphique du diagnostic de bruit multi-flux."""
    if not noise_diag_res:
        print("⚠ Aucune donnée de diagnostic bruit disponible")
        return
    
    all_diagnostics = noise_diag_res.get('all_flux_diagnostics', [])
    if not all_diagnostics:
        print("⚠ Aucun diagnostic de flux disponible")
        return
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Title with verdict
    verdict = noise_diag_res.get('diagnostic_label', 'Inconnu')
    avg_score = noise_diag_res.get('avg_variance_proxy_score', 0)
    fig.suptitle(f'Diagnostic Multi-Flux - {verdict}\nScore moyen: {avg_score:.3f}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Variance Proxy Score par Flux (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :])
    flux_names = [d['flux_name'] for d in all_diagnostics]
    scores = [d['variance_proxy_score'] for d in all_diagnostics]
    colors = ['green' if s < 0.3 else 'orange' if s < 0.6 else 'red' for s in scores]
    ax1.bar(range(len(flux_names)), scores, color=colors, alpha=0.7)
    ax1.axhline(y=0.4, color='blue', linestyle='--', label='Seuil Proxy (0.4)')
    ax1.set_xticks(range(len(flux_names)))
    ax1.set_xticklabels(flux_names, rotation=45, ha='right')
    ax1.set_ylabel('Score Proxy de Variance (0-1)')
    ax1.set_title('Score de Proxy de Variance par Flux')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Q5 Score Comparison (Top 5 fluxes)
    ax2 = fig.add_subplot(gs[1, 0])
    top_5 = sorted(all_diagnostics, key=lambda x: x['variance_proxy_score'], reverse=True)[:5]
    if top_5:
        flux_names_5 = [d['flux_name'] for d in top_5]
        cand_scores = [d['cand_q5_score'] for d in top_5]
        white_scores = [d['white_q5_score'] for d in top_5]
        x = np.arange(len(flux_names_5))
        width = 0.35
        ax2.bar(x - width/2, cand_scores, width, label='Candidat', color='steelblue', alpha=0.7)
        ax2.bar(x + width/2, white_scores, width, label='Bruit Blanc', color='orange', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(flux_names_5, rotation=45, ha='right')
        ax2.set_ylabel('Score Q5')
        ax2.set_title('Top 5: Comparaison Q5 vs Bruit Blanc')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Hurst Exponent Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    hurst_max = [d['max_hurst'] for d in all_diagnostics]
    hurst_avg = [d['avg_hurst'] for d in all_diagnostics]
    x = np.arange(len(flux_names))
    ax3.scatter(x, hurst_max, label='Max Hurst', color='red', s=100, alpha=0.7)
    ax3.scatter(x, hurst_avg, label='Avg Hurst', color='blue', s=100, alpha=0.7)
    ax3.axhline(y=0.5, color='green', linestyle='--', label='Bruit Blanc (0.5)')
    ax3.axhline(y=0.6, color='orange', linestyle=':', label='Limite Mémoire (0.6)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(flux_names, rotation=45, ha='right')
    ax3.set_ylabel('Exposant de Hurst')
    ax3.set_title('Distribution de Hurst par Flux')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation Delta
    ax4 = fig.add_subplot(gs[2, 0])
    delta_corrs = [d['delta_corr'] for d in all_diagnostics]
    colors_corr = ['green' if d < 0.1 else 'orange' if d < 0.3 else 'red' for d in delta_corrs]
    ax4.bar(range(len(flux_names)), delta_corrs, color=colors_corr, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xticks(range(len(flux_names)))
    ax4.set_xticklabels(flux_names, rotation=45, ha='right')
    ax4.set_ylabel('Delta Corrélation Volatilité')
    ax4.set_title('Changement de Corrélation Post-Seuil')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Spectral Slope
    ax5 = fig.add_subplot(gs[2, 1])
    slopes = [d['slope_after'] for d in all_diagnostics]
    colors_slope = ['green' if s > -0.2 else 'orange' if s > -0.5 else 'red' for s in slopes]
    ax5.bar(range(len(flux_names)), slopes, color=colors_slope, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.axhline(y=-0.3, color='orange', linestyle='--', label='Limite CSD (-0.3)')
    ax5.set_xticks(range(len(flux_names)))
    ax5.set_xticklabels(flux_names, rotation=45, ha='right')
    ax5.set_ylabel('Pente Spectrale Post-Seuil')
    ax5.set_title('Pente Spectrale FFT par Flux')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Radar Chart des composantes (Top 5)
    ax6 = fig.add_subplot(gs[3, :], projection='polar')
    if top_5:
        categories = ['Q5', 'Corrélation', 'Mémoire', 'Spectral']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        for i, diag in enumerate(top_5):
            values = []
            # Q5 component (normalized)
            q5_norm = min(diag['cand_q5_score'] / (3 * max(diag['white_q5_score'], 0.01)), 1.0)
            values.append(q5_norm)
            # Correlation component
            corr_norm = min(diag['delta_corr'] / 0.4, 1.0)
            values.append(corr_norm)
            # Memory component
            mem_norm = min((diag['max_hurst'] - 0.5) / 0.2, 1.0)
            values.append(mem_norm)
            # Spectral component
            spec_norm = min(abs(diag['slope_after']) / 0.5, 1.0)
            values.append(spec_norm)
            
            values += values[:1]
            ax6.plot(angles, values, 'o-', linewidth=2, label=diag['flux_name'])
            ax6.fill(angles, values, alpha=0.1)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('Radar des Composantes CSD (Top 5)', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique diagnostic bruit sauvegardé: {output_path}")

def export_csv_results(df_feat, scores, noise_diag_res, output_path):
    """Exporte les résultats numériques complets en CSV."""
    try:
        # Remplacer NaN par None pour éviter les problèmes d'export
        df_export = df_feat.copy()
        for col in df_export.columns:
            if df_export[col].isna().all():
                df_export[col] = 0.0
        
        # Export landscape data avec tous les estimateurs et flux
        df_export.to_csv(output_path, index=False)
        print(f"✓ Résultats numériques exportés: {output_path}")
        
        # Créer un fichier CSV supplémentaire pour les scores de séparation
        scores_path = str(output_path).replace('.csv', '_separation_scores.csv')
        scores_df = pd.DataFrame(scores).T
        scores_df = scores_df.fillna(0.0)  # Remplacer NaN par 0.0
        scores_df.to_csv(scores_path, index=True)
        print(f"✓ Scores de séparation exportés: {scores_path}")
        
        # Créer un fichier CSV pour le diagnostic de bruit multi-flux
        if noise_diag_res and 'all_flux_diagnostics' in noise_diag_res:
            noise_path = str(output_path).replace('.csv', '_noise_diagnostic.csv')
            noise_df = pd.DataFrame(noise_diag_res['all_flux_diagnostics'])
            noise_df = noise_df.fillna(0.0)  # Remplacer NaN par 0.0
            noise_df.to_csv(noise_path, index=False)
            print(f"✓ Diagnostic bruit multi-flux exporté: {noise_path}")
            
    except Exception as e:
        print(f"✗ Erreur lors de l'export CSV: {e}")

def compute_csd_correlation_analysis(noise_diag_res, df_feat, fluxes):
    """Point d'entrée public de l'analyse CSD.

    La version précédente ne contenait qu'une docstring et renvoyait donc
    implicitement ``None``. L'implémentation effective est centralisée dans
    ``compute_csd_estimator_correlations`` juste ci-dessous.
    """
    return compute_csd_estimator_correlations(noise_diag_res, df_feat, fluxes)
def compute_csd_estimator_correlations(noise_diag_res, df_feat, fluxes):
    """
    Analyse de corrélation pour les flux CSD identifiés directement avec leur dérivée première (f').
    Évalue la relation entre le niveau de signal/complexité des CSD et leur vitesse/pente (f').
    """
    try:
        if not noise_diag_res or 'all_flux_diagnostics' not in noise_diag_res:
            return None
        
        all_diagnostics = noise_diag_res['all_flux_diagnostics']
        
        # Sélectionner les flux identifiés comme CSD (score >= 0.4) ou les top proxies si aucun >= 0.4
        csd_fluxes = [d['flux_name'] for d in all_diagnostics if d['variance_proxy_score'] >= 0.4]
        if not csd_fluxes and all_diagnostics:
            sorted_diag = sorted(all_diagnostics, key=lambda x: x['variance_proxy_score'], reverse=True)
            csd_fluxes = [sorted_diag[0]['flux_name']]
        
        if not csd_fluxes:
            return {
                'csd_count': 0,
                'message': 'Aucun flux CSD identifié'
            }
        
        csd_correlations = {}
        metrics = list(ESTIMATOR_NAMES)
        
        for csd_flux in csd_fluxes:
            flux_data = np.asarray(fluxes[csd_flux], dtype=float)
            if np.std(flux_data) > 1e-10:
                flux_zscore = (flux_data - np.mean(flux_data)) / np.std(flux_data)
            else:
                flux_zscore = np.zeros_like(flux_data)
            
            # Dérivée première du signal (f')
            f_prime_signal = np.diff(flux_zscore)
            
            # 1. Corrélation entre le signal CSD et sa propre dérivée première f'
            min_len_sig = min(len(flux_zscore[:-1]), len(f_prime_signal))
            if min_len_sig >= 10:
                corr_sig_fprime, p_sig_fprime = stats.pearsonr(flux_zscore[:min_len_sig], f_prime_signal[:min_len_sig])
                if np.isnan(corr_sig_fprime):
                    corr_sig_fprime, p_sig_fprime = 0.0, 1.0
            else:
                corr_sig_fprime, p_sig_fprime = 0.0, 1.0
            
            # 2. Corrélation entre les estimateurs de complexité CSD et la dérivée première f'
            metric_fprime_corrs = {}
            for metric in metrics:
                _col = f"{csd_flux}_{metric}_mean"
                cols = [_col] if _col in df_feat.columns else []
                if not cols:
                    continue
                
                metric_vals = df_feat[cols[0]].dropna().values
                if len(metric_vals) < 10:
                    continue
                
                # Dérivée première de l'estimateur de complexité (f'_est)
                f_prime_est = np.diff(metric_vals)
                
                min_len_est = min(len(metric_vals[:-1]), len(f_prime_est))
                if min_len_est >= 5:
                    corr_est_fprime, p_est_fprime = stats.pearsonr(metric_vals[:min_len_est], f_prime_est[:min_len_est])
                    if np.isnan(corr_est_fprime):
                        corr_est_fprime, p_est_fprime = 0.0, 1.0
                else:
                    corr_est_fprime, p_est_fprime = 0.0, 1.0
                
                metric_fprime_corrs[metric] = {
                    'estimator_fprime_corr': float(corr_est_fprime),
                    'p_value': float(p_est_fprime),
                    'significant': p_est_fprime < 0.05 and abs(corr_est_fprime) > 0.3
                }
            
            csd_correlations[csd_flux] = {
                'signal_fprime_correlation': float(corr_sig_fprime),
                'signal_fprime_p_value': float(p_sig_fprime),
                'signal_fprime_significant': p_sig_fprime < 0.05 and abs(corr_sig_fprime) > 0.3,
                'metric_fprime_correlations': metric_fprime_corrs
            }
        
        all_sig_corrs = [v['signal_fprime_correlation'] for v in csd_correlations.values()]
        avg_sig_fprime_corr = float(np.mean(all_sig_corrs)) if all_sig_corrs else 0.0
        
        return {
            'csd_fluxes': csd_fluxes,
            'csd_count': len(csd_fluxes),
            'csd_correlations': csd_correlations,
            'statistics': {
                'avg_signal_fprime_correlation': avg_sig_fprime_corr,
                'total_csd_tested': len(csd_fluxes)
            }
        }
    except Exception as e:
        return {
            'error': str(e),
            'message': 'Erreur lors de l\'analyse de corrélation CSD avec f\''
        }

def plot_csd_correlation_analysis(csd_analysis, output_path):
    """
    Génère le graphique d'analyse de corrélation des flux CSD avec leur dérivée première (f').
    """
    if not csd_analysis or 'csd_correlations' not in csd_analysis:
        print("⚠ Aucune donnée de corrélation CSD vs f' disponible")
        return
    
    csd_correlations = csd_analysis['csd_correlations']
    if not csd_correlations:
        print("⚠ Aucun résultat de corrélation CSD vs f' disponible")
        return
    
    csd_fluxes = list(csd_correlations.keys())
    metrics = [m for m in ESTIMATOR_NAMES if any(m in csd_correlations[f].get('metric_fprime_correlations', {}) for f in csd_fluxes)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analyse de Corrélation CSD vs Dérivée Première (f\')', fontsize=16, fontweight='bold')
    
    # Plot 1: Corrélation Signal CSD vs Signal f'
    ax1 = axes[0]
    sig_corrs = [csd_correlations[f]['signal_fprime_correlation'] for f in csd_fluxes]
    colors = ['green' if abs(c) > 0.3 else 'steelblue' for c in sig_corrs]
    ax1.bar(range(len(csd_fluxes)), sig_corrs, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.3, color='red', linestyle='--', label='Seuil Significatif (0.3)')
    ax1.axhline(y=-0.3, color='red', linestyle='--')
    ax1.set_xticks(range(len(csd_fluxes)))
    ax1.set_xticklabels(csd_fluxes, rotation=45, ha='right')
    ax1.set_ylabel('Corrélation r(Signal CSD, f\')')
    ax1.set_title('Corrélation du Signal CSD avec sa Dérivée Première (f\')')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Heatmap des corrélations des estimateurs CSD vs f'
    ax2 = axes[1]
    matrix = np.zeros((len(csd_fluxes), len(metrics)))
    for i, flux in enumerate(csd_fluxes):
        metric_data = csd_correlations[flux].get('metric_fprime_correlations', {})
        for j, metric in enumerate(metrics):
            if metric in metric_data:
                matrix[i, j] = metric_data[metric]['estimator_fprime_corr']
    
    im = ax2.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_yticks(range(len(csd_fluxes)))
    ax2.set_xticklabels(metrics, rotation=45, ha='right')
    ax2.set_yticklabels(csd_fluxes)
    ax2.set_title('Corrélation des Estimateurs CSD avec f\'')
    plt.colorbar(im, ax=ax2, label='Corrélation r')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique de corrélation CSD vs f' sauvegardé: {output_path}")


def export_text_report(results, output_path):
    """Génère un rapport textuel structuré."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ANALYSE DE COMPLEXITÉ - UNIVERSAL LOG v3.4-H3\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write("MÉTADONNÉES\n")
            f.write("-" * 40 + "\n")
            metadata = results.get('metadata', {})
            f.write(f"Domaine: {metadata.get('domain', 'N/A')}\n")
            f.write(f"Mode: {metadata.get('mode', 'N/A')}\n")
            f.write(f"Test Statistique: {metadata.get('stat_test', 'N/A')}\n")
            f.write(f"Taille de Fenêtre: {metadata.get('w_size', 'N/A')}\n")
            f.write(f"Nombre de Points: {metadata.get('n_pts', 'N/A')}\n")
            f.write(f"Seuil Critique (utilisé): {metadata.get('opt_threshold', 'N/A')}%\n\n")
            
            # Optimisation Monte Carlo
            threshold_opt = metadata.get('threshold_optimization')
            if threshold_opt and threshold_opt.get('best_config'):
                f.write("OPTIMISATION MONTE CARLO DES PARAMÈTRES DE SEUIL\n")
                f.write("-" * 40 + "\n")
                best_config = threshold_opt['best_config']
                f.write(f"Itérations: {threshold_opt.get('n_iterations', 'N/A')}\n")
                f.write(f"Meilleur score: {best_config.get('score', 'N/A'):.3f}\n")
                
                f.write("\nDONNÉES TRAITÉES:\n")
                f.write(f"  Seuil Monte Carlo: {best_config.get('threshold', 'N/A'):.1f}%\n")
                f.write(f"  Seuil par défaut: {threshold_opt.get('default_threshold', 'N/A'):.1f}%\n")
                
                f.write("\nDONNÉES BRUTES:\n")
                raw_mc_threshold = threshold_opt.get('raw_mc_threshold')
                raw_default_threshold = threshold_opt.get('raw_default_threshold')
                if raw_mc_threshold is not None:
                    f.write(f"  Seuil Monte Carlo: {raw_mc_threshold:.1f}%\n")
                if raw_default_threshold is not None:
                    f.write(f"  Seuil par défaut: {raw_default_threshold:.1f}%\n")
                
                f.write(f"\nParamètres optimisés (données traitées):\n")
                f.write(f"  n_candidates: {best_config.get('n_candidates', 'N/A')}\n")
                f.write(f"  resolution: {best_config.get('resolution', 'N/A'):.3f}%\n")
                f.write(f"  w_size: {best_config.get('w_size', 'N/A')}\n")
                f.write(f"  min_grp: {best_config.get('min_grp', 'N/A')}\n")
                bounds = threshold_opt.get('loupe_bounds') or {}
                if bounds:
                    f.write(f"  loupe autorisée: {bounds.get('lo', 'N/A')} à {bounds.get('hi', 'N/A')} seuils\n")
                rigid = threshold_opt.get('rigid_1001_threshold')
                if rigid is not None and not (isinstance(rigid, float) and np.isnan(rigid)):
                    f.write(f"  témoin grille 1001 (même fenêtre): {rigid:.1f}%\n")
                f.write(
                    "\n  Note: 1001 n'est plus la grille par défaut. "
                    "Le témoin sert seulement à mesurer l'effet de l'ancienne constante.\n\n"
                )
            
            # v3.4 : repli de seuil, JAMAIS silencieux dans le rapport.
            # En v3.3, si best_config etait absent, la section Monte Carlo
            # n'ecrivait rien et le lecteur ne voyait pas le repli.
            if threshold_opt:
                _ts = threshold_opt.get('threshold_source')
                _warns = threshold_opt.get('warnings') or []
                if _ts and _ts != 'monte_carlo':
                    f.write("ALERTE REPLI DE SEUIL\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Source du seuil utilise : {_ts} (PAS Monte Carlo).\n")
                    f.write(f"Seuil utilise : {metadata.get('opt_threshold', 'N/A')}%\n")
                    f.write(f"Seuil par defaut : {threshold_opt.get('default_threshold', 'N/A')}%\n\n")
                for _w in _warns:
                    f.write(f"AVERTISSEMENT SEUIL : {_w}\n")
                if _warns:
                    f.write("\n")

            # Top signatures
            f.write("SIGNATURES DOMINANTES\n")
            f.write("-" * 40 + "\n")
            top_sigs = results.get('top_signatures', {})
            for q_name, sigs in top_sigs.items():
                f.write(f"\n{q_name}:\n")
                for sig in sigs:
                    f.write(f"  - {sig['feature']}: {sig['score']:.3f}\n")
            
            f.write("\n\nDIAGNOSTIC DE BRUIT\n")
            f.write("-" * 40 + "\n")
            noise_diag = results.get('noise_diagnostic', {})
            f.write(f"Verdict: {noise_diag.get('diagnostic_label', 'N/A')}\n")
            f.write(f"Score moyen proxy: {noise_diag.get('avg_variance_proxy_score', 'N/A')}\n")
            f.write(f"\nRésumé:\n{noise_diag.get('summary_text', 'N/A')}\n")
            
            # Analyse CSD : corrélation de chaque flux CSD avec sa dérivée et
            # avec les dérivées de ses estimateurs de complexité.
            csd_analysis = results.get('csd_correlation_analysis')
            if csd_analysis and 'statistics' in csd_analysis:
                f.write("\n\nANALYSE DE CORRÉLATION CSD vs DÉRIVÉE PREMIÈRE\n")
                f.write("-" * 40 + "\n")
                csd_stats = csd_analysis['statistics']
                f.write(f"Flux CSD analysés: {csd_analysis.get('csd_count', 0)}\n")
                f.write(
                    "Corrélation moyenne signal–dérivée: "
                    f"{csd_stats.get('avg_signal_fprime_correlation', 0):.3f}\n\n"
                )
                for flux_name, flux_result in csd_analysis.get('csd_correlations', {}).items():
                    f.write(f"{flux_name}:\n")
                    f.write(
                        "  Signal vs dérivée: "
                        f"r={flux_result.get('signal_fprime_correlation', 0):.3f}, "
                        f"p={flux_result.get('signal_fprime_p_value', 1):.4f}\n"
                    )
                    for metric, metric_result in flux_result.get(
                        'metric_fprime_correlations', {}
                    ).items():
                        f.write(
                            f"  {metric} vs sa dérivée: "
                            f"r={metric_result.get('estimator_fprime_corr', 0):.3f}, "
                            f"p={metric_result.get('p_value', 1):.4f}\n"
                        )
                    f.write("\n")
            
            # Normalization Diagnostic
            norm_diagnostic = results.get('normalization_diagnostic')
            if norm_diagnostic and 'error' not in norm_diagnostic:
                f.write("\n\nDIAGNOSTIC DE NORMALISATION\n")
                f.write("-" * 40 + "\n")
                for flux_name, flux_data in norm_diagnostic.items():
                    if isinstance(flux_data, dict) and 'verdict' in flux_data:
                        f.write(f"\n{flux_name}:\n")
                        f.write(f"  Divergence moyenne: {flux_data.get('avg_divergence_pct', 'N/A')}%\n")
                        f.write(f"  Verdict: {flux_data.get('verdict', 'N/A')}\n")
            
            # Stability
            f.write("\n\nSTABILITÉ DU SEUIL\n")
            f.write("-" * 40 + "\n")
            stability = metadata.get('stability_report')
            if stability:
                f.write(f"Note: {stability.get('rating', 'N/A')}\n")
                f.write(f"Moyenne: {stability.get('mean', 'N/A')}%\n")
                f.write(f"Écart-type: {stability.get('std', 'N/A')}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Rapport généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Rapport textuel exporté: {output_path}")
    except Exception as e:
        print(f"✗ Erreur lors de l'export du rapport: {e}")

def export_json_results(results, output_path):
    """Exporte les résultats bruts en JSON."""
    try:
        def replace_nan_with_none(obj):
            if isinstance(obj, dict):
                return {k: replace_nan_with_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan_with_none(v) for v in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, tuple):
                return [replace_nan_with_none(v) for v in obj]
            elif isinstance(obj, (set, frozenset)):
                return [replace_nan_with_none(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return [replace_nan_with_none(v) for v in obj.tolist()]
            return obj
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(replace_nan_with_none(results), f, indent=2, ensure_ascii=False, allow_nan=False)
        
        print(f"✓ Résultats JSON exportés: {output_path}")
    except Exception as e:
        print(f"✗ Erreur lors de l'export JSON: {e}")

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================
def run_pipeline(data_source, custom_file_path, mapping, mode, opt_cfg):
    """Exécute le pipeline d'analyse (adapté du code original)."""
    if opt_cfg is None:
        opt_cfg = {}
        
    modes = {
        'fast': {
            'n_est': 1,
            'w_size': 80,
            'test': 'cohen_d',
            'min_grp': 2,
            'min_len': 50,
            'boot': False,
            'window_scan': None,
            'full_triangulation': False
        },
        'robust': {
            'n_est': 5,
            'w_size': 150,
            'test': 'cliff_delta',
            'min_grp': 3,
            'min_len': 100,
            'boot': True,
            'window_scan': None,
            'full_triangulation': False
        },
        'stable': {
            'n_est': N_ESTIMATORS,
            'w_size': WINDOW_SIZE,
            'test': STATISTICAL_TEST,
            'min_grp': MIN_GROUP_SIZE,
            'min_len': MIN_SERIES_LENGTH,
            'boot': ENABLE_BOOTSTRAP,
            'window_scan': [50, 80, 100, 120, 150, 180, 220],
            'full_triangulation': FULL_TRIANGULATION
        }
    }
    
    mode_cfg = modes.get(mode, modes['fast']).copy()
    mode_cfg.setdefault('random_seed', RANDOM_SEED)
    for k, v in opt_cfg.items():
        if v is not None:
            mode_cfg[k] = v
            
    # Load dataset
    df = None
    fluxes = {}
    progress = None
    meta_info = {}
    
    target_path = custom_file_path
    if target_path is None:
        candidates = ['test_user.csv', 'test_user_full.csv', 'test_large.csv']
        for cand in candidates:
            if os.path.exists(cand):
                target_path = cand
                break
                
    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(f"Aucun fichier CSV trouvé. Veuillez spécifier un fichier avec --input")
        
    df = pd.read_csv(target_path)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # Parse column mappings
    if mapping is None:
        # Utiliser la colonne de contrôle configurée ou la première colonne numérique
        if DEFAULT_CONTROL_COLUMN and DEFAULT_CONTROL_COLUMN in df.columns:
            control_col = DEFAULT_CONTROL_COLUMN
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            control_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        mapping = {
            'control_col': control_col,
            'flux_mapping': {}
        }
        
    control_col = mapping.get('control_col')
    if not control_col or control_col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            control_col = num_cols[0]
        else:
            control_col = df.columns[0]
            
    df[control_col] = pd.to_numeric(df[control_col], errors='coerce')
    df = df.dropna(subset=[control_col]).reset_index(drop=True)
    control_series = df[control_col].values
    min_required_points = max(
        20,
        min(mode_cfg.get('min_len', 50), mode_cfg.get('w_size', 80)),
    )
    if len(control_series) < min_required_points:
        raise ValueError(
            f"Série trop courte : {len(control_series)} points valides. "
            f"Au moins {min_required_points} sont requis avec la configuration actuelle."
        )
    progress = 100.0 * stats.rankdata(control_series, method='average') / len(control_series)
    n_pts = len(progress)
    
    # Build fluxes
    flux_map = mapping.get('flux_mapping', {})
    for f_name, cols in flux_map.items():
        if isinstance(cols, list):
            valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                df[f_name] = df[valid_cols].mean(axis=1)
                fluxes[f_name] = df[f_name].values.astype(float)
        else:
            if cols in df.columns:
                fluxes[f_name] = df[cols].values.astype(float)
                
    if not fluxes:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != control_col][:50]
        for idx, col in enumerate(num_cols):
            fluxes[col] = df[col].values.astype(float)
    
    # Appliquer le z-score à tous les flux pour la cohérence du pipeline
    # (scan de seuil, estimateurs, questions, diagnostic de bruit)
    for k in fluxes.keys():
        s = fluxes[k]
        nan_mask = np.isnan(s)
        if np.any(nan_mask):
            s = np.where(nan_mask, np.nanmean(s), s)
        if np.std(s) > 1e-10:
            fluxes[k] = (s - np.mean(s)) / np.std(s)
        else:
            fluxes[k] = np.zeros_like(s)
            
    fluxes_ui = {}
    for k in list(fluxes.keys()):
        s = fluxes[k]
        nan_mask = np.isnan(s)
        if np.any(nan_mask):
            s = np.where(nan_mask, np.nanmean(s), s)
            fluxes[k] = s
        s_min, s_max = np.nanmin(s), np.nanmax(s)
        if s_max == s_min:
            fluxes_ui[k] = np.zeros_like(s)
        else:
            fluxes_ui[k] = np.clip((s - s_min) / (s_max - s_min), 0.0, 1.0)
            
    meta_info = {
        'control_col': control_col,
        'aggressive_fluxes': mapping.get('aggressive_fluxes', list(fluxes.keys())[:2]),
        'active_keys': mapping.get('active_keys', list(fluxes.keys())[:2]),
        'reactive_keys': mapping.get('reactive_keys', list(fluxes.keys())[2:4])
    }
        
    if not fluxes:
        raise ValueError("Aucun flux analytique extrait. Vérifiez les colonnes de données.")
        
    if mapping:
        for k in ['aggressive_fluxes', 'active_keys', 'reactive_keys']:
            if k in mapping and mapping[k]:
                meta_info[k] = mapping[k]

    flux_columns_raw = {}
    for k, v in fluxes.items():
        flux_columns_raw[k] = [v]

    if data_source == 'custom' and 'flux_mapping' in mapping:
        flux_map = mapping.get('flux_mapping', {})
        for f_name, cols in flux_map.items():
            if isinstance(cols, list):
                valid_cols = [c for c in cols if c in df.columns]
                if valid_cols:
                    flux_columns_raw[f_name] = [df[c].values.astype(float) for c in valid_cols]

    if mode_cfg.get('full_triangulation', False):
        print(
            "ATTENTION : triangulation synthétique activée. Les colonnes "
            "générées servent uniquement à une analyse de sensibilité."
        )
        rng = np.random.default_rng(mode_cfg.get('random_seed', RANDOM_SEED))
        for k, v in list(fluxes.items()):
            if k not in flux_columns_raw or len(flux_columns_raw[k]) <= 1:
                col1 = v
                col2 = np.clip(v + rng.normal(0, 0.05, len(v)), 0.0, 1.0)
                col3 = np.clip(v + rng.normal(0, 0.05, len(v)), 0.0, 1.0)
                flux_columns_raw[k] = [col1, col2, col3]

    # Optimisation Monte Carlo des paramètres de seuil (activée par défaut)
    threshold_optimization = None
    default_threshold = None
    default_scan_curve = None
    
    # Scan avec paramètres par défaut sur données traitées (pour comparaison)
    default_threshold, default_scan_curve = scan_critical_threshold(fluxes, progress, mode_cfg)
    
    # Scan sur données brutes (colonnes numériques uniquement)
    raw_fluxes = {}
    for col in df.columns:
        if col != control_col and pd.api.types.is_numeric_dtype(df[col]):
            raw_fluxes[col] = df[col].values.astype(float)
    
    raw_default_threshold = None
    # v3.4 -- journal des replis de seuil (voir changelog en tete).
    # Chaque repli est imprime, journalise ET enregistre ici, pour
    # qu'aucun seuil de repli ne passe pour un seuil Monte Carlo.
    raw_optimization_diagnostics = None
    raw_threshold_source = 'none'
    threshold_warnings = []

    def _record_threshold_warning(msg):
        threshold_warnings.append(msg)
        print(f"⚠ {msg}")
        print(f"⚠ {msg}", file=sys.stderr)
        logging.warning(msg)

    raw_default_scan_curve = None
    raw_mc_threshold = None
    raw_mc_scan_curve = None
    raw_best_config = None
    
    if raw_fluxes:
        # Scan par défaut sur données brutes
        raw_default_threshold, raw_default_scan_curve = scan_critical_threshold(raw_fluxes, progress, mode_cfg)

        # Optimisation Monte Carlo sur données brutes
        if mode_cfg.get('auto_optimize_threshold', True):
            n_iterations = mode_cfg.get('optimization_iterations', 50)
            (raw_best_config, raw_mc_scan_curve, raw_optimization_results,
             raw_optimization_diagnostics) = optimize_threshold_scan_parameters_monte_carlo(
                raw_fluxes, progress, mode_cfg, n_iterations
            )
            if raw_best_config is not None:
                raw_mc_threshold = raw_best_config['threshold']
                raw_threshold_source = 'monte_carlo'
            else:
                # v3.4 : ce cas etait SILENCIEUX en v3.3 (raw_mc_threshold
                # restait None sans explication). Il est desormais alerte.
                raw_threshold_source = 'default_fallback_mc_failed'
                _record_threshold_warning(
                    "REPLI SEUIL (donnees brutes) : Monte Carlo sans "
                    "configuration valide ; seuil brut Monte Carlo absent, "
                    f"seuil brut par defaut conserve ({raw_default_threshold}). "
                    f"Diagnostics : {raw_optimization_diagnostics}.")
        else:
            raw_threshold_source = 'default_config'

    if mode_cfg.get('auto_optimize_threshold', True):
        n_iterations = mode_cfg.get('optimization_iterations', 50)  # 50 par défaut pour équilibre temps/précision
        (best_config, best_scan_curve, optimization_results,
         optimization_diagnostics) = optimize_threshold_scan_parameters_monte_carlo(
            fluxes, progress, mode_cfg, n_iterations
        )
        threshold_optimization = {
            'best_config': best_config,
            'all_results': optimization_results,
            'n_iterations': n_iterations,
            'default_threshold': default_threshold,
            'default_scan_curve': default_scan_curve,
            'raw_default_threshold': raw_default_threshold,
            'raw_default_scan_curve': raw_default_scan_curve,
            'raw_mc_threshold': raw_mc_threshold,
            'raw_mc_scan_curve': raw_mc_scan_curve,
            'raw_best_config': raw_best_config,
            'optimization_diagnostics': optimization_diagnostics,
            'raw_optimization_diagnostics': raw_optimization_diagnostics,
            'raw_threshold_source': raw_threshold_source,
            'scan_fast_fallback_log': list(SCAN_FAST_FALLBACK_LOG),
        }

        # Utiliser directement le meilleur scan lorsqu'il existe. Sinon, le
        # scan par défaut reste disponible et le pipeline ne s'arrête pas.
        # v3.4 : ce repli est JOURNALISE et MARQUE, pas seulement imprime.
        if best_config is not None:
            threshold_source = 'monte_carlo'
            main_threshold = best_config['threshold']
            scan_curve = best_scan_curve
            mode_cfg['loupe_candidates'] = int(best_config['n_candidates'])
            # Témoin : la même fenêtre, mais la grille historique à 1001.
            # Sert à voir si la constante change le seuil. Ce n'est pas la loupe retenue.
            rigid_threshold, rigid_curve = scan_critical_threshold(
                fluxes, progress, mode_cfg,
                1001, None, best_config['w_size'], best_config['min_grp'],
            )
            threshold_optimization['rigid_1001_threshold'] = rigid_threshold
            threshold_optimization['rigid_1001_curve'] = rigid_curve
            lo_b, hi_b = loupe_candidate_bounds(n_pts - best_config['w_size'] + 1)
            threshold_optimization['loupe_bounds'] = {'lo': lo_b, 'hi': hi_b}
        else:
            threshold_source = 'default_fallback_mc_failed'
            _record_threshold_warning(
                "REPLI SEUIL (donnees traitees) : Optimisation Monte Carlo "
                "sans configuration valide ; seuil par defaut utilise a la "
                f"place du seuil Monte Carlo (defaut = {default_threshold}). "
                f"Diagnostics : {optimization_diagnostics}. Ce seuil NE DOIT "
                "PAS etre lu comme un seuil Monte Carlo.")
            main_threshold = default_threshold
            scan_curve = default_scan_curve
    else:
        # Choix explicite de configuration (pas un echec) : on le marque
        # quand meme, pour ne pas le confondre avec un seuil Monte Carlo.
        threshold_source = 'default_config'
        threshold_optimization = {
            'best_config': None,
            'default_threshold': default_threshold,
            'default_scan_curve': default_scan_curve,
            'raw_default_threshold': raw_default_threshold,
            'raw_default_scan_curve': raw_default_scan_curve,
            'raw_mc_threshold': raw_mc_threshold,
            'raw_mc_scan_curve': raw_mc_scan_curve,
            'raw_best_config': raw_best_config,
            'raw_threshold_source': raw_threshold_source,
            'scan_fast_fallback_log': list(SCAN_FAST_FALLBACK_LOG),
        }
        main_threshold = default_threshold
        scan_curve = default_scan_curve
    threshold_optimization['threshold_source'] = threshold_source
    # Seuil par defaut lui-meme invalide (NaN) : pas de seuil utilisable.
    # On ne fabrique pas un seuil, on le signale comme erreur.
    if main_threshold is None or (isinstance(main_threshold, float) and np.isnan(main_threshold)):
        _record_threshold_warning(
            "ERREUR SEUIL : ni le Monte Carlo ni le scan par defaut n'ont "
            "produit de seuil valide (main_threshold = NaN). Les analyses "
            "post-seuil seront ignorees et le diagnostic bruit utilisera "
            "la valeur d'urgence 52,5 % (voir son propre avertissement).")
    threshold_optimization['warnings'] = list(threshold_warnings)
    
    # Compute Threshold Stability
    stability_report = None
    if mode == 'stable':
        window_scan = mode_cfg.get('window_scan', [50, 80, 100, 120, 150, 180, 220])
        stability_report = compute_threshold_stability(fluxes, progress, window_scan, mode_cfg)
        
    # Build Regime Signatures Landscape
    scan_points = np.linspace(progress.min(), progress.max(), 30)
    n_pts = len(progress)
    # Utiliser le w_size optimisé par Monte Carlo si disponible, sinon la valeur par défaut
    if mode_cfg.get('auto_optimize_threshold', True) and threshold_optimization and threshold_optimization.get('best_config'):
        w_size = threshold_optimization['best_config'].get('w_size', mode_cfg.get('w_size', 80))
        min_grp = threshold_optimization['best_config'].get('min_grp', mode_cfg.get('min_grp', 3))
    else:
        w_size = mode_cfg.get('w_size', 80)
        min_grp = mode_cfg.get('min_grp', 3)
    w_size = max(2, min(w_size, n_pts // 2))
    tol = max(3.0, (scan_points[1] - scan_points[0]) / 2.0)
    
    sort_idx = np.argsort(progress)
    prog_sorted = progress[sort_idx]
    flux_sorted = {k: v[sort_idx] for k, v in fluxes.items()}
    
    flux_sorted_multi = {}
    for k, col_arrs in flux_columns_raw.items():
        flux_sorted_multi[k] = [arr[sort_idx] for arr in col_arrs]
    
    window_data = []
    for i in range(n_pts - w_size + 1):
        window_data.append({
            'progress': float(np.mean(prog_sorted[i : i + w_size])),
            'fluxes': {k: flux_sorted[k][i : i + w_size] for k in fluxes.keys()},
            'fluxes_multi': {k: [arr[i : i + w_size] for arr in arrs] for k, arrs in flux_sorted_multi.items()}
        })
        
    rows = []
    n_est = mode_cfg.get('n_est', 1)
    boot = mode_cfg.get('boot', False)
    
    for P in scan_points:
        matching = [w for w in window_data if abs(w['progress'] - P) <= tol]
        if not matching:
            matching = [min(window_data, key=lambda w: abs(w['progress'] - P))]
            
        row = {'Progress': float(P), 'N_cohorts': len(matching)}
        
        # Ajuster min_len dynamiquement basé sur w_size pour éviter les estimateurs à 0
        dynamic_min_len = max(10, w_size // 2)
        mode_cfg_dynamic = mode_cfg.copy() if mode_cfg else {}
        mode_cfg_dynamic['min_len'] = dynamic_min_len
        
        for flux in fluxes.keys():
            has_multi = (flux in flux_sorted_multi and len(flux_sorted_multi[flux]) > 1)
            is_triangulation_active = mode_cfg.get('full_triangulation', False) and has_multi
            
            if is_triangulation_active:
                num_cols = len(flux_sorted_multi[flux])
                cols_data = []
                for col_idx in range(num_cols):
                    if n_est == 1 or not boot:
                        combined_col = np.concatenate([w['fluxes_multi'][flux][col_idx] for w in matching])
                        ests = apply_complexity(combined_col, flux_name=flux, mode_cfg=mode_cfg_dynamic)
                        cols_data.append(ests)
                    else:
                        # Version séquentielle (désactivé parallélisation bootstrap pour éviter deadlocks)
                        rng = np.random.default_rng(42 + col_idx)
                        feat_vals = {nm: [] for nm in ESTIMATOR_NAMES}
                        for _boot_iter in range(n_est):
                            idx = rng.choice(len(matching), size=max(1, len(matching) // 2), replace=True)
                            sub_col = np.concatenate([matching[i]['fluxes_multi'][flux][col_idx] for i in idx])
                            ests = apply_complexity(sub_col, flux_name=flux, mode_cfg=mode_cfg_dynamic)
                            for k, v in ests.items():
                                if not np.isnan(v):
                                    feat_vals[k].append(v)
                        
                        ests_boot = {}
                        for k, vals in feat_vals.items():
                            ests_boot[k] = float(np.mean(vals)) if vals else np.nan
                        cols_data.append(ests_boot)
                
                for metric in ESTIMATOR_NAMES:
                    vals = [d[metric] for d in cols_data if not np.isnan(d.get(metric, np.nan))]
                    if vals:
                        vals = np.array(vals)
                        row[f"{flux}_{metric}_mean"] = float(np.mean(vals))
                        row[f"{flux}_{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    else:
                        row[f"{flux}_{metric}_mean"] = None
                        row[f"{flux}_{metric}_std"] = None
            else:
                if n_est == 1 or not boot:
                    combined = np.concatenate([w['fluxes'][flux] for w in matching])
                    ests = apply_complexity(combined, flux_name=flux, mode_cfg=mode_cfg_dynamic)
                    for k, v in ests.items():
                        row[f"{flux}_{k}_mean"] = float(v) if not np.isnan(v) else None
                        row[f"{flux}_{k}_std"] = 0.0
                else:
                    feat_vals = {nm: [] for nm in ESTIMATOR_NAMES}
                    rng = np.random.default_rng(42)
                    
                    for _boot_iter in range(n_est):
                        idx = rng.choice(len(matching), size=max(1, len(matching) // 2), replace=True)
                        sub = np.concatenate([matching[i]['fluxes'][flux] for i in idx])
                        ests = apply_complexity(sub, flux_name=flux, mode_cfg=mode_cfg_dynamic)
                        for k, v in ests.items():
                            if not np.isnan(v):
                                feat_vals[k].append(v)
                                
                    for k, vals in feat_vals.items():
                        if vals:
                            row[f"{flux}_{k}_mean"] = float(np.mean(vals))
                            row[f"{flux}_{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        else:
                            row[f"{flux}_{k}_mean"] = None
                            row[f"{flux}_{k}_std"] = None
                            
        rows.append(row)
        
    df_feat = pd.DataFrame(rows)
    feature_cols = [c for c in df_feat.columns if c not in ('Progress', 'N_cohorts') and not c.endswith('_std')]
    
    # Solve 4 Core Questions
    r = {}
    stat_test = mode_cfg.get('test', 'cohen_d')
    min_grp = mode_cfg.get('min_grp', 3)
    
    # Q1: Calme vs Chaos
    g_low = df_feat[df_feat['Progress'] < 40]
    g_high = df_feat[df_feat['Progress'] > 70]
    q1_scores = {}
    for col in feature_cols:
        v1 = g_low[col].dropna().values
        v2 = g_high[col].dropna().values
        s = separation_score(v1, v2, stat_test, min_grp)
        q1_scores[col] = float(s) if not np.isnan(s) else 0.0
    r["Q1_Calme_vs_Chaos"] = q1_scores
    
    # Q2: Tension at the Threshold
    q2_scores = {}
    if not np.isnan(main_threshold):
        marge = 8.0
        g_low_q2 = df_feat[df_feat['Progress'].between(main_threshold - 2*marge, main_threshold - marge*0.3)]
        g_high_q2 = df_feat[df_feat['Progress'].between(main_threshold + marge*0.3, main_threshold + 2*marge)]
        for col in feature_cols:
            v1 = g_low_q2[col].dropna().values
            v2 = g_high_q2[col].dropna().values
            s = separation_score(v1, v2, stat_test, min_grp)
            q2_scores[col] = float(s) if not np.isnan(s) else 0.0
    else:
        for col in feature_cols:
            q2_scores[col] = 0.0
    r["Q2_Tension_Seuil"] = q2_scores
    
    # Q3: Aggressiveness
    q3_scores = {}
    aggressive_fluxes = meta_info.get('aggressive_fluxes', [])
    selected_cols = [c for c in feature_cols if any(f in c for f in aggressive_fluxes)]
    if not selected_cols or len(selected_cols) < len(feature_cols) // 2:
        # Si aucun flux agressif n'est défini ou si trop peu de colonnes correspondent,
        # utiliser toutes les colonnes pour Q3
        selected_cols = feature_cols
    for col in selected_cols:
        v1 = g_low[col].dropna().values
        v2 = g_high[col].dropna().values
        s = separation_score(v1, v2, stat_test, min_grp)
        q3_scores[col] = float(s) if not np.isnan(s) else 0.0
    r["Q3_Agressivite"] = q3_scores
    
    # Q4: Persistence
    q4_scores = {}
    q4_per_flux = {}
    # Ajuster half_w dynamiquement pour éviter les problèmes avec w_size petit
    # half_w doit être assez grand pour min_grp, mais laisser de la place pour la seconde moitié
    half_w = max(min_grp, min(w_size // 2, w_size - min_grp))
    n_windows_q4 = n_pts - w_size + 1
    if n_windows_q4 > 0 and half_w < w_size:
        for flux_name, flux_arr in flux_sorted.items():
            effects = []
            if n_windows_q4 > 500:
                sample_ids = np.linspace(0, n_windows_q4 - 1, 500, dtype=int)
            else:
                sample_ids = np.arange(n_windows_q4)
            
            for i in sample_ids:
                first_half = flux_arr[i : i + half_w]
                second_half = flux_arr[i + half_w : i + w_size]
                if len(first_half) < min_grp or len(second_half) < min_grp:
                    continue
                s_score = separation_score(first_half, second_half, stat_test, min_grp)
                if not np.isnan(s_score):
                    effects.append(s_score)
            q4_per_flux[flux_name] = float(np.mean(effects)) if effects else 0.0
    else:
        for flux_name in fluxes.keys():
            q4_per_flux[flux_name] = 0.0
    
    for col in feature_cols:
        flux_name = next((flux for flux in fluxes.keys() if col.startswith(flux + "_")), col.split('_')[0])
        q4_scores[col] = q4_per_flux.get(flux_name, 0.0)
    r["Q4_Persistence"] = q4_scores
    
    # Q5: Active vs Reactive
    active_keys = meta_info.get('active_keys', [])
    reactive_keys = meta_info.get('reactive_keys', [])
    q5_scores = {}
    
    active_cols = [c for c in feature_cols if any(f in c for f in active_keys)]
    reactive_cols = [c for c in feature_cols if any(f in c for f in reactive_keys)]
    
    if active_cols and reactive_cols:
        act_avg = df_feat[active_cols].mean(axis=1).values
        react_avg = df_feat[reactive_cols].mean(axis=1).values
        
        for col in feature_cols:
            flux_name = next((flux for flux in fluxes.keys() if col.startswith(flux + "_")), col.split('_')[0])
            
            if flux_name in active_keys:
                v1 = df_feat[col].values
                v2 = react_avg
                s = separation_score(v1, v2, stat_test, min_grp)
                q5_scores[col] = float(s) if not np.isnan(s) else 0.0
            elif flux_name in reactive_keys:
                v1 = df_feat[col].values
                v2 = act_avg
                s = separation_score(v1, v2, stat_test, min_grp)
                q5_scores[col] = float(s) if not np.isnan(s) else 0.0
            else:
                # Pour les flux ni actifs ni réactifs, utiliser la séparation vs moyenne globale
                v1 = df_feat[col].values
                v2 = (act_avg + react_avg) / 2.0
                s = separation_score(v1, v2, stat_test, min_grp)
                q5_scores[col] = float(s) if not np.isnan(s) else 0.0
    else:
        for col in feature_cols:
            q5_scores[col] = 0.0
            
    r["Q5_Active_Reactive"] = q5_scores
    r["Q5_Actif_Receptif"] = q5_scores
    
    # Top 3 Signatures per Question
    top_signatures = {}
    for q_name, scores in r.items():
        if q_name == 'Q5_Active_Reactive':
            continue
        sorted_sc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_signatures[q_name] = [
            {'feature': f.replace('_mean', ''), 'score': round(val, 3)}
            for f, val in sorted_sc[:3]
        ]
        
    # Post-Threshold Transition Order Analysis
    transition_analysis = {}
    if not np.isnan(main_threshold):
        all_flux_keys = list(fluxes.keys())
            
        post_threshold_df = df_feat[df_feat['Progress'] >= main_threshold]
        if len(post_threshold_df) < 3:
            post_threshold_df = df_feat[df_feat['Progress'] >= max(0.0, main_threshold - 10.0)]
            
        if len(post_threshold_df) >= 3:
            X = post_threshold_df['Progress'].values
            X_centered = np.asarray(X - np.mean(X), dtype=float)
            for metric in ESTIMATOR_NAMES:
                transition_analysis[metric] = {}
                cols = [f"{flux}_{metric}_mean" for flux in all_flux_keys if f"{flux}_{metric}_mean" in df_feat.columns]
                
                fluxes_to_calc = {'Tous': cols}
                for f_key in all_flux_keys:
                    col_name = f"{f_key}_{metric}_mean"
                    if col_name in df_feat.columns:
                        fluxes_to_calc[f_key] = [col_name]
                        
                for flux_label, flux_cols in fluxes_to_calc.items():
                    if not flux_cols:
                        continue
                        
                    Y = np.asarray(post_threshold_df[flux_cols].mean(axis=1).values, dtype=float)
                    valid_mask = ~np.isnan(Y) & ~np.isnan(X_centered)
                    valid_X = X_centered[valid_mask]
                    valid_Y = Y[valid_mask]
                    
                    if len(valid_Y) < 3:
                        transition_analysis[metric][flux_label] = {
                            'status': 'insufficient_data',
                            'reason': 'Points insuffisants après seuil'
                        }
                        continue
                        
                    try:
                        coeffs = np.polyfit(valid_X, valid_Y, 2)
                        a = coeffs[0]
                        b = coeffs[1]
                        second_deriv = float(2.0 * a)
                        first_deriv = float(b)  # Dérivée première au point x=0 (X centré)
                        
                        if first_deriv > 1e-6:
                            order = "2nd"
                            sign = "positive"
                            interpretation = "Transition de 2nd ordre (Transition Douce : Complexité/signal en hausse post-seuil)."
                        elif first_deriv < -1e-6:
                            order = "1st"
                            sign = "negative"
                            interpretation = "Transition de 1er ordre (Transition Brutale : Chute du signal/complexité post-seuil, stabilisation)."
                        else:
                            order = "linear"
                            sign = "neutral"
                            interpretation = "Transition linéaire - Pas de pente notable post-seuil."
                            
                        transition_analysis[metric][flux_label] = {
                            'status': 'success',
                            'first_derivative': first_deriv,
                            'second_derivative': second_deriv,
                            'sign': sign,
                            'order': order,
                            'interpretation': interpretation,
                            'points_count': len(X)
                        }
                    except Exception as poly_err:
                        transition_analysis[metric][flux_label] = {
                            'status': 'error',
                            'reason': str(poly_err)
                        }

                if not transition_analysis[metric]:
                    del transition_analysis[metric]

        # ------------------------------------------------------------
        # v3.4 -- F1/F2 AVANT seuil (pre), en miroir de l'apres-seuil (post).
        # Methode IDENTIQUE au post : polyfit degre 2 sur Progress centree
        # (X - moyenne(X) du segment), F1 = coefficient lineaire (pente au
        # centre du segment), F2 = 2 * coefficient quadratique. Les champs
        # post existants (first_derivative, second_derivative, sign, order,
        # interpretation, points_count) sont INCHANGES ; on ajoute les
        # champs *_pre, les fenetres utilisees, et les ecarts post-pre.
        # Segment pre primaire : Progress < seuil (disjoint du post >=).
        # Si < 3 points, elargissement miroir du post : Progress <
        # min(100, seuil + 10), avec window_expanded_pre = True. Si encore
        # < 3 points, status_pre = 'insufficient_data' (explicite).
        pre_threshold_df = df_feat[df_feat['Progress'] < main_threshold]
        pre_expanded = False
        if len(pre_threshold_df) < 3:
            pre_threshold_df = df_feat[df_feat['Progress'] < min(100.0, main_threshold + 10.0)]
            pre_expanded = True
        if len(pre_threshold_df) >= 3:
            X_pre = pre_threshold_df['Progress'].values
            X_pre_centered = np.asarray(X_pre - np.mean(X_pre), dtype=float)
            for metric in ESTIMATOR_NAMES:
                cols = [f"{flux}_{metric}_mean" for flux in all_flux_keys if f"{flux}_{metric}_mean" in df_feat.columns]
                fluxes_to_calc_pre = {'Tous': cols}
                for f_key in all_flux_keys:
                    col_name = f"{f_key}_{metric}_mean"
                    if col_name in df_feat.columns:
                        fluxes_to_calc_pre[f_key] = [col_name]
                for flux_label, flux_cols in fluxes_to_calc_pre.items():
                    if not flux_cols:
                        continue
                    Y_pre = np.asarray(pre_threshold_df[flux_cols].mean(axis=1).values, dtype=float)
                    valid_mask_pre = ~np.isnan(Y_pre) & ~np.isnan(X_pre_centered)
                    valid_X_pre = X_pre_centered[valid_mask_pre]
                    valid_Y_pre = Y_pre[valid_mask_pre]
                    entry = transition_analysis.setdefault(metric, {}).setdefault(flux_label, {})
                    entry['window_pre'] = [float(np.min(X_pre)), float(np.max(X_pre))]
                    entry['window_expanded_pre'] = bool(pre_expanded)
                    if len(valid_Y_pre) < 3:
                        entry['status_pre'] = 'insufficient_data'
                        entry['reason_pre'] = 'Points insuffisants avant seuil'
                        continue
                    try:
                        coeffs_pre = np.polyfit(valid_X_pre, valid_Y_pre, 2)
                        a_pre = coeffs_pre[0]
                        b_pre = coeffs_pre[1]
                        second_pre = float(2.0 * a_pre)
                        first_pre = float(b_pre)
                        if first_pre > 1e-6:
                            order_pre = "2nd"
                            sign_pre = "positive"
                            interpretation_pre = "Transition de 2nd ordre AVANT seuil (signal en hausse avant seuil)."
                        elif first_pre < -1e-6:
                            order_pre = "1st"
                            sign_pre = "negative"
                            interpretation_pre = "Transition de 1er ordre AVANT seuil (chute du signal avant seuil)."
                        else:
                            order_pre = "linear"
                            sign_pre = "neutral"
                            interpretation_pre = "Transition lineaire AVANT seuil - Pas de pente notable avant seuil."
                        entry['status_pre'] = 'success'
                        entry['first_derivative_pre'] = first_pre
                        entry['second_derivative_pre'] = second_pre
                        entry['sign_pre'] = sign_pre
                        entry['order_pre'] = order_pre
                        entry['interpretation_pre'] = interpretation_pre
                        entry['points_count_pre'] = len(X_pre)
                        # Ecarts post - pre, uniquement si les deux cotes
                        # ont reussi (sinon None, explicite, pas 0).
                        if entry.get('status') == 'success':
                            entry['delta_first_derivative'] = float(entry['first_derivative'] - first_pre)
                            entry['delta_second_derivative'] = float(entry['second_derivative'] - second_pre)
                            entry['sign_change'] = bool(entry['sign'] != sign_pre)
                        else:
                            entry['delta_first_derivative'] = None
                            entry['delta_second_derivative'] = None
                            entry['sign_change'] = None
                    except Exception as poly_err_pre:
                        entry['status_pre'] = 'error'
                        entry['reason_pre'] = str(poly_err_pre)
        else:
            # Pre insuffisant meme apres elargissement : on le consigne
            # pour chaque metrique/flux deja present en post.
            for _metric, _fluxes in list(transition_analysis.items()):
                for _label, _entry in _fluxes.items():
                    _entry['status_pre'] = 'insufficient_data'
                    _entry['reason_pre'] = 'Points insuffisants avant seuil, meme apres elargissement'
                    _entry['window_expanded_pre'] = bool(pre_expanded)

    # Noise Diagnostic
    try:
        # v3.4 : seuil NaN -> diagnostic bruit avec 52,5 % d'urgence,
        # JOURNALISE (en v3.3, ce 52,5 % fixe etait silencieux).
        if main_threshold is None or (isinstance(main_threshold, float) and np.isnan(main_threshold)):
            _record_threshold_warning(
                "REPLI SEUIL (diagnostic bruit) : seuil principal NaN, "
                "le diagnostic bruit utilise la valeur d'urgence fixe "
                "52,5 % ; ce 52,5 % n'est PAS un seuil detecte.")
        opt_thresh_val = float(main_threshold) if (main_threshold is not None and not np.isnan(main_threshold)) else 52.5
        noise_diag_res = compute_noise_diagnostic(df, progress, fluxes, opt_thresh_val, active_keys, reactive_keys)
    except Exception as diag_err:
        fallback_col = list(fluxes.keys())[-1] if (fluxes and len(fluxes) > 0) else "Bruit"
        noise_diag_res = {
            'is_variance_proxy': False,
            'noise_col': fallback_col,
            'opt_threshold': main_threshold if not np.isnan(main_threshold) else 52.5,
            'summary_text': f"Erreur de calcul du diagnostic bruit: {str(diag_err)}",
            'diagnostic_label': "❌ Erreur de Diagnostic",
            'error': str(diag_err),
            'step1': {'candidate_q5_score': 0, 'true_white_noise_q5_score': 0, 'ratio': 0, 'interpretation': f"Erreur: {str(diag_err)}"},
            'step2': {'corr_before_threshold': 0, 'corr_after_threshold': 0, 'delta_corr': 0, 'interpretation': "Non disponible"},
            'step3': {'hurst_by_bin': [], 'interpretation': "Non disponible"},
            'step4': {'spectral_slope_before': 0, 'spectral_slope_after': 0, 'psd_chart': [], 'interpretation': "Non disponible"}
        }

    # Package results - remplacer NaN par None pour les données manquantes
    landscape_data = df_feat.where(pd.notnull(df_feat), None).to_dict(orient='records')
    
    distribution_samples = []
    sample_indices = np.linspace(0, len(df)-1, min(200, len(df)), dtype=int)
    for idx in sample_indices:
        record = {
            'progress': float(progress[idx]),
        }
        for flux_name, val_arr in fluxes_ui.items():
            record[flux_name] = float(val_arr[idx])
        distribution_samples.append(record)
    
    # --- Normalization Sensitivity Diagnostic ---
    # Compare les estimateurs de complexité sous normalisation simple (z-score)
    # vs double normalisation (min-max [0,1] puis z-score)
    try:
        norm_diagnostic = {}
        n_sample_windows = min(10, len(scan_points))
        sample_P_indices = np.linspace(0, len(scan_points) - 1, n_sample_windows, dtype=int)
        
        for flux_name in fluxes.keys():
            flux_raw = flux_sorted[flux_name]
            # Construire version min-max
            f_min, f_max = np.nanmin(flux_raw), np.nanmax(flux_raw)
            if f_max == f_min:
                flux_minmax = np.zeros_like(flux_raw)
            else:
                flux_minmax = np.clip((flux_raw - f_min) / (f_max - f_min), 0.0, 1.0)
            
            zscore_results = {m: [] for m in ESTIMATOR_NAMES}
            double_results = {m: [] for m in ESTIMATOR_NAMES}
            
            for pi in sample_P_indices:
                P = scan_points[pi]
                matching_ids = [i for i in range(n_pts - w_size + 1)
                                if abs(float(np.mean(prog_sorted[i:i+w_size])) - P) <= tol]
                if not matching_ids:
                    closest = min(range(n_pts - w_size + 1),
                                  key=lambda i: abs(float(np.mean(prog_sorted[i:i+w_size])) - P))
                    matching_ids = [closest]
                
                # Z-score seul (défaut pipeline)
                combined_raw = np.concatenate([flux_raw[i:i+w_size] for i in matching_ids])
                est_z = apply_complexity(combined_raw, flux_name=flux_name, mode_cfg=mode_cfg)
                
                # Double normalisation: min-max [0,1] puis z-score
                combined_mm = np.concatenate([flux_minmax[i:i+w_size] for i in matching_ids])
                est_mm = apply_complexity(combined_mm, flux_name=flux_name, mode_cfg=mode_cfg)
                
                for m in zscore_results:
                    if not np.isnan(est_z.get(m, np.nan)):
                        zscore_results[m].append(est_z[m])
                    if not np.isnan(est_mm.get(m, np.nan)):
                        double_results[m].append(est_mm[m])
            
            # Calculer divergence entre les deux stratégies
            metrics_comparison = {}
            for m in ESTIMATOR_NAMES:
                z_vals = zscore_results[m]
                d_vals = double_results[m]
                if z_vals and d_vals:
                    z_mean = float(np.mean(z_vals))
                    d_mean = float(np.mean(d_vals))
                    abs_diff = float(abs(z_mean - d_mean))
                    rel_diff = float(abs_diff / (abs(z_mean) + 1e-9) * 100)
                    metrics_comparison[m] = {
                        'zscore_only_mean': round(z_mean, 4),
                        'minmax_zscore_mean': round(d_mean, 4),
                        'absolute_diff': round(abs_diff, 4),
                        'relative_diff_pct': round(rel_diff, 1),
                        'impact': 'FAIBLE' if rel_diff < 5 else ('MODÉRÉ' if rel_diff < 20 else 'FORT')
                    }
                else:
                    metrics_comparison[m] = {
                        'zscore_only_mean': None,
                        'minmax_zscore_mean': None,
                        'absolute_diff': None,
                        'relative_diff_pct': None,
                        'impact': 'NON_CALCULÉ'
                    }
            
            # Verdict global pour ce flux
            impacts = [v['relative_diff_pct'] for v in metrics_comparison.values()
                       if v['relative_diff_pct'] is not None]
            avg_divergence = float(np.mean(impacts)) if impacts else 0.0
            if avg_divergence < 5:
                verdict = "✅ Normalisation min-max sans impact significatif — les deux stratégies convergent."
            elif avg_divergence < 20:
                verdict = "⚠️ Impact modéré de la double normalisation — certains estimateurs divergent sensiblement."
            else:
                verdict = "🔴 Impact fort — la normalisation min-max altère significativement les signatures de complexité."
            
            norm_diagnostic[flux_name] = {
                'metrics': metrics_comparison,
                'avg_divergence_pct': round(avg_divergence, 1),
                'verdict': verdict
            }
    except Exception as norm_err:
        norm_diagnostic = {
            'error': str(norm_err),
            'verdict': f"Erreur lors du diagnostic de normalisation: {str(norm_err)}"
        }

    # Calculer csd_ranking à partir du diagnostic de bruit
    csd_ranking = []
    if noise_diag_res and 'all_flux_diagnostics' in noise_diag_res:
        for diag in noise_diag_res['all_flux_diagnostics']:
            flux_name = diag['flux_name']
            variance_proxy_score = diag.get('variance_proxy_score', 0.0)
            csd_ranking.append((flux_name, float(variance_proxy_score)))
        # Trier par score décroissant
        csd_ranking.sort(key=lambda x: x[1], reverse=True)

    output = {
        'success': True,
        'metadata': {
            'domain': data_source,
            'control_label': meta_info.get('control_col', 'Progression axis'),
            'mode': mode,
            'stat_test': stat_test,
            'w_size': w_size,
            'n_pts': n_pts,
            'opt_threshold': main_threshold,
            'threshold_source': threshold_source,
            'threshold_warnings': list(threshold_warnings),
            'stability_report': stability_report,
            'threshold_optimization': threshold_optimization,
            'random_seed': mode_cfg.get('random_seed', RANDOM_SEED)
        },
        'landscape': landscape_data,
        'distribution_samples': distribution_samples,
        'scan_curve': scan_curve,
        'scores': r,
        'top_signatures': top_signatures,
        'noise_diagnostic': noise_diag_res,
        # Analyse post-seuil conservée au niveau racine : elle est consommée
        # par les visualisations et disponible dans les exports JSON.
        'transition_analysis': transition_analysis,
        'normalization_diagnostic': norm_diagnostic,
        'csd_ranking': csd_ranking,
    }
    
    return output, df_feat, fluxes, fluxes_ui

def main():
    parser = argparse.ArgumentParser(
        description='UNIVERSAL_LOG v3.4-H3 - Pipeline Autonome + Monte Carlo Harmonique GUIDE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python pipeline_autonome.py --input data.csv --output results/
  python pipeline_autonome.py --input data.csv --mode robust --output analysis/
  python pipeline_autonome.py --input data.csv --test cliff_delta --window 120
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Chemin vers le fichier CSV d\'entrée (auto-détection si non spécifié)')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Dossier de sortie pour les résultats (défaut: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--mode', '-m', type=str, default=ANALYSIS_MODE,
                        choices=['fast', 'robust', 'stable'],
                        help=f'Mode d\'analyse (défaut: {ANALYSIS_MODE})')
    parser.add_argument('--test', '-t', type=str, default=STATISTICAL_TEST,
                        choices=['cohen_d', 'cliff_delta'],
                        help=f'Test statistique (défaut: {STATISTICAL_TEST})')
    parser.add_argument('--window', '-w', type=int, default=WINDOW_SIZE,
                        help=f'Taille de fenêtre (défaut: {WINDOW_SIZE})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Graine de reproductibilité Monte Carlo (défaut: {RANDOM_SEED})')
    parser.add_argument('--run-simulation', action='store_true',
                        help='Exécuter la simulation externe optionnelle si le module simulation est installé')
    parser.add_argument('--config', '-c', action='store_true',
                        help='Afficher la configuration actuelle et quitter')
    
    args = parser.parse_args()
    
    # Afficher la configuration si demandé
    if args.config:
        print("=" * 60)
        print("CONFIGURATION ACTUELLE")
        print("=" * 60)
        print(f"Mode d'analyse: {args.mode}")
        print(f"Test statistique: {args.test}")
        print(f"Taille de fenêtre: {args.window}")
        print(f"Graine aléatoire: {args.seed}")
        print(f"Dossier de sortie: {args.output}")
        print(f"Fichier d'entrée: Auto-détection" if args.input is None else f"Fichier d'entrée: {args.input}")
        print("=" * 60)
        return
    
    # Auto-détection du fichier CSV si non spécifié
    input_file = args.input
    if input_file is None:
        csv_files = glob.glob("*.csv")
        if csv_files:
            input_file = csv_files[0]
            print(f"📂 Fichier CSV auto-détecté: {input_file}")
        else:
            print("❌ Erreur: Aucun fichier CSV trouvé dans le dossier actuel.")
            print("   Veuillez spécifier un fichier avec --input <fichier.csv>")
            sys.exit(1)
    
    # Vérifier que le fichier existe
    if not os.path.exists(input_file):
        print(f"❌ Erreur: Le fichier '{input_file}' n'existe pas.")
        sys.exit(1)
    
    # Créer le dossier de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossier de sortie: {output_dir}")
    
    # Configuration
    opt_cfg = {
        'w_size': args.window,
        'test': args.test,
        'full_triangulation': FULL_TRIANGULATION,
        'random_seed': args.seed,
        'harmonic_prob': HARMONIC_PROB,
    }
    
    print("\n" + "=" * 60)
    print("DÉMARRAGE DU PIPELINE D'ANALYSE — HARMONIQUE GUIDÉ v1")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Test: {args.test}")
    print(f"Fenêtre: {args.window}")
    print(f"Monte Carlo harmonique guidé: P={HARMONIC_PROB}")
    print(f"Fichier: {input_file}")
    print("=" * 60 + "\n")
    
    try:
        # Exécuter le pipeline
        results, df_feat, fluxes, fluxes_ui = run_pipeline(
            data_source='custom',
            custom_file_path=input_file,
            mapping=None,
            mode=args.mode,
            opt_cfg=opt_cfg
        )
        
        if not results.get('success'):
            print(f"❌ Erreur lors de l'exécution du pipeline")
            sys.exit(1)
        
        print("✓ Pipeline exécuté avec succès\n")
        
        # Générer les visualisations
        print("Génération des visualisations...")
        main_threshold = results['metadata'].get('opt_threshold', 50)
        
        plot_complexity_landscape(
            df_feat, fluxes, main_threshold, 
            output_dir / 'landscape_complexity.png'
        )
        
        plot_threshold_scan(
            results['scan_curve'], main_threshold,
            output_dir / 'threshold_scan.png'
        )
        
        plot_raw_fluxes(
            results['distribution_samples'], fluxes_ui, main_threshold,
            output_dir / 'raw_fluxes_distribution.png'
        )
        
        plot_separation_heatmap(
            results['scores'], output_dir / 'separation_heatmap.png'
        )
        
        if results['metadata'].get('stability_report'):
            plot_stability_analysis(
                results['metadata']['stability_report'],
                output_dir / 'stability_analysis.png'
            )
        
        if results.get('transition_analysis'):
            plot_post_threshold_analysis(
                results['transition_analysis'],
                output_dir / 'post_threshold_analysis.png'
            )
        
        plot_noise_diagnostic(
            results['noise_diagnostic'],
            output_dir / 'noise_diagnostic.png'
        )

        # Analyse de corrélation CSD vs non-CSD
        print("\nAnalyse de corrélation CSD vs non-CSD...")
        csd_analysis = compute_csd_correlation_analysis(
            results['noise_diagnostic'], df_feat, fluxes
        )
        
        # Le schéma de sortie contient ``csd_correlations`` ; la figure est
        # produite dès qu'une analyse exploitable est disponible.
        results['csd_correlation_analysis'] = csd_analysis
        if csd_analysis and 'csd_correlations' in csd_analysis:
            plot_csd_correlation_analysis(
                csd_analysis,
                output_dir / 'csd_correlation_analysis.png'
            )
        
        # Exporter les résultats
        print("\nExport des résultats...")
        
        export_csv_results(
            df_feat, results['scores'], results['noise_diagnostic'],
            output_dir / 'numerical_results.csv'
        )
        
        export_text_report(
            results, output_dir / 'text_report.txt'
        )
        
        export_json_results(
            results, output_dir / 'raw_results.json'
        )
        export_json_results(
            results, output_dir / 'pipeline_results.json'
        )
        
        # La simulation externe n'est lancée que sur demande explicite : elle
        # n'appartient pas au pipeline autonome fourni dans ce fichier.
        if args.run_simulation:
            try:
                print("\nLancement de la simulation externe demandée...")
                from simulation.run_simulation import run_simulation_pipeline
                run_simulation_pipeline(output_dir)
            except ImportError:
                print(
                    "⚠ Simulation non lancée : le module optionnel 'simulation' "
                    "n'est pas installé ou n'est pas disponible."
                )
            except Exception as sim_err:
                print(f"⚠ Erreur lors de la simulation externe : {sim_err}")
        
        # Log d'exécution
        log_path = output_dir / 'execution_log.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Execution Log - UNIVERSAL LOG v3.3\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input: {input_file}\n")
            f.write(f"Output: {output_dir}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Test: {args.test}\n")
            f.write(f"Window: {args.window}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Status: SUCCESS\n")
        
        print(f"✓ Log d'exécution sauvegardé: {log_path}")
        
        print("\n" + "=" * 60)
        print("ANALYSE TERMINÉE AVEC SUCCÈS")
        print("=" * 60)
        print(f"📊 Tous les résultats ont été sauvegardés dans: {output_dir}")
        
        # Afficher les 4 seuils
        threshold_opt = results['metadata'].get('threshold_optimization')
        if threshold_opt and threshold_opt.get('best_config'):
            print("\n🎯 SEUILS CRITIQUES (DONNÉES TRAITÉES):")
            mc_threshold = threshold_opt['best_config'].get('threshold', main_threshold)
            default_threshold = threshold_opt.get('default_threshold', main_threshold)
            print(f"  Monte Carlo: {mc_threshold:.1f}%")
            print(f"  Par défaut: {default_threshold:.1f}%")
            
            print("\n🎯 SEUILS CRITIQUES (DONNÉES BRUTES):")
            raw_mc_threshold = threshold_opt.get('raw_mc_threshold')
            raw_default_threshold = threshold_opt.get('raw_default_threshold')
            if raw_mc_threshold is not None:
                print(f"  Monte Carlo: {raw_mc_threshold:.1f}%")
            if raw_default_threshold is not None:
                print(f"  Par défaut: {raw_default_threshold:.1f}%")
        else:
            print(f"🎯 Seuil critique détecté: {main_threshold:.1f}%")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
