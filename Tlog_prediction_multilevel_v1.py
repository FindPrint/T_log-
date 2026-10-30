# -*- coding: utf-8 -*-
"""
UNIVERSAL_LOG — Prédiction Multi-Niveaux v1
==========================================
Suite proactive du pipeline Tlog (pipeline d'analyse NON modifié).

Objectif
--------
Une seule exécution produit **plusieurs objectifs de prédiction** simultanés,
pondérés et interprétés selon le **domaine**, avec un score de **confiance
robuste** et des signaux **proactifs** (alerte anticipée, scénarios divergents).

Niveaux de prédiction (toujours calculés, puis filtrés/pondérés par domaine)
---------------------------------------------------------------------------
  L1  Direction de la cible        (hausse / baisse / neutre)
  L2  Magnitude relative           (amplitude attendue + bande)
  L3  Régime structurel            (pré-seuil / post-seuil / transition)
  L4  Risque structurel            (EWS + CSD + instabilité)
  L5  Trajectoire de complexité    (estimateurs dominants)
  L6  Prochaine zone de seuil      (second pic de scan si disponible)
  L7  Cohérence multi-signaux      (accord entre L1–L6 → confiance)

Domaines → poids relatifs des niveaux + lexique monde réel
  finance | physics | biology | climate | social | generic

Profils de risque → largeur des bandes, seuils d'alerte, agressivité proactive
  conservative | moderate | aggressive

Robustesse
----------
  - Agrégation par vote pondéré (direction) et moyenne robuste (magnitude)
  - Pénalité si les niveaux se contredisent
  - Hold-out directionnel + R² quand une cible numérique est fournie
  - Scénarios Monte Carlo harmoniques (même esprit que le pipeline)
  - Signaux proactifs : « anticiper » si EWS monte alors que le seuil est encore loin

Usage
-----
  python Tlog_prediction_multilevel_v1.py \\
    -p ./results_enriched \\
    --domain finance --risk moderate \\
    --target Close --control Energy_Diff \\
    --source btc_enriched_lean.csv \\
    --horizon 20
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("default")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 110

# =============================================================================
# CONFIG
# =============================================================================

RANDOM_SEED = 42
HARMONIC_PROB = 0.45
DEFAULT_HORIZON = 15
DEFAULT_N_SCENARIOS = 48
HOLD_OUT_FRAC = 0.12

PHI = (1 + 5 ** 0.5) / 2
PI = 3.141592653589793
E = 2.718281828459045
SQRT2 = 1.4142135623730951
SQRT3 = 1.7320508075688772

CORE_METRICS = ["Higuchi", "DFA", "Katz", "Hurst", "SampleEn"]
EXTRA_METRICS = ["Lyapunov", "MultifractalDeltaAlpha", "CorrelationD2"]
ALL_METRICS = CORE_METRICS + EXTRA_METRICS

RISK_PROFILES = {
    "conservative": {
        "alert_threshold": 0.70,
        "band_q_low": 5,
        "band_q_high": 95,
        "neutral_band": 0.012,
        "scenario_vol_scale": 0.75,
        "proactive_lead": 0.12,  # distance au seuil pour anticiper
        "label": "prudent",
    },
    "moderate": {
        "alert_threshold": 0.55,
        "band_q_low": 10,
        "band_q_high": 90,
        "neutral_band": 0.02,
        "scenario_vol_scale": 1.0,
        "proactive_lead": 0.18,
        "label": "équilibré",
    },
    "aggressive": {
        "alert_threshold": 0.40,
        "band_q_low": 15,
        "band_q_high": 85,
        "neutral_band": 0.035,
        "scenario_vol_scale": 1.35,
        "proactive_lead": 0.25,
        "label": "audacieux",
    },
}

# Poids des niveaux par domaine (somme ~1). Adapte la « personnalité » du mini-cerveau.
DOMAIN_LEVEL_WEIGHTS = {
    "finance": {
        "L1_direction": 0.22,
        "L2_magnitude": 0.18,
        "L3_regime": 0.15,
        "L4_risk": 0.18,
        "L5_complexity": 0.12,
        "L6_next_threshold": 0.05,
        "L7_coherence": 0.10,
    },
    "physics": {
        "L1_direction": 0.10,
        "L2_magnitude": 0.15,
        "L3_regime": 0.22,
        "L4_risk": 0.18,
        "L5_complexity": 0.20,
        "L6_next_threshold": 0.10,
        "L7_coherence": 0.05,
    },
    "biology": {
        "L1_direction": 0.12,
        "L2_magnitude": 0.15,
        "L3_regime": 0.20,
        "L4_risk": 0.22,
        "L5_complexity": 0.15,
        "L6_next_threshold": 0.08,
        "L7_coherence": 0.08,
    },
    "climate": {
        "L1_direction": 0.12,
        "L2_magnitude": 0.15,
        "L3_regime": 0.22,
        "L4_risk": 0.20,
        "L5_complexity": 0.15,
        "L6_next_threshold": 0.10,
        "L7_coherence": 0.06,
    },
    "social": {
        "L1_direction": 0.15,
        "L2_magnitude": 0.12,
        "L3_regime": 0.20,
        "L4_risk": 0.22,
        "L5_complexity": 0.12,
        "L6_next_threshold": 0.08,
        "L7_coherence": 0.11,
    },
    "generic": {
        "L1_direction": 0.16,
        "L2_magnitude": 0.14,
        "L3_regime": 0.16,
        "L4_risk": 0.16,
        "L5_complexity": 0.16,
        "L6_next_threshold": 0.08,
        "L7_coherence": 0.14,
    },
}

DOMAIN_LEXICON = {
    "finance": {
        "up": "hausse probable",
        "down": "baisse probable",
        "flat": "consolidation / évolution limitée",
        "pre": "régime calme / accumulation",
        "post": "régime stressé / distribution",
        "transition": "zone de transition critique",
        "risk_high": "risque structurel élevé",
        "risk_med": "risque structurel modéré",
        "risk_low": "risque structurel faible",
        "proactive": "signal anticipé avant la bascule de régime",
        "unit": "variation relative de l'actif",
    },
    "physics": {
        "up": "augmentation d'amplitude / d'ordre",
        "down": "relaxation / diminution",
        "flat": "quasi-équilibre",
        "pre": "phase pré-critique",
        "post": "phase post-critique",
        "transition": "voisinage du point critique",
        "risk_high": "proximité d'une transition critique",
        "risk_med": "instabilité modérée",
        "risk_low": "loin d'une transition détectable",
        "proactive": "précurseur de transition (EWS)",
        "unit": "unités de la grandeur observée",
    },
    "biology": {
        "up": "augmentation d'activité / expression",
        "down": "diminution / inhibition",
        "flat": "état stable",
        "pre": "état basal",
        "post": "état activé / altéré",
        "transition": "bascule physiologique",
        "risk_high": "signature de bascule",
        "risk_med": "stress physiologique modéré",
        "risk_low": "dynamique stable",
        "proactive": "alerte précoce biomarqueur",
        "unit": "variation relative du biomarqueur",
    },
    "climate": {
        "up": "intensification",
        "down": "atténuation",
        "flat": "régime stationnaire",
        "pre": "régime de référence",
        "post": "régime altéré",
        "transition": "bascule climatique locale",
        "risk_high": "risque de bascule",
        "risk_med": "dérive modérée",
        "risk_low": "pas de bascule détectée",
        "proactive": "précurseur de bascule",
        "unit": "unités de la variable climatique",
    },
    "social": {
        "up": "intensification du signal",
        "down": "relâchement",
        "flat": "stabilité relative",
        "pre": "phase stable",
        "post": "phase de tension",
        "transition": "polarisation / bascule collective",
        "risk_high": "risque de bascule collective",
        "risk_med": "tension modérée",
        "risk_low": "dynamique contenue",
        "proactive": "signal anticipé de tension",
        "unit": "variation relative de l'indicateur",
    },
    "generic": {
        "up": "augmentation probable",
        "down": "diminution probable",
        "flat": "évolution limitée",
        "pre": "régime pré-seuil",
        "post": "régime post-seuil",
        "transition": "zone de transition",
        "risk_high": "risque structurel élevé",
        "risk_med": "risque structurel modéré",
        "risk_low": "risque structurel faible",
        "proactive": "signal anticipé",
        "unit": "variation relative de la cible",
    },
}


# =============================================================================
# UTILS
# =============================================================================

def _sf(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _harmonic_sample(rng, low, high, p=HARMONIC_PROB):
    if rng.random() < p:
        rule = rng.choice(["phi", "pi", "e", "sqrt2", "sqrt3"])
        mid = (low + high) / 2.0
        if rule == "phi":
            k = int(rng.integers(1, 4))
            val = mid * (PHI ** (k - 2))
        elif rule == "pi":
            val = mid * (1.0 + 0.3 * np.sin(rng.uniform(0, PI)))
        elif rule == "e":
            val = low + (high - low) * (1.0 - 1.0 / E)
        elif rule == "sqrt2":
            val = low + (high - low) / SQRT2
        else:
            val = low + (high - low) / SQRT3
        return float(np.clip(val, low, high)), rule
    return float(rng.uniform(low, high)), "uniform"


def _fit_poly(x, y, degree=2):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < degree + 2:
        return None
    try:
        coeffs = np.polyfit(x[m], y[m], degree)
        pred = np.polyval(coeffs, x[m])
        resid = y[m] - pred
        r2 = float(np.clip(1.0 - np.var(resid) / (np.var(y[m]) + 1e-12), -2.0, 1.0))
        return {"coeffs": coeffs, "r2": r2, "resid_std": float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0}
    except Exception:
        return None


def _pred_poly(model, x):
    if model is None:
        return np.full(len(np.asarray(x)), np.nan)
    return np.polyval(model["coeffs"], np.asarray(x, float))


# =============================================================================
# LOAD
# =============================================================================

def load_pipeline(pipeline_dir: Path) -> Dict[str, Any]:
    pipeline_dir = Path(pipeline_dir)
    data: Dict[str, Any] = {}
    for name in ("pipeline_results.json", "raw_results.json"):
        p = pipeline_dir / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data["json"] = json.load(f)
            break
    if "json" not in data:
        raise FileNotFoundError(f"JSON pipeline introuvable dans {pipeline_dir}")
    csv_p = pipeline_dir / "numerical_results.csv"
    data["landscape"] = pd.read_csv(csv_p) if csv_p.exists() else pd.DataFrame(data["json"].get("landscape", []))
    meta = data["json"].get("metadata", {})
    data["threshold"] = _sf(meta.get("opt_threshold"), 50.0)
    data["threshold_source"] = meta.get("threshold_source", "unknown")
    data["csd_ranking"] = data["json"].get("csd_ranking", [])
    data["noise_diag"] = data["json"].get("noise_diagnostic", {})
    data["scan_curve"] = data["json"].get("scan_curve", [])
    data["top_signatures"] = data["json"].get("top_signatures", {})
    return data


def load_source(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip().str.replace('"', "").str.replace("'", "")
    return df


def extract_metric(landscape: pd.DataFrame, metric: str):
    if landscape is None or landscape.empty or "Progress" not in landscape.columns:
        return None, None
    progress = landscape["Progress"].values.astype(float)
    cols = [c for c in landscape.columns if c.endswith(f"_{metric}_mean")]
    if not cols:
        return progress, None
    stacked = []
    for c in cols:
        v = landscape[c].values.astype(float)
        if np.any(np.isfinite(v)):
            stacked.append(v)
    if not stacked:
        return progress, None
    with np.errstate(all="ignore"):
        g = np.nanmean(np.vstack(stacked), axis=0)
    return progress, g


# =============================================================================
# BLOCS DE BASE
# =============================================================================

def ews_series(series, progress, threshold, window=8):
    s = np.asarray(series, float)
    p = np.asarray(progress, float)
    n = len(s)
    if n < window + 2:
        return None
    var_roll = np.full(n, np.nan)
    ac1_roll = np.full(n, np.nan)
    for i in range(window, n):
        seg = s[i - window : i]
        if np.sum(np.isfinite(seg)) < window // 2:
            continue
        var_roll[i] = np.nanvar(seg)
        if np.sum(np.isfinite(seg[:-1])) > 2:
            c = np.corrcoef(seg[:-1], seg[1:])[0, 1]
            ac1_roll[i] = c if np.isfinite(c) else np.nan
    pre = p < threshold
    with np.errstate(all="ignore"):
        ref_var = np.nanmean(var_roll[pre]) if pre.sum() > 5 and np.any(np.isfinite(var_roll[pre])) else np.nanmean(var_roll)
        ref_ac1 = np.nanmean(ac1_roll[pre]) if pre.sum() > 5 and np.any(np.isfinite(ac1_roll[pre])) else np.nanmean(ac1_roll)
    if not np.isfinite(ref_var):
        ref_var = 0.0
    if not np.isfinite(ref_ac1):
        ref_ac1 = 0.0
    var_ratio = var_roll / (ref_var + 1e-9)
    ac1_rise = (ac1_roll - ref_ac1) / (abs(ref_ac1) + 0.15)
    alert = np.clip(0.45 * np.tanh(var_ratio - 1.0) + 0.35 * np.tanh(ac1_rise) + 0.20, 0, 1)
    cur = float(alert[-1]) if np.isfinite(alert[-1]) else None
    return {"alert_score": alert, "current_alert": cur, "ref_variance": float(ref_var)}


def mc_paths(progress, series, threshold, horizon, n_sc, risk_prof, seed):
    rng = np.random.default_rng(seed)
    p = np.asarray(progress, float)
    s = np.asarray(series, float)
    m = np.isfinite(p) & np.isfinite(s)
    p, s = p[m], s[m]
    if len(p) < 8:
        return None
    last_p, last_s = float(p[-1]), float(s[-1])
    pre, post = s[p < threshold], s[p >= threshold]
    mu_pre = float(np.nanmean(pre)) if len(pre) > 2 else last_s
    mu_post = float(np.nanmean(post)) if len(post) > 2 else last_s
    sd_pre = float(np.nanstd(pre)) if len(pre) > 2 else 0.1
    sd_post = float(np.nanstd(post)) if len(post) > 2 else 0.1
    recent = min(12, max(3, len(s) // 2))
    slope = 0.0
    if recent >= 3:
        try:
            slope, *_ = stats.linregress(p[-recent:], s[-recent:])
        except Exception:
            pass
    vol_base = float(risk_prof["scenario_vol_scale"])
    future_p = np.linspace(last_p, min(100.0, last_p + horizon * 2.8), horizon)
    paths = []
    for _ in range(n_sc):
        drift, _ = _harmonic_sample(rng, -0.18, 0.28)
        vol_s, _ = _harmonic_sample(rng, 0.35, 1.9)
        vol_s *= vol_base
        path = [last_s]
        for t in range(1, horizon):
            cur_p = future_p[t]
            mu = mu_pre if cur_p < threshold else mu_post
            sd = (sd_pre if cur_p < threshold else sd_post) * vol_s
            rev = 0.08 * (mu - path[-1])
            step = rev + drift * 0.4 + slope * (future_p[t] - future_p[t - 1]) * 0.55
            path.append(path[-1] + step + rng.normal(0, max(sd * 0.35, 1e-6)))
        paths.append(path)
    paths = np.asarray(paths, float)
    qlo, qhi = risk_prof["band_q_low"], risk_prof["band_q_high"]
    return {
        "future_progress": future_p,
        "median": np.nanmedian(paths, axis=0),
        "q_low": np.nanpercentile(paths, qlo, axis=0),
        "q_high": np.nanpercentile(paths, qhi, axis=0),
        "std": np.nanstd(paths, axis=0),
        "paths": paths,
        "n": n_sc,
    }


# =============================================================================
# NIVEAUX L1–L6
# =============================================================================

def level_target(source_df, target_col, control_col, data, horizon, n_sc, risk_prof, seed):
    """L1 direction + L2 magnitude sur une cible numérique."""
    if source_df is None or target_col not in source_df.columns:
        return {"status": "missing_target"}
    if control_col not in source_df.columns:
        nums = source_df.select_dtypes(include=[np.number]).columns.tolist()
        control_col = nums[0] if nums else source_df.columns[0]

    ctrl = pd.to_numeric(source_df[control_col], errors="coerce").values.astype(float)
    tgt = pd.to_numeric(source_df[target_col], errors="coerce").values.astype(float)
    m = np.isfinite(ctrl) & np.isfinite(tgt)
    ctrl, tgt = ctrl[m], tgt[m]
    if len(ctrl) < 30:
        return {"status": "too_short", "n": len(ctrl)}

    progress = 100.0 * stats.rankdata(ctrl, method="average") / len(ctrl)
    threshold = data["threshold"]
    n = len(progress)
    n_hold = max(5, int(n * HOLD_OUT_FRAC))
    train_p, train_y = progress[:-n_hold], tgt[:-n_hold]
    test_p, test_y = progress[-n_hold:], tgt[-n_hold:]

    model = _fit_poly(train_p, train_y, 2)
    model_pre = _fit_poly(train_p[train_p < threshold], train_y[train_p < threshold], 2)
    model_post = _fit_poly(train_p[train_p >= threshold], train_y[train_p >= threshold], 2)

    last_p, last_y = float(progress[-1]), float(tgt[-1])
    future_p = np.linspace(last_p, min(100.0, last_p + horizon * 2.5), horizon)
    pred_full = _pred_poly(model, future_p)
    pred_pre = _pred_poly(model_pre, future_p)
    pred_post = _pred_poly(model_post, future_p)

    if last_p >= threshold:
        w_post = 0.75
    else:
        dist = max(0.0, (threshold - last_p) / max(threshold, 1))
        w_post = float(np.clip(0.25 + 0.5 * (1.0 - dist), 0.2, 0.8))
    pred_mix = (1 - w_post) * pred_pre + w_post * pred_post
    if not np.any(np.isfinite(pred_mix)):
        pred_mix = pred_full

    mc = mc_paths(progress, tgt, threshold, horizon, n_sc, risk_prof, seed)
    future_med = float(mc["median"][-1]) if mc is not None else float(np.nanmedian(pred_mix))
    rel = (future_med - last_y) / (abs(last_y) + 1e-9)
    neutral = risk_prof["neutral_band"]
    direction = "up" if rel > neutral else ("down" if rel < -neutral else "flat")

    val = {}
    if model is not None and len(test_p) >= 3:
        pred_test = _pred_poly(model, test_p)
        mm = np.isfinite(test_y) & np.isfinite(pred_test)
        if mm.sum() >= 3:
            yt, yp = test_y[mm], pred_test[mm]
            mae = float(np.mean(np.abs(yt - yp)))
            mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9)))) * 100
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-12
            r2 = float(np.clip(1.0 - ss_res / ss_tot, -2.0, 1.0))
            true_dir = np.sign(np.diff(yt))
            pred_dir = np.sign(np.diff(yp))
            da = float(np.mean(true_dir == pred_dir)) if len(true_dir) else None
            val = {"mae": mae, "mape": mape, "r2": r2, "directional_accuracy": da, "n": int(mm.sum())}

    band_low = float(mc["q_low"][-1]) if mc is not None else future_med
    band_high = float(mc["q_high"][-1]) if mc is not None else future_med

    # Confiance locale L1/L2
    conf = 0.35
    if val.get("directional_accuracy") is not None:
        conf += 0.35 * val["directional_accuracy"]
    if val.get("r2") is not None and val["r2"] > 0:
        conf += 0.20 * min(1.0, val["r2"])
    if mc is not None:
        width = (band_high - band_low) / (abs(future_med) + 1e-9)
        conf += 0.10 * float(np.clip(1.0 - width, 0, 1))
    conf = float(np.clip(conf, 0.05, 0.95))

    return {
        "status": "ok",
        "target_col": target_col,
        "control_col": control_col,
        "last_value": last_y,
        "forecast_median": future_med,
        "relative_change": float(rel),
        "direction": direction,
        "band_low": band_low,
        "band_high": band_high,
        "validation": val,
        "confidence": conf,
        "w_post_regime": w_post,
        "last_progress": last_p,
        "mc": {
            "median": mc["median"].tolist(),
            "q_low": mc["q_low"].tolist(),
            "q_high": mc["q_high"].tolist(),
            "future_progress": mc["future_progress"].tolist(),
        } if mc else None,
    }


def level_regime(data, horizon, risk_prof):
    landscape = data["landscape"]
    threshold = data["threshold"]
    progress, series = extract_metric(landscape, "Hurst")
    if progress is None or series is None:
        progress, series = extract_metric(landscape, "DFA")
    if progress is None or series is None:
        return {"status": "insufficient", "crossing_prob": None}

    last_p = float(progress[-1])
    ews = ews_series(series, progress, threshold) if series is not None else None
    alert = ews["current_alert"] if ews and ews["current_alert"] is not None else 0.3

    if last_p >= threshold:
        state = "post"
        p_post = float(np.clip(0.55 + 0.35 * min(1.0, (last_p - threshold) / 20.0), 0.55, 0.95))
    else:
        dist = (threshold - last_p) / 100.0
        approach = max(0.0, 1.0 - abs(dist) * 4.0)
        p_post = float(np.clip(0.12 + 0.55 * approach + 0.30 * alert, 0.05, 0.92))
        # Proactif : si EWS haut et encore loin du seuil
        lead = risk_prof["proactive_lead"]
        if alert >= risk_prof["alert_threshold"] and dist > lead:
            state = "transition_early"  # signal anticipé
        elif dist < lead:
            state = "transition"
        else:
            state = "pre"

    return {
        "status": "ok",
        "state": state,
        "crossing_prob": p_post,
        "p_pre": 1.0 - p_post,
        "last_progress": last_p,
        "threshold": threshold,
        "already_post": last_p >= threshold,
        "ews_alert": alert,
        "proactive": state == "transition_early",
        "confidence": float(np.clip(0.4 + 0.3 * abs(p_post - 0.5) * 2 + 0.2 * (alert or 0), 0.1, 0.9)),
    }


def level_risk(data, risk_prof, complexity_alerts: List[Dict]):
    noise = data.get("noise_diag") or {}
    csd = data.get("csd_ranking") or []
    avg_proxy = _sf(noise.get("avg_variance_proxy_score"), 0.0)
    top_csd = 0.0
    if csd:
        try:
            top_csd = float(csd[0][1]) if isinstance(csd[0], (list, tuple)) else float(csd[0].get("score", 0))
        except Exception:
            top_csd = 0.0
    mean_alert = float(np.mean([a["alert"] for a in complexity_alerts])) if complexity_alerts else 0.0
    risk_score = float(np.clip(0.40 * avg_proxy + 0.25 * top_csd + 0.35 * mean_alert, 0, 1))
    level = "high" if risk_score >= 0.65 else ("medium" if risk_score >= 0.40 else "low")
    return {
        "status": "ok",
        "risk_score": risk_score,
        "level": level,
        "avg_csd_proxy": avg_proxy,
        "top_csd": top_csd,
        "n_complexity_alerts": len(complexity_alerts),
        "alerts": complexity_alerts,
        "confidence": float(np.clip(0.5 + 0.3 * abs(risk_score - 0.5) * 2, 0.2, 0.9)),
    }


def level_complexity(data, horizon, n_sc, risk_prof, seed):
    landscape = data["landscape"]
    threshold = data["threshold"]
    ranking = []
    alerts = []
    details = {}
    for metric in ALL_METRICS:
        progress, series = extract_metric(landscape, metric)
        if progress is None or series is None or np.sum(np.isfinite(series)) < 10:
            continue
        model = _fit_poly(progress, series, 2)
        ews = ews_series(series, progress, threshold)
        alert = ews["current_alert"] if ews else None
        r2 = model["r2"] if model else None
        score = 0.0
        if r2 is not None and np.isfinite(r2):
            score += 0.5 * max(0.0, r2)
        if alert is not None and np.isfinite(alert):
            score += 0.5 * alert
            if alert >= risk_prof["alert_threshold"]:
                alerts.append({"metric": metric, "alert": float(alert)})
        ranking.append({"metric": metric, "score": float(score), "r2": r2, "alert": alert})
        details[metric] = {"r2": r2, "alert": alert, "observed_last": float(series[-1]) if np.isfinite(series[-1]) else None}
    ranking.sort(key=lambda x: x["score"], reverse=True)
    conf = 0.3
    if ranking:
        conf = float(np.clip(0.3 + 0.5 * ranking[0]["score"] + 0.1 * min(len(ranking), 5) / 5, 0.1, 0.9))
    return {
        "status": "ok",
        "ranking": ranking,
        "alerts": alerts,
        "top_metric": ranking[0]["metric"] if ranking else None,
        "details": details,
        "confidence": conf,
    }


def level_next_threshold(data):
    scan = data.get("scan_curve") or []
    thr = data["threshold"]
    if not scan:
        return {"status": "no_scan", "current_threshold": thr, "next_zone": None, "confidence": 0.2}
    scores = [(s["candidate"], s["score"]) for s in scan if s.get("score") is not None]
    if len(scores) < 5:
        return {"status": "sparse", "current_threshold": thr, "next_zone": None, "confidence": 0.25}
    cands = np.array([c for c, _ in scores])
    sc = np.array([v for _, v in scores])
    neigh = np.abs(cands - thr) > 8.0
    if not np.any(neigh):
        return {"status": "single_peak", "current_threshold": thr, "next_zone": None, "confidence": 0.3}
    idx = int(np.argmax(sc[neigh]))
    next_c = float(cands[neigh][idx])
    next_s = float(sc[neigh][idx])
    return {
        "status": "ok",
        "current_threshold": thr,
        "next_zone": next_c,
        "next_score": next_s,
        "delta": next_c - thr,
        "confidence": float(np.clip(0.35 + 0.4 * min(1.0, next_s / (np.nanmax(sc) + 1e-9)), 0.2, 0.85)),
    }


# =============================================================================
# L7 COHÉRENCE + SYNTHÈSE PROACTIVE
# =============================================================================

def level_coherence(L1, L2_from_L1, L3, L4, L5, L6, domain_weights, risk_prof, lex):
    """
    Agrège les niveaux en un verdict multi-objectifs robuste.
    - Vote directionnel pondéré
    - Score de confiance global (pénalité si contradiction)
    - Flag proactif
    """
    # Direction votes
    votes = {"up": 0.0, "down": 0.0, "flat": 0.0}
    if L1.get("status") == "ok":
        d = L1["direction"]
        w = domain_weights["L1_direction"] * L1.get("confidence", 0.5)
        votes[d] = votes.get(d, 0) + w
        # Magnitude renforce si |rel| grand
        rel = abs(L1.get("relative_change") or 0)
        if rel > risk_prof["neutral_band"] * 2:
            votes[d] += domain_weights["L2_magnitude"] * 0.5 * L1.get("confidence", 0.5)

    # Régime : post / transition → plutôt down en finance, neutre ailleurs (générique)
    if L3.get("status") == "ok":
        w = domain_weights["L3_regime"] * L3.get("confidence", 0.5)
        if L3.get("state") in ("post", "transition") and L3.get("crossing_prob", 0) > 0.6:
            # En finance, post-seuil souvent distribution ; ailleurs on reste neutre
            votes["down"] += w * 0.4
            votes["flat"] += w * 0.3
        elif L3.get("state") == "pre":
            votes["flat"] += w * 0.3
            votes["up"] += w * 0.2

    # Risque élevé → prudence (tire vers flat/down)
    if L4.get("status") == "ok" and L4.get("level") == "high":
        w = domain_weights["L4_risk"] * L4.get("confidence", 0.5)
        votes["down"] += w * 0.35
        votes["flat"] += w * 0.35

    # Complexité : alertes hautes = instabilité
    if L5.get("status") == "ok" and L5.get("alerts"):
        w = domain_weights["L5_complexity"] * L5.get("confidence", 0.5)
        votes["flat"] += w * 0.3
        votes["down"] += w * 0.2

    total = sum(votes.values()) + 1e-12
    probs = {k: v / total for k, v in votes.items()}
    consensus_dir = max(probs, key=probs.get)
    consensus_strength = probs[consensus_dir]

    # Contradiction : si L1 dit up et consensus down (ou inverse)
    contradiction = 0.0
    if L1.get("status") == "ok":
        if L1["direction"] != consensus_dir and L1["direction"] != "flat" and consensus_dir != "flat":
            contradiction = 0.35
        elif L1["direction"] == "flat" and consensus_strength > 0.5:
            contradiction = 0.1

    # Confiance globale
    confs = []
    for block, key in [
        (L1, "L1_direction"),
        (L3, "L3_regime"),
        (L4, "L4_risk"),
        (L5, "L5_complexity"),
        (L6, "L6_next_threshold"),
    ]:
        if block.get("status") in ("ok",) or block.get("confidence") is not None:
            c = block.get("confidence")
            if c is not None:
                confs.append(c * domain_weights.get(key, 0.1))
    base_conf = float(np.sum(confs)) if confs else 0.3
    # Normaliser approximativement
    base_conf = float(np.clip(base_conf / max(sum(domain_weights.get(k, 0) for k in ["L1_direction", "L3_regime", "L4_risk", "L5_complexity", "L6_next_threshold"]), 1e-9), 0, 1))
    global_conf = float(np.clip(base_conf * consensus_strength * (1.0 - contradiction), 0.05, 0.95))

    proactive = bool(L3.get("proactive")) or (
        L4.get("level") == "high" and L3.get("state") == "pre"
    )

    # Score composite actionnable 0–1 (proactivité × confiance × intensité)
    intensity = abs(L1.get("relative_change") or 0) if L1.get("status") == "ok" else 0.0
    intensity = float(np.clip(intensity / 0.1, 0, 1))
    action_score = float(np.clip(
        0.40 * global_conf
        + 0.25 * consensus_strength
        + 0.20 * intensity
        + 0.15 * (1.0 if proactive else 0.0),
        0, 1,
    ))

    return {
        "status": "ok",
        "consensus_direction": consensus_dir,
        "direction_probs": probs,
        "consensus_strength": float(consensus_strength),
        "contradiction_penalty": contradiction,
        "global_confidence": global_conf,
        "proactive": proactive,
        "action_score": action_score,
        "recommendation": _recommendation(consensus_dir, global_conf, proactive, L4, L3, lex),
    }


def _recommendation(direction, conf, proactive, L4, L3, lex):
    risk_level = L4.get("level", "medium") if L4 else "medium"
    risk_txt = lex["risk_high"] if risk_level == "high" else (lex["risk_med"] if risk_level == "medium" else lex["risk_low"])
    dir_txt = lex.get(direction, direction)
    parts = [f"Consensus : {dir_txt} (confiance {conf*100:.0f}%)."]
    if proactive:
        parts.append(f"Signal proactif : {lex['proactive']}.")
    if L3.get("state") == "transition_early":
        parts.append(f"État : {lex['transition']} (anticipé).")
    elif L3.get("state"):
        st = L3["state"]
        if st == "pre":
            parts.append(f"Régime : {lex['pre']}.")
        elif st == "post":
            parts.append(f"Régime : {lex['post']}.")
        else:
            parts.append(f"Régime : {lex['transition']}.")
    parts.append(f"Risque : {risk_txt}.")
    if conf < 0.35:
        parts.append("Prudence : signaux peu convergents — horizon à interpréter avec réserve.")
    elif conf >= 0.65 and proactive:
        parts.append("Niveau d'attention élevé : convergence + signal anticipé.")
    return " ".join(parts)


# =============================================================================
# NARRATION
# =============================================================================

def narrate_multilevel(results: Dict, domain: str, risk_key: str) -> str:
    lex = DOMAIN_LEXICON.get(domain, DOMAIN_LEXICON["generic"])
    risk_label = RISK_PROFILES[risk_key]["label"]
    lines = []
    lines.append("=" * 70)
    lines.append("PRÉDICTION MULTI-NIVEAUX — UNIVERSAL_LOG")
    lines.append("=" * 70)
    meta = results["meta"]
    lines.append(f"Domaine : {domain}  |  Risque : {risk_label}  |  Horizon : {meta['horizon']}")
    lines.append(f"Cible : {meta.get('target_col')}  |  Contrôle : {meta.get('control_col')}")
    lines.append(f"Seuil pipeline : {meta['threshold']:.2f}% ({meta['threshold_source']})")
    lines.append("")

    L7 = results["levels"].get("L7_coherence", {})
    lines.append("— SYNTHÈSE —")
    lines.append(L7.get("recommendation", "n/a"))
    lines.append(
        f"Score d'action : {L7.get('action_score', 0):.2f}  |  "
        f"Confiance globale : {L7.get('global_confidence', 0)*100:.0f}%  |  "
        f"Proactif : {'OUI' if L7.get('proactive') else 'non'}"
    )
    probs = L7.get("direction_probs") or {}
    if probs:
        lines.append(
            "Prob. direction : "
            + " | ".join(f"{k}={v*100:.0f}%" for k, v in sorted(probs.items(), key=lambda x: -x[1]))
        )
    lines.append("")

    L1 = results["levels"].get("L1_L2_target", {})
    if L1.get("status") == "ok":
        lines.append("— L1/L2 CIBLE —")
        lines.append(f"Direction : {lex.get(L1['direction'], L1['direction'])}")
        lines.append(
            f"Dernière valeur : {L1['last_value']:.6g} → médiane {L1['forecast_median']:.6g} "
            f"({L1['relative_change']*100:+.2f}%)"
        )
        lines.append(f"Bande ({risk_label}) : [{L1['band_low']:.6g} ; {L1['band_high']:.6g}]")
        val = L1.get("validation") or {}
        if val:
            da = val.get("directional_accuracy")
            da_s = f"{da*100:.1f}%" if da is not None else "n/a"
            lines.append(f"Hold-out : R²={val.get('r2')}  MAPE={val.get('mape')}%  dir_acc={da_s}")
        lines.append(f"Confiance L1/L2 : {L1.get('confidence', 0)*100:.0f}%")
        lines.append("")

    L3 = results["levels"].get("L3_regime", {})
    if L3.get("status") == "ok":
        lines.append("— L3 RÉGIME —")
        lines.append(
            f"État : {L3.get('state')}  |  P(post)={L3.get('crossing_prob', 0)*100:.1f}%  |  "
            f"proactif={L3.get('proactive')}"
        )
        lines.append("")

    L4 = results["levels"].get("L4_risk", {})
    if L4.get("status") == "ok":
        lines.append("— L4 RISQUE —")
        lines.append(
            f"Score={L4.get('risk_score', 0):.3f}  niveau={L4.get('level')}  "
            f"CSD_proxy={L4.get('avg_csd_proxy')}  alertes_complexité={L4.get('n_complexity_alerts')}"
        )
        lines.append("")

    L5 = results["levels"].get("L5_complexity", {})
    if L5.get("status") == "ok" and L5.get("ranking"):
        lines.append("— L5 COMPLEXITÉ —")
        lines.append("Top : " + ", ".join(f"{r['metric']}({r['score']:.2f})" for r in L5["ranking"][:4]))
        if L5.get("alerts"):
            lines.append("Alertes : " + ", ".join(f"{a['metric']}={a['alert']:.2f}" for a in L5["alerts"]))
        lines.append("")

    L6 = results["levels"].get("L6_next_threshold", {})
    if L6.get("next_zone") is not None:
        lines.append("— L6 PROCHAIN SEUIL —")
        lines.append(
            f"Actuel={L6['current_threshold']:.1f}%  candidat≈{L6['next_zone']:.1f}%  "
            f"(Δ={L6['delta']:+.1f})"
        )
        lines.append("")

    lines.append(f"Unité de lecture : {lex['unit']}")
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_multilevel(
    pipeline_dir: str,
    domain: str = "finance",
    risk: str = "moderate",
    target_col: str = "Close",
    control_col: str = "Energy_Diff",
    source_csv: Optional[str] = None,
    horizon: int = DEFAULT_HORIZON,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    if domain not in DOMAIN_LEVEL_WEIGHTS:
        domain = "generic"
    if risk not in RISK_PROFILES:
        risk = "moderate"

    risk_prof = RISK_PROFILES[risk]
    weights = DOMAIN_LEVEL_WEIGHTS[domain]
    lex = DOMAIN_LEXICON[domain]
    data = load_pipeline(Path(pipeline_dir))
    source_df = load_source(source_csv)

    print("=" * 66)
    print("UNIVERSAL_LOG — Prédiction Multi-Niveaux v1")
    print("=" * 66)
    print(f"Domaine={domain}  Risque={risk}  Cible={target_col}  Contrôle={control_col}")
    print(f"Seuil={data['threshold']:.2f}%  Horizon={horizon}  Scénarios={n_scenarios}")

    # L5 d'abord (fournit alertes pour L4)
    L5 = level_complexity(data, horizon, n_scenarios, risk_prof, seed)
    L4 = level_risk(data, risk_prof, L5.get("alerts") or [])
    L3 = level_regime(data, horizon, risk_prof)
    L6 = level_next_threshold(data)
    L1 = level_target(source_df, target_col, control_col, data, horizon, n_scenarios, risk_prof, seed)
    L7 = level_coherence(L1, L1, L3, L4, L5, L6, weights, risk_prof, lex)

    results = {
        "meta": {
            "domain": domain,
            "risk": risk,
            "target_col": target_col,
            "control_col": control_col,
            "horizon": horizon,
            "n_scenarios": n_scenarios,
            "seed": seed,
            "threshold": data["threshold"],
            "threshold_source": data["threshold_source"],
            "pipeline_dir": str(pipeline_dir),
            "timestamp": datetime.now().isoformat(),
            "level_weights": weights,
        },
        "levels": {
            "L1_L2_target": L1,
            "L3_regime": L3,
            "L4_risk": L4,
            "L5_complexity": L5,
            "L6_next_threshold": L6,
            "L7_coherence": L7,
        },
    }
    results["narrative"] = narrate_multilevel(results, domain, risk)
    print("\n" + results["narrative"])
    return results


def export_multilevel(results: Dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    with open(output_dir / "multilevel_prediction.json", "w", encoding="utf-8") as f:
        json.dump(clean(results), f, indent=2, ensure_ascii=False)
    with open(output_dir / "narrative.txt", "w", encoding="utf-8") as f:
        f.write(results.get("narrative", ""))

    # Radar des niveaux
    try:
        levels = results["levels"]
        labels = ["Direction", "Magnitude", "Régime", "Risque", "Complexité", "Seuil+", "Cohérence"]
        confs = [
            levels.get("L1_L2_target", {}).get("confidence") or 0,
            levels.get("L1_L2_target", {}).get("confidence") or 0,
            levels.get("L3_regime", {}).get("confidence") or 0,
            levels.get("L4_risk", {}).get("confidence") or 0,
            levels.get("L5_complexity", {}).get("confidence") or 0,
            levels.get("L6_next_threshold", {}).get("confidence") or 0,
            levels.get("L7_coherence", {}).get("global_confidence") or 0,
        ]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        confs_p = confs + confs[:1]
        angles_p = angles + angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.plot(angles_p, confs_p, "o-", lw=2, color="steelblue")
        ax.fill(angles_p, confs_p, alpha=0.25, color="steelblue")
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title("Confiance par niveau de prédiction")
        plt.tight_layout()
        plt.savefig(output_dir / "levels_radar.png", dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠ radar: {e}")

    # Forecast cible
    L1 = results["levels"].get("L1_L2_target", {})
    if L1.get("status") == "ok" and L1.get("mc"):
        mc = L1["mc"]
        fp = np.array(mc["future_progress"])
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(fp, mc["median"], color="crimson", lw=2, label="Médiane")
        ax.fill_between(fp, mc["q_low"], mc["q_high"], color="orange", alpha=0.25, label="Bande")
        ax.axhline(L1["last_value"], color="gray", ls=":", label="Dernière valeur")
        ax.set_title(f"Prévision multi-niveaux — {L1.get('target_col')}")
        ax.set_xlabel("Progression (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "target_forecast.png", dpi=140, bbox_inches="tight")
        plt.close()

    print(f"✓ Exports → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="UNIVERSAL_LOG — Prédiction Multi-Niveaux (proactive, multi-objectifs, multi-domaine)"
    )
    parser.add_argument("--pipeline-dir", "-p", required=True)
    parser.add_argument("--domain", "-d", default="finance", choices=list(DOMAIN_LEVEL_WEIGHTS.keys()))
    parser.add_argument("--risk", "-r", default="moderate", choices=list(RISK_PROFILES.keys()))
    parser.add_argument("--target", default="Close")
    parser.add_argument("--control", default="Energy_Diff")
    parser.add_argument("--source", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else Path(args.pipeline_dir) / "multilevel_predictions"
    results = run_multilevel(
        pipeline_dir=args.pipeline_dir,
        domain=args.domain,
        risk=args.risk,
        target_col=args.target,
        control_col=args.control,
        source_csv=args.source,
        horizon=args.horizon,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
    )
    export_multilevel(results, out)
    print("\n✓ Prédiction multi-niveaux terminée")


if __name__ == "__main__":
    main()
