# -*- coding: utf-8 -*-
"""Module dérivatif CSD séparé pour les dix exécutions Apamea.

Ne modifie ni n'exécute v2/v3. Les seuils sont lus dans leurs exports.
La sélection CSD suit la règle native score >= 0.4, et non un nouveau test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path('/home/ubuntu')
OUT = BASE / 'module_derivatif_csd_apamea'
OUT.mkdir(exist_ok=True)
DATA = pd.read_csv(BASE / 'apamea_experimental_M_control.csv')
CONTROLS = ['M', 'E', 'Flip', 'Cluster', 'LocalVar']
RUN_DIR = {
    ('v2', 'M'): BASE / 'apamea_v2_results',
    ('v3', 'M'): BASE / 'apamea_v3_results',
    **{(v, c): BASE / f'apamea_{v}_{c}_control_results' for v in ['v2', 'v3'] for c in ['E', 'Flip', 'Cluster', 'LocalVar']},
}
sns.set_theme(style='whitegrid', context='talk')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold'})


def prepared_flux(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-10 else np.zeros_like(arr)


def unique_curve(progress: np.ndarray, flux: np.ndarray, mask: np.ndarray):
    frame = pd.DataFrame({'progress': progress[mask], 'flux': flux[mask]})
    curve = frame.groupby('progress', as_index=False)['flux'].mean().sort_values('progress')
    return curve['progress'].to_numpy(), curve['flux'].to_numpy()


def quadratic_summary(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or np.ptp(x) <= 1e-12:
        return {'a': np.nan, 'b': np.nan, 'second_derivative': np.nan, 'n_unique_points': len(x)}
    centered = x - x.mean()
    a, b, _ = np.polyfit(centered, y, 2)
    return {'a': float(a), 'b': float(b), 'second_derivative': float(2*a), 'n_unique_points': len(x)}

summary_rows, curve_rows, cross_rows, selection_rows = [], [], [], []
for (version, control), directory in RUN_DIR.items():
    results = json.loads((directory / 'pipeline_results.json').read_text())
    threshold = float(results['metadata']['opt_threshold'])
    progress = 100 * rankdata(DATA[control].to_numpy(), method='average') / len(DATA)
    pre_mask, post_mask = progress < threshold, progress >= threshold
    diagnostics = {d['flux_name']: d for d in results.get('noise_diagnostic', {}).get('all_flux_diagnostics', [])}
    selected = []
    for flux, diag in diagnostics.items():
        score = diag.get('variance_proxy_score')
        passes = diag.get('status') == 'computed' and score is not None and np.isfinite(score) and score >= 0.4
        selection_rows.append({
            'version': version, 'control': control, 'flux': flux, 'threshold_pct': threshold,
            'native_proxy_score': score, 'native_status': diag.get('status'), 'selected_native_CSD': passes,
            'q5_candidate': diag.get('cand_q5_score'), 'q5_white_noise': diag.get('white_q5_score'),
            'delta_volatility_corr': diag.get('delta_corr'), 'avg_hurst': diag.get('avg_hurst'), 'fft_slope_after': diag.get('slope_after')
        })
        if not passes:
            continue
        selected.append(flux)
        flux_z = prepared_flux(DATA[flux])
        x_pre, y_pre = unique_curve(progress, flux_z, pre_mask)
        x_post, y_post = unique_curve(progress, flux_z, post_mask)
        pre_fit = quadratic_summary(x_pre, y_pre)
        post_fit = quadratic_summary(x_post, y_post)
        summary_rows.append({
            'version': version, 'control': control, 'flux': flux, 'threshold_pct': threshold,
            'n_pre_images': int(pre_mask.sum()), 'n_post_images': int(post_mask.sum()),
            'n_pre_unique_progress': pre_fit['n_unique_points'], 'n_post_unique_progress': post_fit['n_unique_points'],
            'F1_prime_post_b': post_fit['b'], 'F2_second_post_2a': post_fit['second_derivative'],
            'F1_prime_pre_b': pre_fit['b'], 'F2_second_pre_2a': pre_fit['second_derivative'],
            'native_proxy_score': score
        })
        for segment, x, y, fit in [('pre', x_pre, y_pre, pre_fit), ('post', x_post, y_post, post_fit)]:
            if len(x) < 3 or np.ptp(x) <= 1e-12:
                continue
            centered = x - x.mean()
            f1 = 2 * fit['a'] * centered + fit['b']
            f2 = np.full_like(f1, fit['second_derivative'], dtype=float)
            for xi, yi, fi, si in zip(x, y, f1, f2):
                curve_rows.append({'version': version, 'control': control, 'flux': flux, 'segment': segment,
                                   'progress_pct': xi, 'flux_zscore_mean': yi, 'F1_prime': fi, 'F2_second': si,
                                   'threshold_pct': threshold})
    selected_summary = pd.DataFrame([r for r in summary_rows if r['version'] == version and r['control'] == control])
    finite_summary = selected_summary.dropna(subset=['F1_prime_post_b', 'F2_second_pre_2a']).copy()
    if len(finite_summary) >= 3 and finite_summary['F1_prime_post_b'].nunique() > 1 and finite_summary['F2_second_pre_2a'].nunique() > 1:
        x = finite_summary['F1_prime_post_b'].to_numpy(dtype=float)
        y = finite_summary['F2_second_pre_2a'].to_numpy(dtype=float)
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        status = 'computed_descriptive'
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        status = 'insufficient_finite_pre_post_derivatives'
    cross_rows.append({'version': version, 'control': control, 'threshold_pct': threshold,
                       'n_selected_CSD_fluxes': len(selected_summary), 'n_finite_F1post_F2pre': len(finite_summary), 'selected_fluxes': ', '.join(selected),
                       'pearson_F1post_vs_F2pre': pearson_r, 'pearson_p_descriptive_only': pearson_p,
                       'spearman_F1post_vs_F2pre': spearman_r, 'spearman_p_descriptive_only': spearman_p,
                       'status': status})

summary = pd.DataFrame(summary_rows)
curves = pd.DataFrame(curve_rows)
selection = pd.DataFrame(selection_rows)
cross = pd.DataFrame(cross_rows)
selection.to_csv(OUT / '01_selection_CSD_native_4_tests.csv', index=False)
summary.to_csv(OUT / '02_resumes_F1prime_F2second.csv', index=False)
curves.to_csv(OUT / '03_scans_derivatifs_complets.csv', index=False)
cross.to_csv(OUT / '04_correlations_F1post_vs_F2pre.csv', index=False)

# Une figure par run : les scans post-seuil de F1' et F2'', uniquement pour les flux CSD natifs sélectionnés.
for version in ['v2', 'v3']:
    for control in CONTROLS:
        item = curves[(curves['version'] == version) & (curves['control'] == control) & (curves['segment'] == 'post')]
        if item.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for flux, part in item.groupby('flux'):
            axes[0].plot(part['progress_pct'], part['F1_prime'], marker='o', ms=3, label=flux)
            axes[1].plot(part['progress_pct'], part['F2_second'], marker='o', ms=3, label=flux)
        threshold = item['threshold_pct'].iloc[0]
        axes[0].axhline(0, color='black', lw=.8)
        axes[1].axhline(0, color='black', lw=.8)
        axes[0].set_title("F1′ post-seuil : dérivée première")
        axes[1].set_title("F2″ post-seuil : dérivée seconde")
        for ax in axes:
            ax.set_xlabel('Progression du contrôle (%)')
            ax.legend(title='Flux CSD', fontsize=10)
        axes[0].set_ylabel("F1′ du flux z-score")
        axes[1].set_ylabel("F2″ du flux z-score")
        fig.suptitle(f'{version}, contrôle {control} — seuil MC inchangé {threshold:.3f}%')
        fig.tight_layout()
        fig.savefig(OUT / f'05_scan_post_{version}_{control}.png', dpi=220, bbox_inches='tight')
        plt.close(fig)

# Carte des corrélations descriptives demandées à travers les flux CSD sélectionnés.
for coeff, title, filename in [
    ('pearson_F1post_vs_F2pre', 'Pearson : F1′ post-seuil vs F2″ pré-seuil', '06_correlation_Pearson_F1post_F2pre.png'),
    ('spearman_F1post_vs_F2pre', 'Spearman : F1′ post-seuil vs F2″ pré-seuil', '07_correlation_Spearman_F1post_F2pre.png'),
]:
    matrix = cross.pivot(index='control', columns='version', values=coeff).reindex(index=CONTROLS, columns=['v2', 'v3'])
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap='vlag', annot=True, fmt='.3f', linewidths=.5,
                cbar_kws={'label': 'Corrélation descriptive'}, ax=ax)
    ax.set_title(title + '\nà travers les flux CSD individuels sélectionnés')
    ax.set_xlabel('Pipeline')
    ax.set_ylabel('Colonne de contrôle')
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches='tight')
    plt.close(fig)

print('SELECTION CSD NATIVE')
print(selection.to_string(index=False))
print('\nRESUMES F1/F2')
print(summary.to_string(index=False))
print('\nCORRELATIONS CROISEES DESCRIPTIVES')
print(cross.to_string(index=False))
