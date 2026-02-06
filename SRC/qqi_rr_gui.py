# -*- coding: utf-8 -*-
r"""
qqi_rr_gui.py — App Tkinter para:
1) Carregar nuvem (xlsx/csv/txt)
2) Informar GSD
3) Calcular QQI e estimar RR por faixas de GSD (0,25 m)
4) Mostrar erro padrão (±MAE) por faixa (calibração)
5) Gerar mapa XY colorido de RR estimado e abrir com 1 clique

Calibração:
- Lê dados brutos (GSD, QQI (10^-4), RR) diretamente do Excel:
  D:\Users\Gleicon\Documents\DOUTORADO - UFRGS\TESE\Calibrador do QQI_RR_GUI.xlsx

Robustez:
- Winsorização por grupo (GSD_bin, RR) em P5–P95 (quando N >= 8)
- Fallback por N mínimo: se grupo pequeno, usa mediana global do RR
- FIXES incluídos:
  (1) get_rule_for_gsd inclui limite inferior (>=lo) para evitar "None" em bordas do bin
  (2) fallback robusto: evita KeyError quando pivotN não tem coluna RR
  (3) aviso quando GSD está fora do intervalo calibrado
"""

import os
import sys
import math
import pickle
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ============= Configurações =============
RADIUS = 0.5
N_JOBS = -1
QQI_ESCALA = 1e4               # escala aplicada no QQI calculado (para comparar com calibração)
RESULTS_DIRNAME = "Resultados_QQI_RR"
MAX_RR_HALFSTEP = 5.5

# ============= Caminho da calibração (Excel) =============
CALIB_XLSX_PATH = r"D:\Users\Gleicon\Documents\DOUTORADO - UFRGS\TESE\Calibrador do QQI_RR_GUI.xlsx"

# ============= Robustez (outliers / bins instáveis) =============
WINSOR_LO = 0.05
WINSOR_HI = 0.95
WINSOR_MIN_N = 8     # só aplica winsor se o grupo tiver N>=8
GROUP_MIN_N  = 5     # se (GSD_bin,RR) tiver N < 5, usa mediana global do RR (fallback)

# ============= Lógica de bins =============
def make_bins(gsd_min, gsd_max, step=0.25, start_floor=0.50):
    start = max(start_floor, math.floor((gsd_min - start_floor) / step) * step + start_floor)
    stop  = math.ceil((gsd_max - start_floor) / step) * step + start_floor + step
    bins = np.arange(start, stop, step)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    return bins, labels

def label_to_bounds(label: str):
    lo, hi = label.split("-")
    return float(lo), float(hi)

def enforce_monotonic_series(values):
    y = values.copy()
    for i in range(1, len(y)):
        if y[i] < y[i-1]:
            y[i] = y[i-1]
    return y

# ============= Carregamento da calibração (Excel) =============
def load_calibration_excel(path: str) -> pd.DataFrame:
    """
    Espera colunas: GSD, QQI (10^-4), RR
    Aceita variações no nome (ex.: QQI, QQI (10^-4), QQI(10^-4) etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de calibração não encontrado:\n{path}")

    df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]
    cols_upper = {c.upper(): c for c in df.columns}

    def pick_col(candidates):
        for cand in candidates:
            if cand.upper() in cols_upper:
                return cols_upper[cand.upper()]
        return None

    col_gsd = pick_col(["GSD"])
    col_rr  = pick_col(["RR"])
    col_qqi = pick_col(["QQI (10^-4)", "QQI(10^-4)", "QQI 10^-4", "QQI", "QQI (10^-4) "])

    if col_gsd is None or col_rr is None or col_qqi is None:
        raise ValueError(
            "Não foi possível identificar colunas no Excel.\n"
            "Esperado: GSD, QQI (10^-4), RR.\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    out = df[[col_gsd, col_qqi, col_rr]].copy()
    out.columns = ["GSD", "QQI", "RR"]

    out["GSD"] = pd.to_numeric(out["GSD"], errors="coerce")
    out["QQI"] = pd.to_numeric(out["QQI"], errors="coerce")  # já está em escala 10^-4 (compatível com qqi_cal)
    out["RR"]  = pd.to_numeric(out["RR"],  errors="coerce")

    out = out.dropna(subset=["GSD", "QQI", "RR"])
    out = out[(out["RR"] >= 2) & (out["RR"] <= 5)]

    if len(out) < 10:
        raise ValueError("Calibração muito pequena. Verifique se o Excel contém dados suficientes.")

    return out.reset_index(drop=True)

# ============= Robustez (winsor) =============
def winsorize_series(s: pd.Series, lo=WINSOR_LO, hi=WINSOR_HI) -> pd.Series:
    a = s.quantile(lo)
    b = s.quantile(hi)
    return s.clip(lower=a, upper=b)

# ============= Regras a partir da calibração (ROBUSTAS) =============
def compute_rules_from_calib(calib_df: pd.DataFrame, step=0.25) -> pd.DataFrame:
    df = calib_df.copy()

    bins, labels = make_bins(df["GSD"].min(), df["GSD"].max(), step=step, start_floor=0.50)
    df["GSD_bin"] = pd.cut(df["GSD"], bins=bins, labels=labels, include_lowest=True, right=True)

    rr_order = [2, 3, 4, 5]
    global_meds = df.groupby("RR")["QQI"].median().reindex(rr_order).astype(float).values

    # Winsor por grupo (GSD_bin, RR) quando N>=WINSOR_MIN_N
    grp = df.groupby(["GSD_bin", "RR"], observed=False)["QQI"]
    counts = grp.transform("size")

    df["QQI_w"] = df["QQI"].copy()
    mask_w = counts >= WINSOR_MIN_N
    if mask_w.any():
        df.loc[mask_w, "QQI_w"] = (
            df.loc[mask_w]
              .groupby(["GSD_bin", "RR"], observed=False)["QQI"]
              .transform(lambda s: winsorize_series(s, lo=WINSOR_LO, hi=WINSOR_HI))
        )

    med = df.groupby(["GSD_bin", "RR"], observed=False)["QQI_w"].median().reset_index()
    n_by = df.groupby(["GSD_bin", "RR"], observed=False)["QQI_w"].size().reset_index(name="N")
    med = med.merge(n_by, on=["GSD_bin", "RR"], how="left")

    pivot  = med.pivot(index="GSD_bin", columns="RR", values="QQI_w").reindex(labels)
    pivotN = med.pivot(index="GSD_bin", columns="RR", values="N").reindex(labels)

    # Fallback robusto (evita KeyError)
    for rr in rr_order:
        if rr in pivot.columns:
            if rr in pivotN.columns:
                bad = (pivotN[rr].isna()) | (pivotN[rr] < GROUP_MIN_N)
            else:
                bad = pd.Series(True, index=pivot.index)  # força fallback
            pivot.loc[bad, rr] = global_meds[rr_order.index(rr)]
        else:
            pivot[rr] = global_meds[rr_order.index(rr)]

    pivot_interp = pivot.astype(float).interpolate(method="linear", limit_direction="both")

    # Monotonicidade RR2<=RR3<=RR4<=RR5
    try:
        from sklearn.isotonic import IsotonicRegression
        def mono(row):
            rr = np.array(rr_order, dtype=float)
            y  = row.values.astype(float)
            y  = np.where(np.isnan(y), global_meds, y)
            iso = IsotonicRegression(increasing=True)
            y_fit = iso.fit_transform(rr, y)
            return pd.Series(y_fit, index=rr_order)
    except Exception:
        def mono(row):
            y = row.values.astype(float)
            y = np.where(np.isnan(y), global_meds, y)
            return pd.Series(enforce_monotonic_series(y), index=rr_order)

    fitted = pivot_interp.apply(mono, axis=1)

    T23 = (fitted[2] + fitted[3]) / 2.0
    T34 = (fitted[3] + fitted[4]) / 2.0
    T45 = (fitted[4] + fitted[5]) / 2.0

    rules = pd.DataFrame({
        "Faixa_GSD": fitted.index,
        "A_QQI_max_RR2": T23.values,
        "B_QQI_max_RR3": T34.values,
        "C_QQI_max_RR4": T45.values,
        "T23": T23.values, "T34": T34.values, "T45": T45.values,
        "Center_RR2": fitted[2].values, "Center_RR3": fitted[3].values,
        "Center_RR4": fitted[4].values, "Center_RR5": fitted[5].values,
    })

    bins_with_data = df["GSD_bin"].dropna().unique().tolist()
    return rules[rules["Faixa_GSD"].isin(bins_with_data)].reset_index(drop=True)

def get_rule_for_gsd(gsd: float, rules_df: pd.DataFrame):
    """FIX: inclui limite inferior (>= lo) para não perder bordas do bin."""
    for _, row in rules_df.iterrows():
        lo, hi = label_to_bounds(row["Faixa_GSD"])
        if (gsd >= lo) and (gsd <= hi):
            return row
    return None

def classify_rr(gsd: float, qqi_scaled: float, rules_df: pd.DataFrame) -> int | None:
    row = get_rule_for_gsd(gsd, rules_df)
    if row is not None:
        A, B, C = row["A_QQI_max_RR2"], row["B_QQI_max_RR3"], row["C_QQI_max_RR4"]
        if qqi_scaled <= A: return 2
        if qqi_scaled <= B: return 3
        if qqi_scaled <= C: return 4
        return 5
    return None

def estimate_rr_continuous(gsd: float, qqi_scaled: float, rules_df: pd.DataFrame) -> float | None:
    row = get_rule_for_gsd(gsd, rules_df)
    if row is not None:
        c2 = float(row["Center_RR2"]); c3 = float(row["Center_RR3"])
        c4 = float(row["Center_RR4"]); c5 = float(row["Center_RR5"])
        eps = 1e-12
        if not np.isfinite(c2) or not np.isfinite(c3) or not np.isfinite(c4) or not np.isfinite(c5):
            return None
        if abs(c3 - c2) < eps: c3 = c2 + eps
        if abs(c4 - c3) < eps: c4 = c3 + eps
        if abs(c5 - c4) < eps: c5 = c4 + eps

        if qqi_scaled <= c3:
            return 2.0 + (qqi_scaled - c2) / (c3 - c2)
        elif qqi_scaled <= c4:
            return 3.0 + (qqi_scaled - c3) / (c4 - c3)
        elif qqi_scaled <= c5:
            return 4.0 + (qqi_scaled - c4) / (c5 - c4)
        else:
            return 5.0 + (qqi_scaled - c5) / (c5 - c4)
    return None

def compute_rr_vectorized(qqi_array: np.ndarray, gsd: float, rules_df: pd.DataFrame) -> np.ndarray:
    row = get_rule_for_gsd(gsd, rules_df)
    if row is None:
        return np.full_like(qqi_array, np.nan, dtype=float)

    c2 = float(row["Center_RR2"]); c3 = float(row["Center_RR3"])
    c4 = float(row["Center_RR4"]); c5 = float(row["Center_RR5"])

    eps = 1e-12
    if abs(c3 - c2) < eps: c3 = c2 + eps
    if abs(c4 - c3) < eps: c4 = c3 + eps
    if abs(c5 - c4) < eps: c5 = c4 + eps

    q = qqi_array * QQI_ESCALA

    cond1 = q <= c3
    cond2 = (q > c3) & (q <= c4)
    cond3 = (q > c4) & (q <= c5)
    cond4 = q > c5

    rr = np.full_like(q, np.nan, dtype=float)
    rr[cond1] = 2.0 + (q[cond1] - c2) / (c3 - c2)
    rr[cond2] = 3.0 + (q[cond2] - c3) / (c4 - c3)
    rr[cond3] = 4.0 + (q[cond3] - c4) / (c5 - c4)
    rr[cond4] = 5.0 + (q[cond4] - c5) / (c5 - c4)

    return rr

def estimate_rr_halfstep(gsd: float, qqi_scaled: float, rules_df: pd.DataFrame,
                         max_rr: float = MAX_RR_HALFSTEP) -> float | None:
    rr_cont = estimate_rr_continuous(gsd, qqi_scaled, rules_df)
    if rr_cont is None or not np.isfinite(rr_cont):
        return None
    rr_cont = max(2.0, min(max_rr, rr_cont))
    return round(rr_cont * 2.0) / 2.0

def calib_mae_by_bin(calib_df: pd.DataFrame, rules_df: pd.DataFrame,
                     max_rr_halfstep: float = MAX_RR_HALFSTEP) -> pd.DataFrame:
    out_rows = []
    for _, r in rules_df.iterrows():
        label = r["Faixa_GSD"]; lo, hi = label_to_bounds(label)
        sub = calib_df[(calib_df["GSD"] > lo) & (calib_df["GSD"] <= hi)][["GSD","QQI","RR"]]
        if len(sub) == 0:
            out_rows.append({"Faixa_GSD": label, "MAE_05": np.nan, "MAE_cont": np.nan, "MAE_int": np.nan, "N": 0})
            continue

        errs_05, errs_cont, errs_int = [], [], []
        for _, s in sub.iterrows():
            gsd_i = float(s["GSD"])
            qqi_i = float(s["QQI"])
            rr_real = float(s["RR"])

            rr05   = estimate_rr_halfstep(gsd_i, qqi_i, rules_df, max_rr=max_rr_halfstep)
            rrcont = estimate_rr_continuous(gsd_i, qqi_i, rules_df)
            rrint  = classify_rr(gsd_i, qqi_i, rules_df)

            if rr05   is not None: errs_05.append(abs(rr05 - rr_real))
            if rrcont is not None: errs_cont.append(abs(rrcont - rr_real))
            if rrint  is not None: errs_int.append(abs(rrint - rr_real))

        out_rows.append({
            "Faixa_GSD": label,
            "MAE_05": float(np.mean(errs_05)) if errs_05 else np.nan,
            "MAE_cont": float(np.mean(errs_cont)) if errs_cont else np.nan,
            "MAE_int": float(np.mean(errs_int)) if errs_int else np.nan,
            "N": int(len(sub))
        })
    return pd.DataFrame(out_rows)

# ============= Cálculo QQI ponto a ponto =============
def process_point(point, df, neigh):
    idxs_neigh = neigh.radius_neighbors([point], return_distance=False)[0]
    if len(idxs_neigh) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(idxs_neigh)

    df_neighbors = df.iloc[idxs_neigh]
    if np.all(df_neighbors.var().values < 1e-8):
        return np.nan, np.nan, np.nan, np.nan, len(idxs_neigh)

    pca = PCA(n_components=3)
    pca.fit(df_neighbors.values)
    s = pca.singular_values_

    rug_1 = s[2] / (s[0] * s[1]) if (s[0] != 0 and s[1] != 0) else np.nan

    n1 = np.array([0, 0, 1])
    n2 = pca.components_[2]
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return rug_1, np.nan, np.nan, theta, len(idxs_neigh)

def read_cloud(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, header=None)
    elif ext == ".csv":
        try:
            return pd.read_csv(path, header=None)
        except Exception:
            return pd.read_csv(path, header=None, sep=';')
    elif ext == ".txt":
        return pd.read_csv(path, sep=r'\s+', header=None, engine='python', skiprows=1)
    else:
        raise ValueError("Formato não suportado. Use .xlsx, .csv ou .txt")

class QQICalculator:
    @staticmethod
    def compute_metrics(df_xyz: pd.DataFrame, radius=RADIUS):
        if df_xyz.shape[1] >= 3:
            df_xyz = df_xyz.iloc[:, :3]
            df_xyz.columns = ['x', 'y', 'z']
            df_xyz = df_xyz.apply(pd.to_numeric, errors='coerce').dropna()
        else:
            raise ValueError("Arquivo com menos de 3 colunas válidas (X,Y,Z).")

        neigh = NearestNeighbors(radius=radius)
        neigh.fit(df_xyz.values)

        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_point)(pt, df_xyz, neigh) for pt in df_xyz.values
        )

        rugs_1, _, _, thetas, neighbors_counts = zip(*results)
        rugs_1 = np.array(rugs_1, dtype=float)
        thetas = np.array(thetas, dtype=float)
        knn_cnt = np.array(neighbors_counts, dtype=float)

        PCA_index = rugs_1 * thetas
        media_KNN = np.nanmean(knn_cnt)

        qqi_points = PCA_index / media_KNN if (media_KNN and np.isfinite(media_KNN)) else np.nan
        df_xyz["QQI"] = qqi_points

        global_qqi = float(np.nanmean(qqi_points)) if np.ndim(qqi_points) else float(qqi_points)
        return global_qqi, df_xyz

    @staticmethod
    def generate_plot(df_plot: pd.DataFrame, output_path: str, value_col="RR_est", label="RR Estimado"):
        df_valid = df_plot.dropna(subset=[value_col])

        if value_col == "RR_est":
            vmin, vmax = 2.0, 5.0
            cmap = "jet"
        else:
            vmin, vmax = np.nanpercentile(df_valid[value_col], [5, 95])
            cmap = "viridis"

        fig, ax = plt.subplots(figsize=(6, 10))
        sc = ax.scatter(df_valid["x"], df_valid["y"], c=df_valid[value_col],
                        cmap=cmap, s=25, edgecolor="none",
                        vmin=vmin, vmax=vmax)
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label(label)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(f"Mapa Espacial - {label}")
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
        else:
            plt.show()

# ============= Interface (Tkinter) =============
class QQIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Estimador de RR a partir de QQI e GSD")
        self.geometry("820x520")
        self.resizable(False, False)

        self.cloud_path = tk.StringVar()
        self.gsd_value  = tk.StringVar()
        self.last_map_path = None
        self.queue = queue.Queue()

        frm = ttk.Frame(self, padding=16); frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Arquivo da nuvem (xlsx/csv/txt):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.cloud_path, width=62).grid(row=1, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Button(frm, text="Inserir Nuvem de Pontos", command=self.pick_file).grid(row=1, column=2, padx=8)

        ttk.Label(frm, text="GSD do voo (cm/pixel):").grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frm, textvariable=self.gsd_value, width=20).grid(row=3, column=0, sticky="w", pady=4)

        self.btn_calc = ttk.Button(frm, text="Calcular QQI & RR", command=self.start_pipeline)
        self.btn_calc.grid(row=3, column=2, padx=8)

        self.btn_open_map = ttk.Button(frm, text="Abrir Mapa RR", command=self.open_map, state="disabled")
        self.btn_open_map.grid(row=3, column=1, sticky="w", padx=8)

        ttk.Separator(frm, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="we", pady=12)

        self.lbl_qqi_raw    = ttk.Label(frm, text="QQI (sem escala): —")
        self.lbl_qqi_scaled = ttk.Label(frm, text=f"QQI (x{QQI_ESCALA:.0f}, calibração): —")
        self.lbl_rr         = ttk.Label(frm, text="RR estimada (0,5): —    |    RR (inteira): —")
        self.lbl_qqi_raw.grid(row=5, column=0, columnspan=3, sticky="w", pady=3)
        self.lbl_qqi_scaled.grid(row=6, column=0, columnspan=3, sticky="w", pady=3)
        self.lbl_rr.grid(row=7, column=0, columnspan=3, sticky="w", pady=3)

        self.progress = ttk.Progressbar(frm, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky="we", pady=(10, 5))

        self.status = ttk.Label(frm, text="", foreground="#666")
        self.status.grid(row=9, column=0, columnspan=3, sticky="w")

        for i in range(3):
            frm.columnconfigure(i, weight=1)

        # ======= Carrega calibração do Excel e constrói regras robustas =======
        try:
            self.calib_df = load_calibration_excel(CALIB_XLSX_PATH)
        except Exception as e:
            messagebox.showerror("Erro de calibração", f"Falha ao carregar calibração:\n{e}")
            raise

        self.rules_df = compute_rules_from_calib(self.calib_df, step=0.25)
        self.df_mae   = calib_mae_by_bin(self.calib_df, self.rules_df,
                                         max_rr_halfstep=MAX_RR_HALFSTEP)

        # Relatórios (opcional)
        try:
            self.rules_df.to_csv("qqi_rr_rules_by_gsd_bin.csv", index=False, encoding="utf-8")
            self.df_mae.to_csv("qqi_rr_mae_by_gsd_bin.csv", index=False, encoding="utf-8")
        except Exception:
            pass

        self.check_queue()

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Selecione a nuvem de pontos",
            filetypes=[("Padrões aceitos", "*.xlsx *.xls *.csv *.txt"),
                       ("Excel", "*.xlsx *.xls"), ("CSV", "*.csv"), ("TXT", "*.txt")]
        )
        if path:
            self.cloud_path.set(path)

    def _mae_for_gsd(self, gsd: float) -> float | None:
        mae_show = None
        for _, row_bin in self.rules_df.iterrows():
            lo, hi = label_to_bounds(row_bin["Faixa_GSD"])
            if (gsd >= lo) and (gsd <= hi):
                sub_mae = self.df_mae[self.df_mae["Faixa_GSD"] == row_bin["Faixa_GSD"]]
                if len(sub_mae):
                    v = float(sub_mae["MAE_05"].iloc[0])
                    if np.isfinite(v):
                        mae_show = v
                break

        if mae_show is None:
            valid = self.df_mae.dropna(subset=["MAE_05"])
            if len(valid) > 0:
                mae_show = float((valid["MAE_05"] * valid["N"]).sum() / max(1, valid["N"].sum()))
        return mae_show

    def open_map(self):
        if not self.last_map_path or not os.path.exists(self.last_map_path):
            messagebox.showinfo("Mapa RR", "Gere o mapa primeiro (Calcular QQI & RR).")
            return
        try:
            if os.name == "nt":
                os.startfile(self.last_map_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.last_map_path])
            else:
                subprocess.Popen(["xdg-open", self.last_map_path])
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir o mapa:\n{e}")

    def start_pipeline(self):
        path = self.cloud_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Erro", "Selecione um arquivo válido de nuvem.")
            return
        try:
            gsd = float(self.gsd_value.get().strip().replace(",", "."))
        except Exception:
            messagebox.showerror("Erro", "Informe um GSD válido (número).")
            return

        # Aviso se o GSD estiver fora do intervalo calibrado
        if get_rule_for_gsd(gsd, self.rules_df) is None:
            messagebox.showwarning(
                "GSD fora da calibração",
                "O GSD informado está fora do intervalo calibrado.\n"
                "A estimativa pode ser inválida (RR pode sair NaN)."
            )

        self.btn_calc.config(state="disabled")
        self.status.config(text="Lendo nuvem e calculando QQI (pode demorar)...")
        self.progress.start(10)

        threading.Thread(target=self.run_calculation_thread, args=(path, gsd), daemon=True).start()

    def run_calculation_thread(self, path, gsd):
        try:
            df_cloud = read_cloud(path)

            qqi_raw, df_with_qqi = QQICalculator.compute_metrics(df_cloud, radius=RADIUS)
            qqi_cal = qqi_raw * QQI_ESCALA

            out_dir = os.path.join(os.path.dirname(path), RESULTS_DIRNAME)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]

            df_with_qqi["RR_est"] = compute_rr_vectorized(df_with_qqi["QQI"].values, gsd, self.rules_df)

            map_out = os.path.join(out_dir, f"{base}_RR_map.png")
            QQICalculator.generate_plot(df_with_qqi, map_out, value_col="RR_est", label="RR Estimado")

            rr_half = estimate_rr_halfstep(gsd, qqi_cal, self.rules_df, max_rr=MAX_RR_HALFSTEP)
            rr_int  = classify_rr(gsd, qqi_cal, self.rules_df)
            mae_show = self._mae_for_gsd(gsd)

            out_csv = os.path.join(out_dir, f"{base}_qqi_rr.csv")
            pd.DataFrame([{
                "arquivo": base, "GSD": gsd,
                "QQI_sem_escala": qqi_raw, f"QQI_x{int(QQI_ESCALA)}": qqi_cal,
                "RR_estimada_0_5": rr_half, "RR_estimada_inteira": rr_int,
                "RR_MAE_faixa": (mae_show if (mae_show is not None and np.isfinite(mae_show)) else np.nan),
                "mapa_RR_png": map_out
            }]).to_csv(out_csv, index=False, encoding="utf-8")

            out_pkl = os.path.join(out_dir, f"{base}_resultado.pkl")
            with open(out_pkl, "wb") as f:
                pickle.dump({
                    "qqi_raw": qqi_raw, "qqi_calibracao": qqi_cal, "gsd": gsd,
                    "rr_05": rr_half, "rr_int": rr_int,
                    "mae_faixa": (mae_show if (mae_show is not None and np.isfinite(mae_show)) else None),
                    "mapa_RR_png": map_out
                }, f)

            rules_path = os.path.join(out_dir, "qqi_rr_rules_by_gsd_bin.csv")
            if not os.path.exists(rules_path):
                self.rules_df.to_csv(rules_path, index=False, encoding="utf-8")

            mae_path = os.path.join(out_dir, "qqi_rr_mae_by_gsd_bin.csv")
            if not os.path.exists(mae_path):
                self.df_mae.to_csv(mae_path, index=False, encoding="utf-8")

            self.queue.put(("success", {
                "qqi_raw": qqi_raw,
                "qqi_cal": qqi_cal,
                "rr_half": rr_half,
                "rr_int": rr_int,
                "mae_show": mae_show,
                "out_csv": out_csv,
                "map_out": map_out
            }))

        except Exception as e:
            self.queue.put(("error", str(e)))

    def check_queue(self):
        try:
            msg_type, data = self.queue.get_nowait()
            self.progress.stop()
            self.btn_calc.config(state="normal")

            if msg_type == "success":
                self.update_ui_success(data)
            elif msg_type == "error":
                messagebox.showerror("Erro", data)
                self.status.config(text="Erro no processamento.")
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_queue)

    def update_ui_success(self, data):
        self.last_map_path = data["map_out"]
        self.btn_open_map.config(state="normal")

        self.lbl_qqi_raw.config(text=f"QQI (sem escala): {data['qqi_raw']:.6g}")
        self.lbl_qqi_scaled.config(text=f"QQI (x{QQI_ESCALA:.0f}, calibração): {data['qqi_cal']:.6g}")

        rr_half = data["rr_half"]
        rr_int  = data["rr_int"]
        mae_show = data["mae_show"]

        rr_half_str = f"{rr_half:.1f}" if rr_half is not None else "—"
        if rr_half is not None and mae_show is not None and np.isfinite(mae_show):
            rr_half_str += f" ± {mae_show:.2f}"

        rr_int_str = str(rr_int) if rr_int is not None else "—"

        self.lbl_rr.config(text=f"RR estimada (0,5): {rr_half_str}    |    RR (inteira): {rr_int_str}")
        self.status.config(text=f"Concluído. CSV salvo em: {data['out_csv']}")

# ============= Run =============
if __name__ == "__main__":
    app = QQIApp()
    app.mainloop()
