# -*- coding: utf-8 -*-
r"""
qqi_rr_gui_limiares.py — GUI Tkinter (SEM classificador)

- QQI calculado como (Rugosidade * Theta) / REF_KNN
- RR estimado por limiares (t23, t34, t45) por GSD_bin
- Calibrador vem de um Excel com abas: LIMIARES e GLOBAL

RESULTADO FINAL (para não confundir usuário):
- Exibe APENAS UM RR FINAL (decimal): média truncada (5–95%) do RR_est (mapa).
  Isso produz valores como RR=2.68 e é robusto a outliers.

Observação:
- Ajuste QQI_ESCALA para bater com a escala do seu Excel.
  Você disse que o QQI da base já está multiplicado por 10^5, então:
  QQI_ESCALA = 1e5
"""

import os
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

# =========================
# CONFIGURAÇÕES
# =========================
DEFAULT_RADIUS = 0.5
N_JOBS = -1

QQI_ESCALA = 1e5  # sua base já está em 10^5
RESULTS_DIRNAME = "Resultados_QQI_RR"

CALIB_XLSX_PATH = r"D:\Users\Gleicon\Documents\DOUTORADO - UFRGS\TESE\TESTE\Calibrador_LIMIARES.xlsx"


# =========================
# Leitura do Calibrador
# =========================
def load_threshold_calibrator(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibrador não encontrado: {path}")

    df_lim = pd.read_excel(path, sheet_name="LIMIARES")
    df_glb = pd.read_excel(path, sheet_name="GLOBAL")

    # limpa
    df_lim.columns = [str(c).strip() for c in df_lim.columns]
    for c in ["GSD_bin", "REF_KNN", "t23", "t34", "t45"]:
        if c in df_lim.columns:
            df_lim[c] = pd.to_numeric(df_lim[c], errors="coerce")

    ref_knn = float(pd.to_numeric(df_glb.loc[0, "REF_KNN"], errors="coerce"))
    if not np.isfinite(ref_knn) or ref_knn <= 0:
        ref_knn = 50.0

    t23g = float(pd.to_numeric(df_glb.loc[0, "t23"], errors="coerce"))
    t34g = float(pd.to_numeric(df_glb.loc[0, "t34"], errors="coerce"))
    t45g = float(pd.to_numeric(df_glb.loc[0, "t45"], errors="coerce"))
    global_thr = {"t23": t23g, "t34": t34g, "t45": t45g}

    df_lim = df_lim.dropna(subset=["GSD_bin", "t23", "t34", "t45"]).copy()
    df_lim = df_lim.sort_values("GSD_bin").reset_index(drop=True)
    if len(df_lim) == 0:
        raise ValueError("A aba LIMIARES está vazia ou inválida.")

    return df_lim, global_thr, ref_knn


def nearest_bin_thresholds(df_lim, gsd_val, global_thr):
    """
    Busca o GSD_bin mais próximo (em valor absoluto) e retorna t23,t34,t45.
    Se algo falhar, retorna GLOBAL.
    """
    try:
        idx = int(np.argmin(np.abs(df_lim["GSD_bin"].values - float(gsd_val))))
        row = df_lim.iloc[idx]
        return {
            "GSD_bin": float(row["GSD_bin"]),
            "t23": float(row["t23"]),
            "t34": float(row["t34"]),
            "t45": float(row["t45"]),
        }
    except Exception:
        return {"GSD_bin": None, **global_thr}


def rr_from_thresholds(qqi_scaled, thr):
    """
    qqi_scaled: QQI já na escala do calibrador (ex: 10^5)
    thr: dict com t23,t34,t45
    Retorna RR inteiro 2..5
    """
    t23, t34, t45 = thr["t23"], thr["t34"], thr["t45"]
    if not np.isfinite(qqi_scaled):
        return np.nan
    if qqi_scaled < t23:
        return 2
    if qqi_scaled < t34:
        return 3
    if qqi_scaled < t45:
        return 4
    return 5


# =========================
# Cálculo PCA Local
# =========================
def process_point(point, df, neigh):
    idxs = neigh.radius_neighbors([point], return_distance=False)[0]
    if len(idxs) < 3:
        return np.nan, np.nan, len(idxs)

    df_n = df.iloc[idxs]
    if np.all(df_n.var().values < 1e-8):
        return np.nan, np.nan, len(idxs)

    pca = PCA(n_components=3).fit(df_n.values)
    s = pca.singular_values_

    rug = s[2] / (s[0] * s[1]) if (s[0] * s[1]) != 0 else np.nan

    n_z = np.array([0, 0, 1.0])
    n_pca = pca.components_[2]
    denom = (np.linalg.norm(n_z) * np.linalg.norm(n_pca))
    if denom == 0:
        theta = np.nan
    else:
        cos_theta = np.dot(n_z, n_pca) / denom
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return rug, theta, len(idxs)


def read_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, header=None)
    if ext == ".csv":
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 3:
            df = pd.read_csv(path, header=None, sep=';')
        return df
    if ext == ".txt":
        return pd.read_csv(path, sep=r'\s+', header=None, engine='python', skiprows=1)
    raise ValueError("Formato não suportado. Use txt/csv/xlsx.")


class QQICalculator:
    @staticmethod
    def compute_metrics(df_xyz, radius, ref_knn):
        df_xyz = df_xyz.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna()
        df_xyz.columns = ['x', 'y', 'z']

        neigh = NearestNeighbors(radius=radius).fit(df_xyz.values)
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_point)(pt, df_xyz, neigh) for pt in df_xyz.values
        )

        rugs, thetas, knns = zip(*results)
        df_xyz["Rugosidade"] = np.array(rugs, dtype=float)
        df_xyz["Theta"] = np.array(thetas, dtype=float)
        df_xyz["KNN"] = np.array(knns, dtype=float)

        # QQI normalizado pela densidade de referência (não depende do KNN local)
        pca_index = df_xyz["Rugosidade"] * df_xyz["Theta"]  # escala "pequena"
        if ref_knn > 0:
            df_xyz["QQI"] = pca_index / ref_knn
        else:
            df_xyz["QQI"] = np.nan

        return df_xyz

    @staticmethod
    def generate_plot(df, out_path, col, label, vmin=None, vmax=None, cmap="viridis"):
        df_v = df.dropna(subset=[col])
        if len(df_v) == 0:
            return

        if vmin is None or vmax is None:
            vmin, vmax = np.nanpercentile(df_v[col], [2, 98])

        fig, ax = plt.subplots(figsize=(8, 10))
        sc = ax.scatter(
            df_v["x"], df_v["y"],
            c=df_v[col], cmap=cmap, s=20, edgecolor="none",
            vmin=vmin, vmax=vmax
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(label, fontsize=11)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


# =========================
# APP
# =========================
class QQIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QQI & RR — Resultado Único (média truncada)")
        self.geometry("880x720")
        self.resizable(True, True)

        self.cloud_path = tk.StringVar()
        self.gsd_val = tk.StringVar()
        self.rad_val = tk.StringVar(value=str(DEFAULT_RADIUS))
        self.queue = queue.Queue()

        self.df_lim = None
        self.global_thr = None
        self.ref_knn = 50.0
        self.last_rr_map = None

        frm = ttk.Frame(self, padding=20)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(2, weight=1)

        ttk.Label(frm, text="Arquivo da nuvem (txt/csv/xlsx):", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        ttk.Entry(frm, textvariable=self.cloud_path).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 10), ipady=3)
        ttk.Button(frm, text="Selecionar...", command=self.pick_file).grid(row=1, column=2, sticky="ew")

        ttk.Label(frm, text="GSD (cm/px):", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", pady=(15, 2))
        ttk.Label(frm, text="Raio Busca (m):", font=("Segoe UI", 9)).grid(row=2, column=1, sticky="w", pady=(15, 2))
        ttk.Label(frm, text="REF_KNN (do calibrador):", font=("Segoe UI", 9)).grid(row=2, column=2, sticky="w", pady=(15, 2))

        ttk.Entry(frm, textvariable=self.gsd_val).grid(row=3, column=0, sticky="ew", padx=(0, 5), ipady=3)
        ttk.Entry(frm, textvariable=self.rad_val).grid(row=3, column=1, sticky="ew", padx=(0, 5), ipady=3)

        self.lbl_ref = ttk.Label(frm, text="(carregando...)", font=("Segoe UI", 9, "bold"))
        self.lbl_ref.grid(row=3, column=2, sticky="w")

        self.btn_calc = ttk.Button(frm, text="CALCULAR", command=self.start)
        self.btn_calc.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(25, 10), ipady=5)

        self.btn_open = ttk.Button(frm, text="Abrir Mapa RR", command=self.open_map, state="disabled")
        self.btn_open.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5, ipady=2)

        ttk.Separator(frm).grid(row=7, column=0, columnspan=3, sticky="ew", pady=20)

        self.lbl_res = ttk.Label(frm, text="Inicializando...", font=("Segoe UI", 11), justify="center", anchor="center")
        self.lbl_res.grid(row=8, column=0, columnspan=3, sticky="we")

        self.pbar = ttk.Progressbar(frm, mode='indeterminate')
        self.pbar.grid(row=9, column=0, columnspan=3, sticky="ew", pady=15)

        self.load_calibrator()
        self.check_queue()

    def load_calibrator(self):
        try:
            self.df_lim, self.global_thr, self.ref_knn = load_threshold_calibrator(CALIB_XLSX_PATH)
            self.lbl_ref.config(text=f"{self.ref_knn:.2f}")
            self.lbl_res.config(text=f"Calibrador OK | REF_KNN={self.ref_knn:.2f} | bins={len(self.df_lim)}")
        except Exception as e:
            self.lbl_res.config(text="Erro ao carregar calibrador.")
            messagebox.showerror("Erro", f"Falha ao carregar calibrador:\n{e}")

    def pick_file(self):
        f = filedialog.askopenfilename()
        if f:
            self.cloud_path.set(f)

    def open_map(self):
        if self.last_rr_map and os.path.exists(self.last_rr_map):
            if os.name == 'nt':
                os.startfile(self.last_rr_map)
            else:
                subprocess.Popen(["xdg-open", self.last_rr_map])

    def start(self):
        if not self.cloud_path.get():
            return messagebox.showerror("Erro", "Selecione um arquivo.")
        if self.df_lim is None:
            return messagebox.showerror("Erro", "Calibrador não carregado.")
        try:
            gsd = float(self.gsd_val.get().replace(",", "."))
            rad = float(self.rad_val.get().replace(",", "."))
        except:
            return messagebox.showerror("Erro", "GSD e Raio devem ser numéricos.")

        self.btn_calc.config(state="disabled")
        self.lbl_res.config(text="Processando...")
        self.pbar.start(10)

        threading.Thread(target=self.run, args=(self.cloud_path.get(), gsd, rad), daemon=True).start()

    def run(self, path, gsd, rad):
        try:
            df = read_cloud(path)

            df_xyz = QQICalculator.compute_metrics(df, rad, self.ref_knn)

            # QQI na escala do calibrador
            df_xyz["QQI_scaled"] = df_xyz["QQI"] * QQI_ESCALA

            # thresholds do bin mais próximo
            thr = nearest_bin_thresholds(self.df_lim, gsd, self.global_thr)

            # RR ponto a ponto (para mapa)
            rr_est = np.full(len(df_xyz), np.nan, dtype=float)
            qqi_vals = df_xyz["QQI_scaled"].values
            valid = np.isfinite(qqi_vals)
            for i in np.where(valid)[0]:
                rr_est[i] = rr_from_thresholds(qqi_vals[i], thr)
            df_xyz["RR_est"] = rr_est

            # ==========================
            # RESULTADO ÚNICO (RR_FINAL) – MÉDIA TRUNCADA 5–95%
            # ==========================
            rr_vals = df_xyz["RR_est"].values
            rr_vals = rr_vals[np.isfinite(rr_vals)]

            if rr_vals.size == 0:
                rr_final = np.nan
            else:
                lo, hi = np.percentile(rr_vals, [5, 95])
                rr_trim = rr_vals[(rr_vals >= lo) & (rr_vals <= hi)]
                rr_final = float(np.mean(rr_trim)) if rr_trim.size else float(np.mean(rr_vals))

            # saída
            out_dir = os.path.join(os.path.dirname(path), RESULTS_DIRNAME)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]

            map_rr = os.path.join(out_dir, f"{base}_MAPA_RR.png")
            QQICalculator.generate_plot(df_xyz, map_rr, "RR_est", "RR Estimado", 2, 5, "jet")

            map_rug = os.path.join(out_dir, f"{base}_MAPA_RUGOSIDADE.png")
            QQICalculator.generate_plot(df_xyz, map_rug, "Rugosidade", "Rugosidade PCA (Bruta)", cmap="viridis")

            map_qqi = os.path.join(out_dir, f"{base}_MAPA_QQI.png")
            QQICalculator.generate_plot(df_xyz, map_qqi, "QQI_scaled", f"QQI (escala {QQI_ESCALA:.0e})", cmap="plasma")

            # resumo (CSV) — mantém métricas auditáveis
            resumo = pd.DataFrame([{
                "Arquivo": base,
                "GSD": gsd,
                "Raio": rad,
                "REF_KNN": self.ref_knn,
                "GSD_bin_usado": thr.get("GSD_bin", None),
                "t23": thr["t23"], "t34": thr["t34"], "t45": thr["t45"],
                "RR_FINAL_truncated_mean": rr_final,
                "N_total": len(df_xyz),
                "N_valid": int(np.isfinite(df_xyz["QQI_scaled"].values).sum()),
                "pct_area_RR_ge4": float(np.mean(rr_vals >= 4) * 100.0) if rr_vals.size else np.nan,
            }])
            resumo.to_csv(os.path.join(out_dir, f"{base}_resumo.csv"), index=False)

            self.queue.put(("ok", {
                "rr_final": rr_final,
                "map": map_rr,
                "gsd_bin": thr.get("GSD_bin", None),
            }))

        except Exception as e:
            self.queue.put(("err", str(e)))

    def check_queue(self):
        try:
            kind, data = self.queue.get_nowait()
            self.pbar.stop()
            self.btn_calc.config(state="normal")

            if kind == "ok":
                self.last_rr_map = data["map"]
                self.btn_open.config(state="normal")

                # EXIBIÇÃO SIMPLES: APENAS UM RR
                self.lbl_res.config(
                    text=f"RR_Estimado FINAL (média truncada): {data['rr_final']:.2f} | bin: {data['gsd_bin']}"
                )
                messagebox.showinfo(
                    "Resultado",
                    f"RR_Estimado da): {data['rr_final']:.2f}\n"
                    f"GSD_bin usado: {data['gsd_bin']}\n\n"
                    "Obs.: detalhes adicionais foram salvos no resumo.csv."
                )
            else:
                messagebox.showerror("Erro", data)
                self.lbl_res.config(text="Erro.")
        except queue.Empty:
            pass

        self.after(100, self.check_queue)


if __name__ == "__main__":
    app = QQIApp()
    app.mainloop()
