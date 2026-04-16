# -*- coding: utf-8 -*-
r"""
qqi_rr_gui_limiares.py — GUI Tkinter (SEM classificador)

- QQI calculado como (Rugosidade * Theta) / KNN_ref_efetivo
- RR estimado por limiares (t23, t34, t45) por GSD_bin
- Calibrador vem de um Excel com abas: LIMIARES e GLOBAL

REGRA DE NORMALIZAÇÃO:
- O modelo usa aproximação por vizinho mais próximo dentro da amplitude calibrada
  de GSD, assumindo continuidade do comportamento do índice entre os bins.
- Se o GSD informado estiver dentro dessa amplitude e o bin selecionado possuir
  KNN_ref válido, o QQI é normalizado pelo KNN_ref do bin correspondente.
- Se o GSD estiver fora dessa amplitude, ou se o bin selecionado não possuir
  KNN_ref válido, usa-se o REF_KNN global como fallback.

RESULTADO FINAL:
- Exibe APENAS UM RR FINAL baseado na MODA PONDERADA do RR_est (mapa).
  RR é uma variável ordinal discreta — a moda (valor mais frequente) é mais
  representativa do que a média, que é puxada por pontos de ruído na nuvem.
  O resultado decimal vem da média ponderada das classes dominantes (>= 20% da moda).

ESCALA:
- QQI_ESCALA = 1e5
- REF_KNN global = valor da aba GLOBAL, usado apenas como fallback
"""

import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue

import laspy
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

QQI_ESCALA = 1e5
RESULTS_DIRNAME = "Resultados_QQI_RR"

# O filtro de densidade acompanha o KNN_ref_efetivo para manter consistência
# geométrica entre bins com diferentes densidades de referência.
KNN_MIN_RATIO = 0.15

CALIB_XLSX_PATH = r"D:\Users\Gleicon\Documents\DOUTORADO - UFRGS\TESE\Calibrador_QQI_RR_P75_v4.xlsx"


# =========================
# Leitura do Calibrador
# =========================
def _normalize_excel_colname(name):
    text = str(name).strip().replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _rename_excel_columns(df, alias_map):
    rename_map = {}
    for col in df.columns:
        key = _normalize_excel_colname(col)
        if key in alias_map:
            rename_map[col] = alias_map[key]
    return df.rename(columns=rename_map)


def load_threshold_calibrator(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibrador não encontrado: {path}")

    df_lim = pd.read_excel(path, sheet_name="LIMIARES", header=1)
    df_glb = pd.read_excel(path, sheet_name="GLOBAL", header=1)

    # limpa colunas
    df_lim.columns = [str(c).strip() for c in df_lim.columns]
    df_glb.columns = [str(c).strip() for c in df_glb.columns]

    lim_alias_map = {
        "gsd_bin": "GSD_bin",
        "gsd_bin (cm/px)": "GSD_bin",
        "gsd_bin(cm/px)": "GSD_bin",
        "ref_knn": "KNN_ref",
        "ref_knn (mediana)": "KNN_ref",
        "ref_knn(mediana)": "KNN_ref",
        "knn_ref": "KNN_ref",
        "knn_ref (mediana)": "KNN_ref",
        "knn_ref(mediana)": "KNN_ref",
        "knn_bin": "KNN_ref",
        "t23": "t23",
        "t23 (rr2->3)": "t23",
        "t23 (rr2→3)": "t23",
        "t34": "t34",
        "t34 (rr3->4)": "t34",
        "t34 (rr3→4)": "t34",
        "t45": "t45",
        "t45 (rr4->5)": "t45",
        "t45 (rr4→5)": "t45",
    }
    glb_alias_map = {
        "ref_knn": "REF_KNN",
        "ref_knn (mediana)": "REF_KNN",
        "ref_knn(mediana)": "REF_KNN",
        "t23": "t23",
        "t34": "t34",
        "t45": "t45",
    }

    df_lim = _rename_excel_columns(df_lim, lim_alias_map)
    df_glb = _rename_excel_columns(df_glb, glb_alias_map)
    df_lim = df_lim.dropna(how="all").copy()
    df_glb = df_glb.dropna(how="all").reset_index(drop=True)

    missing_lim = [c for c in ["GSD_bin", "t23"] if c not in df_lim.columns]
    if missing_lim:
        raise ValueError(f"Colunas obrigatórias ausentes na aba LIMIARES: {missing_lim}")

    missing_glb = [c for c in ["t23", "t34", "t45"] if c not in df_glb.columns]
    if missing_glb:
        raise ValueError(f"Colunas obrigatórias ausentes na aba GLOBAL: {missing_glb}")

    # Lê todas as colunas numéricas relevantes, incluindo KNN_ref se existir
    for c in ["GSD_bin", "t23", "t34", "t45", "KNN_ref"]:
        if c in df_lim.columns:
            df_lim[c] = pd.to_numeric(df_lim[c], errors="coerce")

    for c in ["REF_KNN", "t23", "t34", "t45"]:
        if c in df_glb.columns:
            df_glb[c] = pd.to_numeric(df_glb[c], errors="coerce")

    if len(df_glb) == 0:
        raise ValueError("A aba GLOBAL está vazia ou inválida.")

    global_row = df_glb.iloc[0]

    # lê REF_KNN da aba GLOBAL
    ref_knn_global = float(pd.to_numeric(global_row.get("REF_KNN"), errors="coerce"))
    if not np.isfinite(ref_knn_global) or ref_knn_global <= 0:
        ref_knn_global = 68.70  # fallback: mediana da base de calibração

    t23g = float(pd.to_numeric(global_row.get("t23"), errors="coerce"))
    t34g = float(pd.to_numeric(global_row.get("t34"), errors="coerce"))
    t45g = float(pd.to_numeric(global_row.get("t45"), errors="coerce"))
    if not all(np.isfinite(v) for v in [t23g, t34g, t45g]):
        raise ValueError("Os limiares globais t23/t34/t45 estão ausentes ou inválidos na aba GLOBAL.")
    global_thr = {"t23": t23g, "t34": t34g, "t45": t45g}

    # Exige apenas GSD_bin e t23 válidos
    df_lim = df_lim.dropna(subset=["GSD_bin", "t23"]).copy()
    df_lim = df_lim.sort_values("GSD_bin").reset_index(drop=True)

    if len(df_lim) == 0:
        raise ValueError("A aba LIMIARES está vazia ou inválida.")

    return df_lim, global_thr, ref_knn_global


def nearest_bin_thresholds(df_lim, gsd_val, global_thr):
    """
    Resolve o bin de GSD mais próximo e retorna:
    - GSD_bin selecionado
    - limiares t23, t34, t45
    - KNN_ref do bin, se disponível
    - flag 'in_range', que indica apenas se o GSD informado está dentro da
      amplitude total calibrada [min(GSD_bin), max(GSD_bin)]

    Observação:
    - Dentro da amplitude calibrada de GSD, usa-se o bin mais próximo.
    - Fora dessa amplitude, o chamador deve usar o REF_KNN global como fallback.
    - t34 e t45 podem usar fallback global caso estejam ausentes no bin.
    - Não há validação explícita da distância entre o GSD informado e o bin mais
      próximo; assume-se que o calibrador possui densidade suficiente de bins.
    """
    try:
        gsd_val = float(gsd_val)
        gsd_bins = df_lim["GSD_bin"].values.astype(float)

        gsd_min = float(np.nanmin(gsd_bins))
        gsd_max = float(np.nanmax(gsd_bins))
        in_range = (gsd_val >= gsd_min) and (gsd_val <= gsd_max)

        idx = int(np.argmin(np.abs(gsd_bins - gsd_val)))
        row = df_lim.iloc[idx]

        t23 = float(row["t23"])
        t34 = float(row["t34"]) if pd.notna(row.get("t34")) else global_thr["t34"]
        t45 = float(row["t45"]) if pd.notna(row.get("t45")) else global_thr["t45"]

        # KNN_ref do bin — None se coluna ausente ou valor NaN
        knn_bin = None
        if "KNN_ref" in df_lim.columns and pd.notna(row.get("KNN_ref")):
            knn_bin = float(row["KNN_ref"])

        return {
            "GSD_bin": float(row["GSD_bin"]),
            "t23": t23,
            "t34": t34,
            "t45": t45,
            "KNN_bin": knn_bin,
            "in_range": in_range,
            "gsd_min": gsd_min,
            "gsd_max": gsd_max,
        }
    except Exception:
        return {
            "GSD_bin": None,
            "t23": global_thr["t23"],
            "t34": global_thr["t34"],
            "t45": global_thr["t45"],
            "KNN_bin": None,
            "in_range": False,
            "gsd_min": None,
            "gsd_max": None,
        }


def rr_from_thresholds(qqi_scaled, thr):
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
    denom = np.linalg.norm(n_z) * np.linalg.norm(n_pca)
    if denom == 0:
        theta = np.nan
    else:
        cos_theta = np.dot(n_z, n_pca) / denom
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return rug, theta, len(idxs)


def read_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".las":
        las = laspy.read(path)
        df = pd.DataFrame({"x": las.x, "y": las.y, "z": las.z})
        return df
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, header=None)
    if ext == ".csv":
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 3:
            df = pd.read_csv(path, header=None, sep=';')
        return df
    if ext == ".txt":
        return pd.read_csv(path, sep=r'\s+', header=None, engine='python', skiprows=1)
    raise ValueError("Formato não suportado. Use LAS, TXT ou CSV.")


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

        pca_index = df_xyz["Rugosidade"] * df_xyz["Theta"]
        df_xyz["QQI"] = pca_index / ref_knn if ref_knn > 0 else np.nan

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
        self.title("QQI & RR — Resultado Único")
        self.geometry("880x720")
        self.resizable(True, True)

        self.cloud_path = tk.StringVar()
        self.gsd_val = tk.StringVar()
        self.rad_val = tk.StringVar(value=str(DEFAULT_RADIUS))
        self.queue = queue.Queue()

        self.df_lim = None
        self.global_thr = None
        self.ref_knn_global = 68.70
        self.last_rr_map = None

        frm = ttk.Frame(self, padding=20)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(2, weight=1)

        ttk.Label(frm, text="Arquivo da nuvem (LAS/TXT/CSV):", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        ttk.Entry(frm, textvariable=self.cloud_path).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 10), ipady=3)
        ttk.Button(frm, text="Selecionar...", command=self.pick_file).grid(row=1, column=2, sticky="ew")

        ttk.Label(frm, text="GSD (cm/px):", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", pady=(15, 2))
        ttk.Label(frm, text="Raio Busca (m):", font=("Segoe UI", 9)).grid(row=2, column=1, sticky="w", pady=(15, 2))
        ttk.Label(frm, text="REF_KNN global (fallback):", font=("Segoe UI", 9)).grid(row=2, column=2, sticky="w", pady=(15, 2))

        ttk.Entry(frm, textvariable=self.gsd_val).grid(row=3, column=0, sticky="ew", padx=(0, 5), ipady=3)
        ttk.Entry(frm, textvariable=self.rad_val).grid(row=3, column=1, sticky="ew", padx=(0, 5), ipady=3)

        self.lbl_ref = ttk.Label(frm, text="(carregando...)", font=("Segoe UI", 9, "bold"))
        self.lbl_ref.grid(row=3, column=2, sticky="w")

        # Mostra o KNN efetivo realmente usado no cálculo do QQI.
        ttk.Label(frm, text="KNN efetivo usado no QQI:", font=("Segoe UI", 9)).grid(
            row=4, column=0, sticky="w", pady=(10, 2)
        )
        self.lbl_knn_bin = ttk.Label(frm, text="—", font=("Segoe UI", 9, "bold"))
        self.lbl_knn_bin.grid(row=4, column=1, sticky="w", pady=(10, 2))
        # ────────────────────────────────────────────────────────────────────

        self.btn_calc = ttk.Button(frm, text="CALCULAR", command=self.start)
        self.btn_calc.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(20, 10), ipady=5)

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
            self.df_lim, self.global_thr, self.ref_knn_global = load_threshold_calibrator(CALIB_XLSX_PATH)
            self.lbl_ref.config(text=f"{self.ref_knn_global:.2f}")
            self.lbl_res.config(
                text=f"Calibrador OK | REF_KNN global (fallback)={self.ref_knn_global:.2f} | bins={len(self.df_lim)}"
            )
        except Exception as e:
            self.lbl_res.config(text="Erro ao carregar calibrador.")
            messagebox.showerror("Erro", f"Falha ao carregar calibrador:\n{e}")

    def pick_file(self):
        f = filedialog.askopenfilename(
            filetypes=[
                ("Nuvens de pontos", "*.las *.txt *.csv"),
                ("LAS", "*.las"),
                ("TXT", "*.txt"),
                ("CSV", "*.csv"),
                ("Todos os arquivos", "*.*"),
            ]
        )
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
        except Exception:
            return messagebox.showerror("Erro", "GSD e Raio devem ser numéricos.")

        self.btn_calc.config(state="disabled")
        self.lbl_knn_bin.config(text="calculando...")
        self.lbl_res.config(text="Processando...")
        self.pbar.start(10)

        threading.Thread(target=self.run, args=(self.cloud_path.get(), gsd, rad), daemon=True).start()

    def run(self, path, gsd, rad):
        try:
            df = read_cloud(path)

            thr = nearest_bin_thresholds(self.df_lim, gsd, self.global_thr)

            # in_range indica apenas que o GSD está dentro da amplitude calibrada
            # [min_bin, max_bin]; a seleção do bin segue vizinho mais próximo.
            if thr["in_range"] and (thr.get("KNN_bin") is not None) and np.isfinite(thr["KNN_bin"]):
                knn_eff = float(thr["KNN_bin"])
                knn_source = "bin_gsd"
            else:
                knn_eff = float(self.ref_knn_global)
                knn_source = "fallback_global"

            df_xyz = QQICalculator.compute_metrics(df, rad, knn_eff)

            df_xyz["QQI_scaled"] = df_xyz["QQI"] * QQI_ESCALA

            # O limiar mínimo de vizinhos acompanha o KNN efetivo do bin/fallback.
            knn_min = max(3, int(KNN_MIN_RATIO * knn_eff))
            df_xyz["KNN_ok"] = df_xyz["KNN"] >= knn_min

            rr_est = np.full(len(df_xyz), np.nan, dtype=float)
            qqi_vals = df_xyz["QQI_scaled"].values
            valid = np.isfinite(qqi_vals) & df_xyz["KNN_ok"].values
            for i in np.where(valid)[0]:
                rr_est[i] = rr_from_thresholds(qqi_vals[i], thr)
            df_xyz["RR_est"] = rr_est

            rr_vals = df_xyz["RR_est"].values
            rr_vals = rr_vals[np.isfinite(rr_vals)]

            if rr_vals.size == 0:
                rr_final = np.nan
            else:
                classes, counts = np.unique(rr_vals, return_counts=True)
                freq = counts / counts.sum()
                moda_freq = freq.max()
                mask_dom = freq >= 0.20 * moda_freq
                rr_final = float(np.average(classes[mask_dom], weights=freq[mask_dom]))

            out_dir = os.path.join(os.path.dirname(path), RESULTS_DIRNAME)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]

            map_rr = os.path.join(out_dir, f"{base}_MAPA_RR.png")
            QQICalculator.generate_plot(df_xyz, map_rr, "RR_est", "RR Estimado", 2, 5, "jet")

            map_rug = os.path.join(out_dir, f"{base}_MAPA_RUGOSIDADE.png")
            QQICalculator.generate_plot(df_xyz, map_rug, "Rugosidade", "Rugosidade PCA (Bruta)", cmap="viridis")

            map_qqi = os.path.join(out_dir, f"{base}_MAPA_QQI.png")
            QQICalculator.generate_plot(df_xyz, map_qqi, "QQI_scaled", f"QQI (escala {QQI_ESCALA:.0e})", cmap="plasma")

            knn_bin = thr.get("KNN_bin")
            knn_bin_str = f"{knn_bin:.1f}" if knn_bin is not None else "N/D"
            knn_eff_str = f"{knn_eff:.1f}"

            resumo = pd.DataFrame([{
                "Arquivo": base,
                "GSD_informado": gsd,
                "Raio_busca_m": rad,
                "REF_KNN_global_fallback": self.ref_knn_global,
                "KNN_bin_calibrador": knn_bin_str,
                "KNN_efetivo_usado": knn_eff_str,
                "Origem_KNN": knn_source,
                "KNN_min_usado": knn_min,
                "GSD_bin_usado": thr.get("GSD_bin", None),
                "GSD_dentro_amplitude_calibrada": thr.get("in_range", False),
                "Amplitude_GSD_min": thr.get("gsd_min", None),
                "Amplitude_GSD_max": thr.get("gsd_max", None),
                "t23": thr["t23"],
                "t34": thr["t34"],
                "t45": thr["t45"],
                "RR_FINAL_moda_ponderada": rr_final,
                "N_total": len(df_xyz),
                "N_valid": int(valid.sum()),
                "N_descartado_baixo_KNN": int((~df_xyz["KNN_ok"]).sum()),
                "pct_area_RR_ge4": float(np.mean(rr_vals >= 4) * 100.0) if rr_vals.size else np.nan,
            }])
            resumo.to_csv(os.path.join(out_dir, f"{base}_resumo.csv"), index=False)

            self.queue.put(("ok", {
                "rr_final": rr_final,
                "map": map_rr,
                "gsd_bin": thr.get("GSD_bin", None),
                "knn_bin": knn_bin,
                "knn_bin_str": knn_bin_str,
                "knn_eff": knn_eff,
                "knn_eff_str": knn_eff_str,
                "knn_source": knn_source,
                "in_range": thr.get("in_range", False),
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

                origem = "bin GSD" if data["knn_source"] == "bin_gsd" else "fallback global"
                self.lbl_knn_bin.config(text=f"{data['knn_eff_str']} ({origem})")

                self.lbl_res.config(
                    text=(
                        f"RR_Final: {data['rr_final']:.2f} | "
                        f"GSD_bin: {data['gsd_bin']} | "
                        f"KNN_efetivo: {data['knn_eff_str']} ({origem})"
                    )
                )
                messagebox.showinfo(
                    "Resultado",
                    f"RR estimado (moda ponderada): {data['rr_final']:.2f}\n"
                    f"GSD_bin usado: {data['gsd_bin']}\n"
                    f"KNN efetivo usado no QQI: {data['knn_eff_str']} ({origem})\n\n"
                    "Os detalhes completos foram salvos no arquivo resumo.csv."
                )
            else:
                self.lbl_knn_bin.config(text="—")
                messagebox.showerror("Erro", data)
                self.lbl_res.config(text="Erro.")
        except queue.Empty:
            pass

        self.after(100, self.check_queue)


if __name__ == "__main__":
    app = QQIApp()
    app.mainloop()
