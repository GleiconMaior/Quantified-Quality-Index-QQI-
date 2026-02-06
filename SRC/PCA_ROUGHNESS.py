import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import pickle
from datetime import datetime

# Função para escalar os valores para a escala de 1 a 10


def scale_values(values):
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    return [(value - min_value) / (max_value - min_value) * 10 for value in values]

# Função para processar cada ponto


def process_point(point, df, neigh, pca):
    idxs_neigh = neigh.radius_neighbors([point], return_distance=False)[0]
    if len(idxs_neigh) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(idxs_neigh)

    df_neighbors = df.iloc[idxs_neigh]

    # Verificar se todos os pontos são iguais (variância zero)
    if np.all(df_neighbors.var().values < 1e-8):
        return np.nan, np.nan, np.nan, np.nan, len(idxs_neigh)

    pca.fit(df_neighbors.values)

    s = pca.singular_values_
    rug_1 = s[2] / (s[0] * s[1]) if s[0] != 0 and s[1] != 0 else np.nan
    rug_2 = s[2] / s[0] if s[0] != 0 else np.nan
    rug_3 = s[2] / max(s[0], s[1]) if max(s[0], s[1]) != 0 else np.nan

    n1 = np.array([0, 0, 1])
    n2 = pca.components_[2]
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return rug_1, rug_2, rug_3, theta, len(idxs_neigh)


# Função para salvar gráficos
def salvar_graficos(df_red, rug_scaled, titulo, nome_base, output_dir):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_red['x'], df_red['y'], c=rug_scaled,
                     cmap='viridis', s=10, vmin=0, vmax=4)
    cbar = plt.colorbar(sc)
    cbar.set_label("Índice PCA (PC3/(PC1*PC2))")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(titulo)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{nome_base}_plot.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(rug_scaled, bins=20, color='gray')
    plt.title(f"Histograma - {nome_base}")
    plt.xlabel("Rugosidade Escalada")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{nome_base}_hist.png"))
    plt.close()

# Função principal de análise


def analyze_file(filepath, output_dir, log_file, radius=0.5):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    try:
        # Leitura do arquivo
        df = pd.read_csv(filepath, sep=r'\s+', header=None,
                         engine='python', skiprows=1)

# Forçar uso somente das 3 primeiras colunas (X, Y, Z)
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
            df.columns = ['x', 'y', 'z']
            # Converte para numérico e remove linhas inválidas
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
        else:
            raise ValueError("Arquivo com menos de 3 colunas válidas.")

        df_red = df.sample(frac=1)

        neigh = NearestNeighbors(radius=radius)
        neigh.fit(df.values)

        pca = PCA(n_components=3)
        results = Parallel(n_jobs=-1)(
            delayed(process_point)(point, df, neigh, pca) for point in df_red.values
        )

        rugs_1, rugs_2, rugs_3, thetas, neighbors_counts = zip(*results)

        rugs_1 = np.array(rugs_1)
        rugs_2 = np.array(rugs_2)
        rugs_3 = np.array(rugs_3)
        thetas = np.array(thetas)
        neighbors_counts = np.array(neighbors_counts)

        # Índice PCA_index = rug_1 * theta
        PCA_index = rugs_1 * thetas
        # QQI = PCA_index / media_KNN
        media_KNN = np.nanmean(neighbors_counts)
        QQI = PCA_index / media_KNN if media_KNN != 0 else np.nan

        rugs_1_scaled = scale_values(rugs_1[~np.isnan(rugs_1)])
        df_red_valid = df_red.iloc[~np.isnan(rugs_1)]

        salvar_graficos(df_red_valid, rugs_1_scaled,
                        f"{base_name} - Rugosidade 1", base_name, output_dir)

        with open(os.path.join(output_dir, f'{base_name}_resultados.pickle'), 'wb') as f:
            pickle.dump((rugs_1_scaled, rugs_2, rugs_3, thetas,
                        neighbors_counts, PCA_index, QQI), f)

        resumo = {
            'arquivo': base_name,
            'media_rugosidade_1': np.nanmean(rugs_1),
            'media_rugosidade_1_esc': np.nanmean(rugs_1_scaled),
            'var_rugosidade_1': np.nanvar(rugs_1),
            'var_rugosidade_1_esc': np.nanvar(rugs_1_scaled),
            'media_KNN': np.nanmean(neighbors_counts),
            'mediana_KNN': np.nanmedian(neighbors_counts),
            'min_KNN': np.nanmin(neighbors_counts),
            'max_KNN': np.nanmax(neighbors_counts),
            'media_PCA_index': np.nanmean(PCA_index),
            'var_PCA_index': np.nanvar(PCA_index),
            'media_QQI': np.nanmean(QQI),
            'var_QQI': np.nanvar(QQI)
        }

        log_file.write(f"[{datetime.now()}] ✅ Sucesso: {base_name}\n")
        print(f"✅ {base_name} analisado com sucesso.")
        return resumo

    except Exception as e:
        log_file.write(f"[{datetime.now()}] ❌ Erro em {base_name}: {str(e)}\n")
        print(f"❌ Erro ao processar {base_name}: {str(e)}")
        return None


# EXECUÇÃO
if __name__ == "__main__":
    input_dir = r"D:\Users\Gleicon\Documents\DOUTORADO - UFRGS\TESE\TRECHOS LEVANTAMENTOS"
    output_dir = os.path.join(input_dir, "Resultados_05.02.26")
    os.makedirs(output_dir, exist_ok=True)

    arquivos = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if f.lower().endswith(".txt")]

    resumos = []
    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== INÍCIO DA EXECUÇÃO: {datetime.now()} ===\n")
        for i, arquivo in enumerate(arquivos, 1):
            print(
                f"🔎 ({i}/{len(arquivos)}) Analisando {os.path.basename(arquivo)}...")
            resumo = analyze_file(arquivo, output_dir, log_file)
            if resumo:
                resumos.append(resumo)
        log_file.write(f"=== FIM DA EXECUÇÃO: {datetime.now()} ===\n")

    df_resumo = pd.DataFrame(resumos)
    df_resumo.to_csv(os.path.join(output_dir, "resumo_geral.csv"), index=False)
    print("\n✅ Análise concluída. Resultados e log salvos em:", output_dir)
