"""
Seleção de Atributos com Algoritmos Genéticos e Redes Neurais
CEFET-RJ - Prof. Laercio Brito
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Para rodar sem display
import copy
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURAÇÃO DE DISPOSITIVO (CPU / GPU)
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Apple MPS (Metal) detectado")
    else:
        device = torch.device("cpu")
        print("✓ Usando CPU")
    return device


DEVICE = get_device()

# ─────────────────────────────────────────────
#  PARÂMETROS GLOBAIS
# ─────────────────────────────────────────────
PARAMS = {
    # Algoritmo Genético
    "pop_size": 30,
    "pc": 0.85,  # prob. crossover
    "elitism": 10,  # elitismo: top-10 preservados
    "gap": 2,  # Steady-State gap
    "max_gen": 5,  # máx. gerações
    "no_improve": 20,  # parada antecipada
    "n_experiments": 3,  # experimentos para curva média
    # Rede Neural
    "hidden1": 32,
    "hidden2": 16,
    "lr": 0.001,
    "epochs": 10,  # máx. épocas por treino
    "batch_size": 32,
    "patience": 10,  # early stopping na RNA
    # Dados
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    # Fitness
    "w_f1": 0.9,
    "w_feat": 0.1,
}


# ─────────────────────────────────────────────
#  1. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
def preprocess(filepath: str):
    print("\n══════════════════════════════════════")
    print("  PRÉ-PROCESSAMENTO")
    print("══════════════════════════════════════")

    df = pd.read_csv(filepath, sep=";")
    print(f"Shape original : {df.shape}")

    # ── Separar o alvo ANTES de qualquer conversão numérica ──
    # (Correção: se a coluna alvo for texto/categórica, forçar pd.to_numeric
    # no DataFrame inteiro a transformaria toda em NaN -> 0, colapsando as classes.)
    target_col = df.columns[-1]
    print(f"Coluna alvo: '{target_col}'")

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    # --------------------------------------------------
    # Conversão de colunas categóricas binárias S/N → 0/1
    # --------------------------------------------------
    #
    binary_sn_cols = [
        "TPPOS",
        "STCODIFICA",
        "CODIFICADO",
        "res_AMAZONIA",
        "res_FRONTEIRA",
        "res_CAPITAL",
        "ocor_AMAZONIA",
        "ocor_FRONTEIRA",
        "ocor_CAPITAL",
    ]
    for col in binary_sn_cols:
        X_df[col] = X_df[col].str.strip().str.upper().map({"S": 1, "N": 0})

    # --------------------------------------------------
    # Conversão de colunas categóricas nominais → Label Encoding
    # --------------------------------------------------

    label_enc_cols = [
        "arquivo_UF",
        "CAUSABAS",
        "CB_PRE",
        "CAUSABAS_O",
        "causabas_categoria",
        "causabas_subcategoria",
        "VERSAOSIST",
        "FONTES",
        "TPNIVELINV",
        "res_MUNNOMEX",
        "ocor_MUNNOMEX",
        "ocor_REGIAO",
        "res_REGIAO",
    ]
    le = LabelEncoder()
    for col in label_enc_cols:
        X_df[col] = X_df[col].astype(str)
        X_df[col] = le.fit_transform(X_df[col])

    # ── Se os números usarem vírgula decimal (formato BR), normaliza para ponto ──
    X_df = X_df.replace(",", ".", regex=True)

    # ── Remover linhas com NULL ou vazio apenas nos atributos ──
    X_df = X_df.replace("", np.nan)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    null_count = X_df.isnull().sum().sum()
    X_df.fillna(0, inplace=True)
    print(f"Valores NULL/vazio (atributos) substituídos por 0: {null_count}")

    X = X_df.values.astype(np.float64)
    y = LabelEncoder().fit_transform(y_raw.values)

    # ── Análise exploratória ──
    print(f"\nShape após limpeza : {X_df.shape}")
    print(f"Atributos          : {X.shape[1]}")
    print(f"Classes            : {np.unique(y)}")
    class_counts = pd.Series(y).value_counts().sort_index()
    print("Distribuição de classes:")
    for cls, cnt in class_counts.items():
        print(f"  Classe {cls}: {cnt} amostras ({100 * cnt / len(y):.1f}%)")

    print(
        "\nNota: a normalização Min-Max é feita DEPOIS do split (em split_data), "
        "usando apenas o conjunto de treino, para evitar vazamento de dados (data leakage)."
    )

    # ─────────────────────────────────────────────
    # SALVAR DATASET TRATADO (sem normalização, que agora é por split)
    # ─────────────────────────────────────────────
    colunas = X_df.columns.tolist()

    df_tratado = pd.DataFrame(X, columns=colunas)
    df_tratado[target_col] = y

    arquivo_saida = "dados_tratados.csv"
    df_tratado.to_csv(arquivo_saida, index=False)

    print(f"\n✓ Dataset tratado salvo em: {arquivo_saida}")
    print(f"✓ Shape do dataset tratado: {df_tratado.shape}")

    return X, y


# ─────────────────────────────────────────────
#  2. DIVISÃO DE DADOS + NORMALIZAÇÃO (sem leakage)
# ─────────────────────────────────────────────
def split_data(X, y, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X,
        y,
        test_size=(PARAMS["val_ratio"] + PARAMS["test_ratio"]),
        random_state=seed,
        stratify=y,
    )
    val_frac = PARAMS["val_ratio"] / (PARAMS["val_ratio"] + PARAMS["test_ratio"])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_frac), random_state=seed, stratify=y_tmp
    )

    # ── Normalização Min-Max calculada SOMENTE no treino, aplicada nos 3 conjuntos ──
    # (Correção: antes era feita no dataset inteiro, antes do split, o que causa
    # vazamento de dados — val/teste influenciavam o min/max do treino.)
    xmin = X_tr.min(axis=0)
    xmax = X_tr.max(axis=0)
    denom = xmax - xmin
    denom[denom == 0] = 1

    X_tr = (X_tr - xmin) / denom
    X_val = (X_val - xmin) / denom
    X_te = (X_te - xmin) / denom

    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─────────────────────────────────────────────
#  3. REDE NEURAL (PyTorch)
# ─────────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, PARAMS["hidden1"]),
            nn.ReLU(),
            nn.Linear(PARAMS["hidden1"], PARAMS["hidden2"]),
            nn.ReLU(),
            nn.Linear(PARAMS["hidden2"], n_classes),
            # Softmax embutido no CrossEntropyLoss do PyTorch
        )

    def forward(self, x):
        return self.net(x)


def train_eval_nn(X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, device):
    """
    Treina a RNA com Adam + Backpropagation.
    Retorna o F1-Score no conjunto de teste usando os pesos de melhor validação.
    """
    n_inputs = X_tr.shape[1]
    model = NeuralNet(n_inputs, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["lr"])
    criterion = nn.CrossEntropyLoss()

    # Tensores
    def to_tensor(X, y):
        return (
            torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.long).to(device),
        )

    Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
    Xval_t, yval_t = to_tensor(X_val, y_val)
    Xte_t, yte_t = to_tensor(X_te, y_te)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=PARAMS["batch_size"], shuffle=True
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_imp = 0

    for epoch in range(PARAMS["epochs"]):
        if epoch == 0:
            print(
                f"      Treinando RNA ({n_inputs} atributos, {PARAMS['epochs']} épocas)"
            )

        model.train()

        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_loss = criterion(model(Xval_t), yval_t).item()

        # salva melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 5 == 0:
            print(
                f"         Época {epoch + 1:2d}/"
                f"{PARAMS['epochs']} "
                f"| val_loss={val_loss:.5f}"
            )

        # early stopping
        if no_imp >= PARAMS["patience"]:
            break

    # Melhor modelo → teste
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).argmax(dim=1).cpu().numpy()

    f1 = f1_score(y_te, preds, average="weighted", zero_division=0)
    return f1


# ─────────────────────────────────────────────
#  4. FUNÇÃO DE APTIDÃO
# ─────────────────────────────────────────────
def fitness_fn(chromosome, X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, device):
    """Avalia um cromossomo. Retorna fitness ∈ [0,1]."""
    selected = np.where(chromosome == 1)[0]
    Nt = len(chromosome)
    Ns = len(selected)

    # Cromossomo vazio → fitness 0
    if Ns == 0:
        return 0.0

    f1 = train_eval_nn(
        X_tr[:, selected],
        y_tr,
        X_val[:, selected],
        y_val,
        X_te[:, selected],
        y_te,
        n_classes,
        device,
    )
    fitness = PARAMS["w_f1"] * f1 + PARAMS["w_feat"] * (1 - Ns / Nt)
    return fitness


def normalize_fitness(pop_fitness: np.ndarray) -> np.ndarray:
    """Normalização linear dos valores de fitness."""
    f_min = pop_fitness.min()
    f_max = pop_fitness.max()
    if f_max == f_min:
        return np.ones_like(pop_fitness)
    return (pop_fitness - f_min) / (f_max - f_min)


# ─────────────────────────────────────────────
#  5. OPERADORES GENÉTICOS
# ─────────────────────────────────────────────
def crossover_uniform(p1, p2):
    """Crossover Uniforme: cada gene é herdado de um dos pais com prob. 0.5."""
    mask = np.random.randint(0, 2, size=len(p1))
    child1 = np.where(mask, p1, p2)
    child2 = np.where(mask, p2, p1)
    return child1, child2


def mutate(chromosome):
    """Mutação bit-flip com Pm = 1/L."""
    L = len(chromosome)
    pm = 1.0 / L
    child = chromosome.copy()
    for i in range(L):
        if np.random.rand() < pm:
            child[i] = 1 - child[i]
    return child


def roulette_select(population, norm_fitness):
    """Seleção por roleta (proporcional ao fitness normalizado)."""
    total = norm_fitness.sum()
    if total == 0:
        idx = np.random.randint(len(population))
    else:
        probs = norm_fitness / total
        idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()


# ─────────────────────────────────────────────
#  6. ALGORITMO GENÉTICO (Steady-State)
# ─────────────────────────────────────────────
def run_ga(
    X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, n_attrs, device, verbose=True
):
    """
    Steady-State GA com elitismo, crossover uniforme e mutação bit-flip.
    Retorna: melhor cromossomo, histórico de best fitness por geração.
    """
    pop_size = PARAMS["pop_size"]
    elitism = PARAMS["elitism"]
    gap = PARAMS["gap"]
    pc = PARAMS["pc"]
    max_gen = PARAMS["max_gen"]
    no_imp_lim = PARAMS["no_improve"]

    # ── Inicialização aleatória ──
    population = np.random.randint(0, 2, size=(pop_size, n_attrs))
    # Garantir ao menos 1 gene ativo em cada cromossomo
    for i in range(pop_size):
        if population[i].sum() == 0:
            population[i][np.random.randint(n_attrs)] = 1

    # Avaliação inicial
    fitness = []

    print(f"\nAvaliando população inicial ({pop_size} cromossomos)...")

    for i in range(pop_size):
        t_chr = time.time()

        fit = fitness_fn(
            population[i],
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_te,
            y_te,
            n_classes,
            device,
        )

        fitness.append(fit)

        print(
            f"  Cromossomo {i + 1:3d}/{pop_size} "
            f"| atributos={population[i].sum():2d} "
            f"| fitness={fit:.4f} "
            f"| tempo={time.time() - t_chr:.1f}s"
        )

    fitness = np.array(fitness)

    best_fit_history = []
    best_fit = fitness.max()
    best_chr = population[fitness.argmax()].copy()
    no_imp = 0

    if verbose:
        print(f"\n  Gen 0 | Best fitness: {best_fit:.4f} | Attrs: {best_chr.sum()}")

    for gen in range(1, max_gen + 1):
        gen_start = time.time()

        norm_fit = normalize_fitness(fitness)

        # ── Elitismo: preservar top-k ──
        elite_idx = np.argsort(fitness)[::-1][:elitism]
        elites = population[elite_idx].copy()
        elite_fit = fitness[elite_idx].copy()

        # ── Steady-State: substituir 'gap' piores por novos filhos ──
        new_chroms = []
        while len(new_chroms) < gap:
            p1 = roulette_select(population, norm_fit)
            p2 = roulette_select(population, norm_fit)

            if np.random.rand() < pc:
                c1, c2 = crossover_uniform(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate(c1)
            c2 = mutate(c2)

            for c in [c1, c2]:
                if c.sum() == 0:
                    c[np.random.randint(n_attrs)] = 1

            new_chroms.extend([c1, c2])

        # Avaliar filhos
        new_chroms = new_chroms[:gap]
        new_fits = np.array(
            [
                fitness_fn(c, X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, device)
                for c in new_chroms
            ]
        )

        # Substituir os piores
        worst_idx = np.argsort(fitness)[:gap]
        for j, idx in enumerate(worst_idx):
            population[idx] = new_chroms[j]
            fitness[idx] = new_fits[j]

        # Reintroduzir elites (garantia)
        for j, idx in enumerate(elite_idx):
            population[idx] = elites[j]
            fitness[idx] = elite_fit[j]

        # Melhor da geração
        gen_best = fitness.max()
        best_fit_history.append(gen_best)

        if gen_best > best_fit:
            best_fit = gen_best
            best_chr = population[fitness.argmax()].copy()
            no_imp = 0
        else:
            no_imp += 1

        if verbose:
            print(
                f"Gen {gen:2d}/{max_gen}"
                f" | Melhor={best_fit:.4f}"
                f" | Atributos={best_chr.sum()}"
                f" | Sem melhora={no_imp}"
                f" | Tempo={time.time() - gen_start:.1f}s"
            )

        if no_imp >= no_imp_lim:
            if verbose:
                print(
                    f"  ► Parada antecipada na geração {gen} ({no_imp} gerações sem melhora)"
                )
            # Completar histórico com último valor
            best_fit_history += [best_fit] * (max_gen - gen)
            break

    return best_chr, best_fit_history


# ─────────────────────────────────────────────
#  7. EXPERIMENTOS (N repetições)
# ─────────────────────────────────────────────
def run_experiments(X, y, n_classes, n_attrs, device):
    n_exp = PARAMS["n_experiments"]
    all_histories = []
    all_best_chr = []
    all_best_fit = []

    print("\n══════════════════════════════════════")
    print(f"  EXECUTANDO {n_exp} EXPERIMENTOS")
    print("══════════════════════════════════════")

    for exp in range(n_exp):
        seed = PARAMS["random_seed"] + exp
        np.random.seed(seed)
        torch.manual_seed(seed)

        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y, seed=seed)

        print(f"\n── Experimento {exp + 1}/{n_exp} ──")
        best_chr, history = run_ga(
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_te,
            y_te,
            n_classes,
            n_attrs,
            device,
            verbose=(exp == 0),  # verbose apenas no 1º experimento
        )
        all_histories.append(history)
        all_best_chr.append(best_chr)
        all_best_fit.append(max(history))
        print(f"  Fitness: {max(history):.4f} | Attrs: {best_chr.sum()}")

    return all_histories, all_best_chr, all_best_fit


# ─────────────────────────────────────────────
#  8. RESULTADOS E GRÁFICOS
# ─────────────────────────────────────────────
def plot_convergence(all_histories, out_dir="."):
    max_len = max(len(h) for h in all_histories)
    # Pad com último valor
    padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in all_histories])
    mean_curve = padded.mean(axis=0)
    std_curve = padded.std(axis=0)
    gens = np.arange(1, max_len + 1)
    n_exp = len(all_histories)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Correção: a legenda usava "20 exp." fixo, independente do número real
    # de experimentos configurado em PARAMS["n_experiments"].
    ax.plot(gens, mean_curve, color="royalblue", lw=2, label=f"Média ({n_exp} exp.)")
    ax.fill_between(
        gens,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        color="royalblue",
        label="±1 Desvio Padrão",
    )
    ax.set_xlabel("Geração", fontsize=12)
    ax.set_ylabel("Melhor Fitness", fontsize=12)
    ax.set_title(
        f"Curva de Convergência do Algoritmo Genético\n(Média de {n_exp} experimentos)",
        fontsize=13,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "curva_convergencia.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Gráfico salvo: {path}")
    return path


def plot_fitness_distribution(all_best_fit, out_dir="."):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        range(1, len(all_best_fit) + 1),
        all_best_fit,
        color="steelblue",
        edgecolor="white",
    )
    ax.axhline(
        np.mean(all_best_fit),
        color="red",
        ls="--",
        lw=1.5,
        label=f"Média = {np.mean(all_best_fit):.4f}",
    )
    ax.set_xlabel("Experimento", fontsize=12)
    ax.set_ylabel("Melhor Fitness", fontsize=12)
    ax.set_title("Melhor Fitness por Experimento", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "fitness_por_experimento.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico salvo: {path}")
    return path


def print_summary(all_best_chr, all_best_fit, feature_names, out_dir="."):
    best_idx = int(np.argmax(all_best_fit))
    best_chr = all_best_chr[best_idx]
    selected = np.where(best_chr == 1)[0]

    print("\n══════════════════════════════════════")
    print("  RESULTADOS FINAIS")
    print("══════════════════════════════════════")
    print(f"  Melhor experimento  : {best_idx + 1}")
    print(f"  Melhor fitness      : {all_best_fit[best_idx]:.4f}")
    print(
        f"  Média fitness       : {np.mean(all_best_fit):.4f} ± {np.std(all_best_fit):.4f}"
    )
    print(f"  Atributos totais    : {len(best_chr)}")
    print(f"  Atributos selecion. : {len(selected)}")
    print(f"  Redução             : {(1 - len(selected) / len(best_chr)) * 100:.1f}%")
    print(f"\n  Cromossomo: {best_chr.tolist()}")
    print(f"\n  Atributos selecionados:")
    for i in selected:
        name = feature_names[i] if i < len(feature_names) else f"A{i}"
        print(f"    [{i:2d}] {name}")

    # Salvar relatório em txt
    report_path = os.path.join(out_dir, "relatorio_resultados.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO - AG + RNA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Melhor experimento  : {best_idx + 1}\n")
        f.write(f"Melhor fitness      : {all_best_fit[best_idx]:.4f}\n")
        f.write(
            f"Média fitness       : {np.mean(all_best_fit):.4f} ± {np.std(all_best_fit):.4f}\n"
        )
        f.write(f"Atributos totais    : {len(best_chr)}\n")
        f.write(f"Atributos selecion. : {len(selected)}\n")
        f.write(
            f"Redução             : {(1 - len(selected) / len(best_chr)) * 100:.1f}%\n\n"
        )
        f.write(f"Cromossomo:\n{best_chr.tolist()}\n\n")
        f.write("Atributos selecionados:\n")
        for i in selected:
            name = feature_names[i] if i < len(feature_names) else f"A{i}"
            f.write(f"  [{i}] {name}\n")
        f.write("\n\nParâmetros utilizados:\n")
        for k, v in PARAMS.items():
            f.write(f"  {k}: {v}\n")
    print(f"\n  Relatório salvo: {report_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    import sys

    print("╔══════════════════════════════════════════════╗")
    print("║  AG + RNA - Seleção de Atributos - CEFET-RJ ║")
    print("╚══════════════════════════════════════════════╝")

    # ── Arquivo de dados ──
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "dados.csv"  # nome padrão

    if not os.path.exists(csv_path):
        print(f"\n  ERRO: arquivo '{csv_path}' não encontrado.")
        print("  Uso: python ag_rna_cancer.py <caminho_do_csv>")
        return

    out_dir = "resultados"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # 1. Pré-processamento
    cache_X = csv_path + "_X.npy"
    cache_y = csv_path + "_y.npy"

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print("Carregando dados pré-processados do cache...")
        X = np.load(cache_X)
        y = np.load(cache_y)
    else:
        X, y = preprocess(csv_path)
        np.save(cache_X, X)
        np.save(cache_y, y)
        print(f"Cache salvo: {cache_X}, {cache_y}")
    n_attrs = X.shape[1]
    n_classes = len(np.unique(y))

    # Nome dos atributos (para relatório)
    df_head = pd.read_csv(csv_path, sep=";", nrows=0)
    feature_names = [c for c in df_head.columns if c != df_head.columns[-1]]

    print(f"\nAtributos: {n_attrs} | Classes: {n_classes} | Amostras: {len(y)}")
    print(f"\nParâmetros AG:")
    print(
        f"  Pop.: {PARAMS['pop_size']} | Pc: {PARAMS['pc']} | "
        f"Pm: 1/L={1 / n_attrs:.4f} | Elitismo: {PARAMS['elitism']} | Gap: {PARAMS['gap']}"
    )

    # 2. Experimentos
    all_histories, all_best_chr, all_best_fit = run_experiments(
        X, y, n_classes, n_attrs, DEVICE
    )

    # 3. Resultados
    print_summary(all_best_chr, all_best_fit, feature_names, out_dir)
    plot_convergence(all_histories, out_dir)
    plot_fitness_distribution(all_best_fit, out_dir)

    elapsed = time.time() - t0
    print(f"\n  Tempo total: {elapsed / 60:.1f} minutos")
    print(f"\n  Arquivos salvos em: ./{out_dir}/")
    print("  ✓ Concluído!\n")


if __name__ == "__main__":
    main()
