
"""
Analisi degli errori e generazione di grafici comparativi a partire dai
risultati salvati da evaluate_macid.py.

Da eseguire dopo evaluate_macid.py, quando risultati_macid.csv è disponibile.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RISULTATI_PATH = "risultati_macid.csv"
DATA_PATH = "macid_data.csv"

# Baseline da CALAMITA (Nissim, Croce et al. 2025, arXiv:2512.04759,
# Sezione 4.1.14 "MACID", risultati zero-shot sui 4 modelli open-weight testati
# nel paper originale).
BASELINE_CALAMITA = {
    "LLaMA3.1-70B (zero-shot)": 0.58,
    "LLaMA3.1-8B (zero-shot)": 0.43,
    "Anita-8B (zero-shot)": 0.43,
    "Minerva-7B (zero-shot)": 0.27,
    "Caso casuale": 0.25,
}

STRATEGIE_BASE = ["zero-shot", "zero-shot+role", "few-shot", "chain-of-thought"]


def carica_dati():
    df_ris = pd.read_csv(RISULTATI_PATH)
    df_data = pd.read_csv(DATA_PATH)
    df = df_ris.merge(df_data, on="id", how="left")
    return df


def grafico_confronto_modelli(df, output_path="grafico_modelli_vs_baseline.png"):
    
    # grafico comparativo a 5 modelli x 4 strategie.
    df_unique = df[df["strategia"].isin(STRATEGIE_BASE)].drop_duplicates(
        subset=["id", "modello", "strategia"])
    tabella = df_unique.groupby(["modello", "strategia"])["corretto"].mean().unstack("strategia")
    modelli = tabella.index.tolist()
    strategie = [s for s in STRATEGIE_BASE if s in tabella.columns]

    fig, ax = plt.subplots(figsize=(11, 6))

    n_strategie = len(strategie)
    larghezza_barra = 0.8 / n_strategie
    x = np.arange(len(modelli))

    colori_per_nome = {
        "zero-shot": "#4C72B0",
        "zero-shot+role": "#DD8452",
        "few-shot": "#55A868",
        "chain-of-thought": "#C44E52",
    }
    for i, strategia in enumerate(strategie):
        offset = (i - n_strategie / 2) * larghezza_barra + larghezza_barra / 2
        valori = tabella[strategia].values

        # Calcoliamo l'intervallo di confidenza per ogni barra
        errori_basso, errori_alto = [], []
        for modello in modelli:
            sotto = df_unique[(df_unique["modello"] == modello) & (df_unique["strategia"] == strategia)]
            corrette, totali = sotto["corretto"].sum(), len(sotto)
            acc = corrette / totali if totali else 0
            ic_b, ic_a = intervallo_confidenza_wilson(corrette, totali)
            errori_basso.append(acc - ic_b)
            errori_alto.append(ic_a - acc)

        ax.bar(x + offset, valori, larghezza_barra,
               label=strategia, color=colori_per_nome[strategia],
               yerr=[errori_basso, errori_alto], capsize=3,
               error_kw={"linewidth": 1, "ecolor": "#333333"})

    stili = ["--", "-.", ":", "-", (0, (3, 1, 1, 1))]
    
    offset_per_valore = {}
    for (nome, valore), stile in zip(BASELINE_CALAMITA.items(), stili):
        ax.axhline(y=valore, linestyle=stile, color="gray", alpha=0.7, linewidth=1.2)
        n_ripetizioni = offset_per_valore.get(valore, 0)
        offset_per_valore[valore] = n_ripetizioni + 1
        ax.text(len(modelli) - 0.5, valore + 0.005 + n_ripetizioni * 0.025,
                nome, fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(modelli)
    ax.set_ylabel("Accuratezza")
    ax.set_title("Accuratezza su MACID: modelli locali (Ollama) vs baseline CALAMITA\n(barre di errore = intervallo di confidenza al 95%)")
    ax.set_ylim(0, 1)
    ax.legend(title="Strategia", loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Grafico salvato in: {output_path}")
    plt.close()


# ------------------------------------------------------------------
# SIGNIFICATIVITÀ STATISTICA
# ------------------------------------------------------------------
#
# Con solo 100 tuple, una differenza come 0.55 vs 0.60 tra due modelli
# potrebbe essere semplice rumore statistico.
#
# 1. Intervallo di confidenza di Wilson per ogni accuratezza singola
#    (più affidabile della formula normale classica su campioni piccoli
#    o con accuratezze vicine a 0 o 1).
# 2. Test di McNemar: confronta due condizioni (es. due modelli, o due strategie)
#    SULLE STESSE 100 TUPLE,
#    guardando solo i casi in cui le due condizioni sono in disaccordo.


def intervallo_confidenza_wilson(corrette, totali, confidenza=0.95):
    if totali == 0:
        return (float("nan"), float("nan"))
    p_hat = corrette / totali
    z = stats.norm.ppf(1 - (1 - confidenza) / 2)
    denom = 1 + z**2 / totali
    centro = (p_hat + z**2 / (2 * totali)) / denom
    margine = (z * np.sqrt(p_hat * (1 - p_hat) / totali + z**2 / (4 * totali**2))) / denom
    return (max(0.0, centro - margine), min(1.0, centro + margine))


def tabella_accuratezza_con_ic(df, gruppo_colonne):
    righe = []
    for chiave, gruppo in df.groupby(gruppo_colonne):
        corrette = gruppo["corretto"].sum()
        totali = len(gruppo)
        acc = corrette / totali
        ic_basso, ic_alto = intervallo_confidenza_wilson(corrette, totali)

        if not isinstance(chiave, tuple):
            chiave = (chiave,)
        riga = dict(zip(gruppo_colonne, chiave))
        riga.update({
            "accuratezza": round(acc, 3),
            "ic_95_basso": round(ic_basso, 3),
            "ic_95_alto": round(ic_alto, 3),
            "corrette": corrette,
            "totali": totali,
        })
        righe.append(riga)

    return pd.DataFrame(righe)


def test_mcnemar(corretto_a, corretto_b):
    """Test di McNemar per due condizioni appaiate.
    Restituisce: (statistica, p-value, tabella di contingenza 2x2)
    """
    corretto_a = np.asarray(corretto_a, dtype=bool)
    corretto_b = np.asarray(corretto_b, dtype=bool)

    entrambi_giusti = np.sum(corretto_a & corretto_b)
    solo_a_giusto = np.sum(corretto_a & ~corretto_b)
    solo_b_giusto = np.sum(~corretto_a & corretto_b)
    entrambi_sbagliati = np.sum(~corretto_a & ~corretto_b)

    n_discordanti = solo_a_giusto + solo_b_giusto

    if n_discordanti == 0:
        # le due condizioni sbagliano/giustano esattamente sulle stesse tuple:
        return (0.0, 1.0, (entrambi_giusti, solo_a_giusto, solo_b_giusto, entrambi_sbagliati))

    if n_discordanti < 25:
        # con pochi casi discordanti usiamo la versione esatta (binomiale),
        p_value = 2 * stats.binom.cdf(min(solo_a_giusto, solo_b_giusto), n_discordanti, 0.5)
        p_value = min(p_value, 1.0)
        statistica = min(solo_a_giusto, solo_b_giusto)
    else:
        statistica = (abs(solo_a_giusto - solo_b_giusto) - 1) ** 2 / n_discordanti
        p_value = 1 - stats.chi2.cdf(statistica, df=1)

    return (statistica, p_value, (entrambi_giusti, solo_a_giusto, solo_b_giusto, entrambi_sbagliati))


def confronta_modelli_mcnemar(df, strategia="zero-shot", alpha=0.05):
    """Confronta tutte le coppie di modelli (a parità di strategia) con
    il test di McNemar, e stampa una tabella con i p-value."""
    df_strat = df[df["strategia"] == strategia].drop_duplicates(subset=["id", "modello"])
    pivot = df_strat.pivot(index="id", columns="modello", values="corretto")
    modelli = pivot.columns.tolist()

    print(f"\n{'='*70}")
    print(f"TEST DI MCNEMAR TRA MODELLI (strategia: {strategia})")
    print(f"H0: i due modelli hanno la stessa accuratezza sulla popolazione")
    print(f"{'='*70}")

    righe = []
    for i in range(len(modelli)):
        for j in range(i + 1, len(modelli)):
            m1, m2 = modelli[i], modelli[j]
            # Rimuoviamo le tuple in cui manca la risposta per uno dei due modelli
            sotto = pivot[[m1, m2]].dropna()
            stat, p, (bb, ba, ab, aa) = test_mcnemar(sotto[m1], sotto[m2])

            acc1 = sotto[m1].mean()
            acc2 = sotto[m2].mean()
            significativo = "SÌ" if p < alpha else "no"

            righe.append({
                "modello_1": m1, "acc_1": round(acc1, 3),
                "modello_2": m2, "acc_2": round(acc2, 3),
                "p_value": round(p, 4),
                f"significativo (α={alpha})": significativo,
            })

    df_confronti = pd.DataFrame(righe)
    print(df_confronti.to_string(index=False))
    print(f"\nNota: 'significativo' = differenza non spiegabile dal caso (p < {alpha}).")
    print("Con sole 100 tuple, molte differenze plausibili NON risulteranno significative:")
    print("risultato onesto, non un difetto dell'analisi.")
    return df_confronti


def confronta_strategie_mcnemar(df, modello, alpha=0.05):
    """Confronta tutte le coppie di strategie (a parità di modello) con
    il test di McNemar."""
    df_mod = df[df["modello"] == modello].drop_duplicates(subset=["id", "strategia"])
    pivot = df_mod.pivot(index="id", columns="strategia", values="corretto")
    strategie = pivot.columns.tolist()

    print(f"\n{'='*70}")
    print(f"TEST DI MCNEMAR TRA STRATEGIE (modello: {modello})")
    print(f"{'='*70}")

    righe = []
    for i in range(len(strategie)):
        for j in range(i + 1, len(strategie)):
            s1, s2 = strategie[i], strategie[j]
            sotto = pivot[[s1, s2]].dropna()
            stat, p, _ = test_mcnemar(sotto[s1], sotto[s2])

            righe.append({
                "strategia_1": s1, "acc_1": round(sotto[s1].mean(), 3),
                "strategia_2": s2, "acc_2": round(sotto[s2].mean(), 3),
                "p_value": round(p, 4),
                f"significativo (α={alpha})": "SÌ" if p < alpha else "no",
            })

    df_confronti = pd.DataFrame(righe)
    print(df_confronti.to_string(index=False))
    return df_confronti


def analisi_errori_per_pattern_verbale(df):
    """Analizza se il modello sbaglia di più quando l'intruso condivide
    lo stesso verbo principale di almeno un'altra frase"""

    def stesso_verbo_radice(row):
        """Euristica semplice: controlla se le prime parole-chiave (verbi comuni
        nel dataset: spingere, premere, spostare, sollevare...) si ripetono
        tra la frase intrusa e almeno un'altra frase del gruppo.

        Restituisce una delle tre categorie:
        - "condiviso": il verbo dell'intruso ricorre in almeno un'altra frase
        - "non_condiviso": il verbo dell'intruso NON ricorre altrove
        - "non_determinato": il verbo dell'intruso non rientra nella lista di
          radici note, quindi l'euristica non può classificare il caso. Questi
          casi vengono tenuti separati per non inquinare la categoria
          "non_condiviso"""
        verbi_chiave = ["sping", "prem", "spost", "solleva", "allontan",
                        "abbass", "alz", "scost", "schiacc", "pigi"]
        frasi = [row["s1"], row["s2"], row["s3"], row["s4"]]
        intruso_idx = int(row["atteso"]) - 1
        frase_intrusa = frasi[intruso_idx].lower()

        verbo_intruso = next((v for v in verbi_chiave if v in frase_intrusa), None)
        if verbo_intruso is None:
            return "non_determinato"

        altre_frasi = [f.lower() for i, f in enumerate(frasi) if i != intruso_idx]
        return "condiviso" if any(verbo_intruso in f for f in altre_frasi) else "non_condiviso"

    df_unique = df.drop_duplicates(subset=["id", "modello", "strategia"]).copy()
    df_unique["verbo_condiviso"] = df_unique.apply(stesso_verbo_radice, axis=1)

    print("\n" + "=" * 60)
    print("ACCURATEZZA IN BASE ALLA CONDIVISIONE DEL VERBO (per modello e strategia)")
    print("(intruso con verbo diverso vs. intruso con verbo uguale ad almeno")
    print(" un'altra frase nel gruppo -> caso linguisticamente più ambiguo)")
    print("=" * 60)

    tabella = df_unique.groupby(["modello", "strategia", "verbo_condiviso"])["corretto"].agg(["mean", "count"])
    print(tabella.to_string())

    return df_unique


def esempi_errori(df, modello=None, strategia="zero-shot", n=5):
    df_filtrato = df[(df["strategia"] == strategia) & (~df["corretto"])]
    if modello is not None:
        df_filtrato = df_filtrato[df_filtrato["modello"] == modello]
    df_filtrato = df_filtrato.drop_duplicates(subset=["id", "modello"])

    titolo_modello = f"{modello} | " if modello else ""
    print(f"\n--- Esempi di errori ({titolo_modello}{strategia}) ---")
    for _, row in df_filtrato.head(n).iterrows():
        print(f"\nID {row['id']} | atteso={row['atteso']} | predetto={row['predetto']}")
        for i in range(1, 5):
            marcatore = " <-- INTRUSO ATTESO" if i == row["atteso"] else ""
            marcatore += " <-- PREDETTO DAL MODELLO" if i == row["predetto"] else ""
            print(f"  {i}: {row[f's{i}']}{marcatore}")


# ------------------------------------------------------------------
# ANALISI DEGLI ERRORI APPROFONDITA 
# ------------------------------------------------------------------
#
# Le due funzioni seguenti sfruttano i risultati già salvati per
# caratterizzare la presenza di un bias sistematico nella posizione
# scelta, e la struttura degli errori. 
# si limitano di default alle quattro strategie base.

def analisi_bias_posizionale(df):
    """Verifica se i modelli tendono a scegliere certe posizioni (1-4) a
    prescindere dalla posizione reale dell'intruso."""
    d = df[df["strategia"].isin(STRATEGIE_BASE)].copy()
    pred = d.dropna(subset=["predetto"]).copy()
    pred["predetto"] = pred["predetto"].astype(int)

    print("\n" + "=" * 60)
    print("BIAS POSIZIONALE")
    print("=" * 60)
    gold = d.drop_duplicates("id")["atteso"].value_counts(normalize=True).sort_index()
    print("Distribuzione posizione dell'intruso reale (gold, %):")
    print((gold * 100).round(1).to_string())
    print(f"\nDistribuzione posizione predetta ({len(pred)} risposte valide, %):")
    print((pred["predetto"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())

    print("\nDistribuzione posizione predetta per modello (%):")
    tab = (pred.groupby("modello")["predetto"].value_counts(normalize=True)
           .unstack().reindex(columns=[1, 2, 3, 4]) * 100)
    print(tab.round(1).to_string())

    print("\nAccuratezza in funzione della posizione reale dell'intruso:")
    print(d.groupby("atteso")["corretto"].mean().round(3).to_string())
    return tab


def matrice_confusione_posizioni(df):
    """matrice di confusione fra posizione reale dell'intruso e
    posizione predetta, sulle sole risposte valide."""
    d = df[df["strategia"].isin(STRATEGIE_BASE)].copy()
    pred = d.dropna(subset=["predetto"]).copy()
    pred["predetto"] = pred["predetto"].astype(int)
    conf = pd.crosstab(pred["atteso"], pred["predetto"], normalize="index") * 100

    print("\n" + "=" * 60)
    print("MATRICE DI CONFUSIONE POSIZIONALE (righe=gold, colonne=predetto, %)")
    print("=" * 60)
    print(conf.round(1).to_string())

    err = pred[pred["atteso"] != pred["predetto"]]
    print(f"\nQuando sbagliano ({len(err)} errori), posizione scelta (%):")
    print((err["predetto"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    return conf


if __name__ == "__main__":
    df = carica_dati()

    print("Caricati", len(df), "risultati totali.")
    print("Modelli presenti:", df["modello"].unique().tolist())
    print()

    grafico_confronto_modelli(df)

    # Tabelle di accuratezza con intervallo di confidenza al 95% 
    print("\n" + "=" * 70)
    print("ACCURATEZZA CON INTERVALLO DI CONFIDENZA AL 95% (Wilson)")
    print("=" * 70)
    tabella_per_modello = tabella_accuratezza_con_ic(df.drop_duplicates(subset=["id","modello","strategia"]),
                                                       ["modello"])
    print("\nPer modello (media su tutte le strategie):")
    print(tabella_per_modello.to_string(index=False))

    tabella_per_modello_strategia = tabella_accuratezza_con_ic(
        df.drop_duplicates(subset=["id", "modello", "strategia"]),
        ["modello", "strategia"]
    )
    print("\nPer modello e strategia:")
    print(tabella_per_modello_strategia.to_string(index=False))

    tabella_per_modello_strategia.to_csv("accuratezza_con_intervalli_confidenza.csv", index=False)
    print("\nSalvata in: accuratezza_con_intervalli_confidenza.csv")

    # Test di McNemar: i modelli differiscono davvero? 
    for strategia in df["strategia"].unique():
        confronta_modelli_mcnemar(df, strategia=strategia)

    # Test di McNemar: le strategie differiscono davvero, per ogni modello? 
    for modello in df["modello"].unique():
        confronta_strategie_mcnemar(df, modello=modello)

    # Analisi qualitativa (pattern verbale + esempi di errori) 
    analisi_errori_per_pattern_verbale(df)

    # Analisi degli errori approfondita (bias, confusione) 
    analisi_bias_posizionale(df)
    matrice_confusione_posizioni(df)

    for modello in df["modello"].unique():
        esempi_errori(df, modello=modello, strategia="zero-shot", n=3)

    print("\nFatto. ")
