"""
Valutazione di un LLM sul task MACID (Multimodal ACtion IDentification).

Strategie base testate:
  - zero-shot        
  - zero-shot+role  
  - few-shot         
  - chain-of-thought 

Tutte le strategie tranne lo zero-shot puro usano:
  - Role Prompting
  - Output vincolato a tag XML: la risposta finale va dentro
    <risposta>N</risposta>, estratta poi in modo deterministico via regex

ESECUZIONE DINAMICA DA RIGA DI COMANDO:
  python3 evaluate_macid.py
      -> esegue tutto il catalogo MODELLI_DISPONIBILI con le 4 strategie base

  python3 evaluate_macid.py --modelli llama3.1:8b
      -> esegue un solo modello 

  python3 evaluate_macid.py --strategie few-shot+cot --modelli qwen2.5:7b gemma2:9b
      -> esegue la strategia di follow-up "few-shot+cot" solo sui modelli indicati

ogni singola risposta del modello viene scritta subito su disco (risultati_macid.csv)

"""

import argparse

import os
import re
import csv
import time
import pandas as pd
import ollama

# ------------------------------------------------------------------
# CONFIGURAZIONE
# ------------------------------------------------------------------

# Catalogo di tutti i modelli

MODELLI_DISPONIBILI = [
    "llama3.1:8b",
    "mistral",
    "qwen2.5:7b",
    "gemma2:9b",
    "phi4:latest"
]

DATA_PATH = "macid_data.csv"
OUTPUT_PATH = "risultati_macid.csv"
SLEEP_BETWEEN_CALLS = 0.0   


# ------------------------------------------------------------------
# COSTRUZIONE DEI PROMPT
# ------------------------------------------------------------------

# Identità assegnata al modello nelle strategie "potenziate" (tutte tranne
# lo zero-shot), per restringere lo spazio di risposta verso un'analisi
# linguistica più attenta alle sfumature semantiche del task

RUOLO = ("Agisci come un esperto linguista e analista semantico, specializzato "
         "nel riconoscere le sfumature e le ambiguità delle azioni descritte "
         "in lingua italiana.")

# Istruzione di formato condivisa dalle strategie potenziate: il modello deve
# isolare la risposta finale dentro un tag XML dedicato

ISTRUZIONE_FORMATO_XML = ("Restituisci l'ID della frase intrusa esclusivamente "
                           "all'interno del tag <risposta>...</risposta>, "
                           "ad esempio <risposta>3</risposta>.")


def build_prompt_zeroshot(row):
    """Prompt SENZA Role Prompting né tag XML"""
    return f"""Le seguenti 4 frasi sono descrizioni di azioni fisiche. Tre di queste azioni sono dello stesso tipo, mentre una è di un tipo diverso. Individua la frase che descrive l'azione di tipo diverso rispondendo soltanto con il numero della frase (1, 2, 3 o 4).

1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}"""


def build_prompt_zeroshot_role(row):
    """Zero-shot con Role Prompting, ma senza esempi (few-shot) e senza
    CoT: isola l'effetto del solo Role Prompting.
    Include il vincolo di formato XML"""
    return f"""{RUOLO}

Le seguenti 4 frasi sono descrizioni di azioni fisiche. Tre di queste azioni sono dello stesso tipo, mentre una è di un tipo diverso. Individua la frase che descrive l'azione di tipo diverso.

1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}

{ISTRUZIONE_FORMATO_XML}"""


# ------------------------------------------------------------------
# ESPERIMENTO DI SENSIBILITÀ AL PARAPHRASING (robustezza del prompt)
# ------------------------------------------------------------------
# Due riformulazioni della strategia zero-shot+role.

# Variante 2: ruolo più assertivo/diretto:
RUOLO_V2 = ("Sei un linguista esperto di semantica delle azioni. Il tuo compito "
            "è distinguere con precisione i diversi tipi di azione fisica "
            "descritti in lingua italiana.")

ISTRUZIONE_FORMATO_XML_V2 = ("Fornisci il numero della frase diversa unicamente "
                             "dentro il tag <risposta>...</risposta>, per esempio "
                             "<risposta>3</risposta>.")


def build_prompt_role_v2(row):
    """Riformulazione 2 di zero-shot+role: stesso contenuto informativo,
    wording diverso (ruolo più imperativo). Usata per l'esperimento di sensibilità al paraphrasing."""
    return f"""{RUOLO_V2}

Qui sotto trovi quattro frasi, ciascuna descrive un'azione fisica. Tre descrivono lo stesso tipo di azione; una descrive un tipo di azione differente. Trova quella diversa dalle altre.

1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}

{ISTRUZIONE_FORMATO_XML_V2}"""


# Variante 3: ruolo più neutro:
RUOLO_V3 = ("In qualità di analista del linguaggio, valuta con attenzione le "
            "seguenti descrizioni di azioni.")

ISTRUZIONE_FORMATO_XML_V3 = ("Indica il numero corrispondente alla frase estranea "
                             "soltanto all'interno del tag <risposta>...</risposta>, "
                             "come in <risposta>3</risposta>.")


def build_prompt_role_v3(row):
    """Riformulazione 3 di zero-shot+role: stesso contenuto informativo,
    wording diverso (ruolo più neutro). Usata per l'esperimento di sensibilità al paraphrasing."""
    return f"""{RUOLO_V3}

Tra le quattro frasi seguenti, tutte relative ad azioni fisiche, tre appartengono alla stessa categoria di azione e una a una categoria diversa. Identifica la frase che non appartiene alla stessa categoria delle altre tre.

1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}

{ISTRUZIONE_FORMATO_XML_V3}"""


FEWSHOT_EXAMPLES_XML = """Esempio 1:
1: L'uomo preme il pulsante dell'ascensore
2: La donna preme il tasto del telecomando
3: Il ragazzo spinge il carrello lungo il corridoio
4: Il soldato preme il grilletto della pistola
(le frasi 1, 2 e 4 descrivono tutte un'azione di "premere" qualcosa con un dito, mentre la 3 descrive un'azione di "spingere" con tutto il corpo)
<risposta>3</risposta>

Esempio 2:
1: La donna spinge la sedia a rotelle
2: L'infermiere spinge la barella
3: Il ragazzo preme il pulsante di accensione
4: Il facchino spinge il carrello dei bagagli
(le frasi 1, 2 e 4 descrivono tutte un'azione di "spingere" un oggetto con ruote per farlo muovere, mentre la 3 descrive un'azione di "premere" un pulsante)
<risposta>3</risposta>

Esempio 3:
1: Il ragazzo si solleva sulle braccia per alzarsi
2: L'uomo si tira su in ginocchio
3: La donna spinge il carrello della spesa
4: Il soldato si spinge sulle braccia per mettersi in ginocchio
(le frasi 1, 2 e 4 descrivono tutte un movimento del corpo per rialzarsi, mentre la 3 descrive lo spostamento di un oggetto)
<risposta>3</risposta>

"""


def build_prompt_fewshot(row):
    """Few-shot con Role Prompting e risposta finale vincolata al tag XML
    <risposta>...</risposta>, per un'estrazione deterministica."""
    domanda = f"""Ora risolvi questo caso:
1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}"""
    return (RUOLO + "\n\n"
            "Le seguenti 4 frasi sono descrizioni di azioni fisiche. Tre di queste azioni "
            "sono dello stesso tipo, mentre una è di un tipo diverso. Ti mostro alcuni "
            "esempi già risolti, poi ti chiedo di risolvere un nuovo caso nello stesso modo.\n\n"
            + ISTRUZIONE_FORMATO_XML + "\n\n"
            + FEWSHOT_EXAMPLES_XML + domanda)


def build_prompt_cot(row):
    """Chain-of-thought con Role Prompting, istruzione di ragionamento
    esplicito passo dopo passo, e risposta finale vincolata al tag XML."""
    return f"""{RUOLO}

Le seguenti 4 frasi sono descrizioni di azioni fisiche. Tre di queste azioni sono dello stesso tipo, mentre una è di un tipo diverso.

1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}

Analizza le frasi passo dopo passo (Let's think step by step): qual è il tipo di azione condiviso da tre delle frasi? Quale frase non rientra in quel tipo di azione, anche se può usare un verbo simile?

{ISTRUZIONE_FORMATO_XML}"""


def build_prompt_fewshot_cot(row):
    """Combina gli esempi risolti del few-shot con la richiesta di
    ragionamento esplicito del chain-of-thought, Role Prompting e tag XML.
    Strategia di follow-up, da applicare tipicamente solo ai modelli che
    si sono dimostrati migliori nelle 4 strategie base, per vedere se la
    combinazione porta un miglioramento ulteriore."""
    domanda = f"""Ora risolvi questo caso. Analizza le frasi passo dopo passo (Let's think step by step),
poi dai la risposta finale.

1: {row['s1']}
2: {row['s2']}
3: {row['s3']}
4: {row['s4']}"""
    return (RUOLO + "\n\n"
            "Le seguenti 4 frasi sono descrizioni di azioni fisiche. Tre di queste azioni "
            "sono dello stesso tipo, mentre una è di un tipo diverso. Ti mostro alcuni "
            "esempi già risolti con il loro ragionamento, poi ti chiedo di risolvere un "
            "nuovo caso nello stesso modo.\n\n"
            + ISTRUZIONE_FORMATO_XML + "\n\n"
            + FEWSHOT_EXAMPLES_XML + domanda)


STRATEGIE_BASE = {
    "zero-shot": build_prompt_zeroshot,
    "zero-shot+role": build_prompt_zeroshot_role,
    "few-shot": build_prompt_fewshot,
    "chain-of-thought": build_prompt_cot,
}


STRATEGIE_DISPONIBILI = {
    **STRATEGIE_BASE,
    "few-shot+cot": build_prompt_fewshot_cot,
    # Riformulazioni per l'esperimento di sensibilità al paraphrasing
    # (stesso contenuto di zero-shot+role, wording diverso).
    "role-v2": build_prompt_role_v2,
    "role-v3": build_prompt_role_v3,
}


# ------------------------------------------------------------------
# SALVATAGGIO INCREMENTALE 
# ------------------------------------------------------------------

CAMPI_CSV = ["modello", "id", "strategia", "atteso", "predetto", "corretto", "risposta_raw"]


def carica_combinazioni_gia_fatte():
    if not os.path.exists(OUTPUT_PATH):
        return set()

    try:
        df_esistente = pd.read_csv(OUTPUT_PATH)
    except Exception:
        # file vuoto o corrotto (es. interrotto a metà scrittura) -> ripartiamo da zero
        return set()

    fatte = set(
        zip(df_esistente["modello"], df_esistente["id"], df_esistente["strategia"])
    )
    return fatte


def assicura_csv_con_intestazione():
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CAMPI_CSV)
            writer.writeheader()


def salva_riga_incrementale(riga):
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPI_CSV)
        writer.writerow(riga)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------
# CHIAMATA AL MODELLO ED ESTRAZIONE DELLA RISPOSTA
# ------------------------------------------------------------------

def chiedi_al_modello(prompt, nome_modello, max_retries=2):
    for tentativo in range(max_retries):
        try:
            risposta = ollama.chat(
                model=nome_modello,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            return risposta["message"]["content"]
        except Exception as e:
            if tentativo == max_retries - 1:
                print(f"  Errore dopo {max_retries} tentativi su {nome_modello}: {e}")
                print("  Assicurati che Ollama sia in esecuzione e che il modello sia "
                      f"stato scaricato con: ollama pull {nome_modello}")
                return None
            time.sleep(2)
    return None


def estrai_numero(testo):
    if testo is None:
        return None

    match_xml = re.search(r'<risposta>\s*([1-4])\s*</risposta>', testo, re.IGNORECASE)
    if match_xml:
        return int(match_xml.group(1))

    # Fallback: nessun tag XML trovato 
    #  -> cerchiamo l'ultimo numero 1-4 nel testo.
    matches = re.findall(r'\b[1-4]\b', testo)
    return int(matches[-1]) if matches else None

# ------------------------------------------------------------------
# CICLO DI VALUTAZIONE
# ------------------------------------------------------------------

def valuta(modelli, strategie):
    """Esegue la valutazione per i modelli e le strategie indicati.
    modelli: lista di nomi modello (es. ["llama3.1:8b", "mistral"])
    strategie: dict {nome_strategia: funzione_build_prompt}
    """
    df = pd.read_csv(DATA_PATH)

    assicura_csv_con_intestazione()
    gia_fatte = carica_combinazioni_gia_fatte()

    combinazioni_modello_strategia = [
        (nome_modello, nome_strategia, build_fn)
        for nome_modello in modelli
        for nome_strategia, build_fn in strategie.items()
    ]

    totale = len(df) * len(combinazioni_modello_strategia)
    contatore = 0
    saltate = 0

    if gia_fatte:
        print(f"Trovati {len(gia_fatte)} risultati già presenti in {OUTPUT_PATH}.")
        print("Lo script li salterà e riprenderà da dove si era interrotto.\n")

    for nome_modello, nome_strategia, build_fn in combinazioni_modello_strategia:
        print(f"\n{'='*60}\nMODELLO: {nome_modello} | STRATEGIA: {nome_strategia}\n{'='*60}")

        for _, row in df.iterrows():
            atteso = int(row["intruder"])
            id_riga = row["id"]
            contatore += 1

            chiave = (nome_modello, id_riga, nome_strategia)
            if chiave in gia_fatte:
                saltate += 1
                continue

            prompt = build_fn(row)
            risposta_raw = chiedi_al_modello(prompt, nome_modello)
            predetto = estrai_numero(risposta_raw)
            corretto = (predetto == atteso)

            riga_risultato = {
                "modello": nome_modello,
                "id": id_riga,
                "strategia": nome_strategia,
                "atteso": atteso,
                "predetto": predetto,
                "corretto": corretto,
                "risposta_raw": (risposta_raw or "").replace("\n", " | "),
            }

            salva_riga_incrementale(riga_risultato)

            print(f"[{contatore}/{totale}] {nome_modello:<14} | id={id_riga:>3} | "
                  f"{nome_strategia:<16} | atteso={atteso} predetto={predetto} "
                  f"| {'OK' if corretto else 'NO'}")

            time.sleep(SLEEP_BETWEEN_CALLS)


    if saltate:
        print(f"\n({saltate} combinazioni erano già state completate in una "
              "esecuzione precedente e sono state saltate)")

    df_risultati = pd.read_csv(OUTPUT_PATH)
    return df_risultati


def riassunto(df_risultati):
    print("\n" + "=" * 60)
    print("RIASSUNTO ACCURATEZZA PER MODELLO E STRATEGIA")
    print("=" * 60)
    riepilogo = df_risultati.groupby(["modello", "strategia"])["corretto"].agg(["mean", "sum", "count"])
    riepilogo.columns = ["accuratezza", "corrette", "totali"]
    riepilogo = riepilogo.sort_values("accuratezza", ascending=False)
    print(riepilogo.to_string())

    # La media "per modello" qui sotto si limita alle strategie BASE
    # (zero-shot, few-shot, chain-of-thought), così il confronto tra modelli
    # resta omogeneo anche se alcuni hanno anche strategie extra come
    # "few-shot+cot" eseguite solo su un sottoinsieme di modelli.

    df_base = df_risultati[df_risultati["strategia"].isin(STRATEGIE_BASE.keys())]

    if len(df_base) > 0:
        print("\n" + "-" * 60)
        print("RIASSUNTO ACCURATEZZA SOLO PER MODELLO (media sulle strategie base presenti)")
        print("-" * 60)
        riepilogo_modello = df_base.groupby("modello")["corretto"].agg(["mean", "sum", "count"])
        riepilogo_modello.columns = ["accuratezza", "corrette", "totali"]
        riepilogo_modello = riepilogo_modello.sort_values("accuratezza", ascending=False)
        print(riepilogo_modello.to_string())

    # Per ogni strategia presente ma NON tra quelle base (es. "few-shot+cot"),
    # mostriamo un confronto diretto con la migliore delle strategie base
    # disponibili per lo stesso modello, per vedere se ha aiutato.
    strategie_extra_presenti = set(df_risultati["strategia"].unique()) - set(STRATEGIE_BASE.keys())
    for nome_strategia_extra in strategie_extra_presenti:
        df_extra = df_risultati[df_risultati["strategia"] == nome_strategia_extra]
        print("\n" + "-" * 60)
        print(f"CONFRONTO: '{nome_strategia_extra}' vs. la migliore strategia base disponibile")
        print("-" * 60)
        for modello in df_extra["modello"].unique():
            acc_extra = df_extra[df_extra["modello"] == modello]["corretto"].mean()
            ha_strategie_base = (
                modello in riepilogo.index.get_level_values("modello")
                and len(riepilogo.loc[[modello]].index.get_level_values("strategia")
                        .intersection(STRATEGIE_BASE.keys())) > 0
            )
            if not ha_strategie_base:
                print(f"  {modello}: nessuna strategia base eseguita su questo modello "
                      f"in questa run, quindi nessun confronto disponibile "
                      f"({nome_strategia_extra} = {acc_extra:.3f})")
                continue
            riepilogo_base_modello = riepilogo.loc[modello].loc[
                riepilogo.loc[modello].index.intersection(STRATEGIE_BASE.keys())
            ]
            acc_migliore_base = riepilogo_base_modello["accuratezza"].max()
            strategia_migliore_base = riepilogo_base_modello["accuratezza"].idxmax()
            differenza = acc_extra - acc_migliore_base
            segno = "+" if differenza >= 0 else ""
            print(f"  {modello}:")
            print(f"    {nome_strategia_extra:<16} = {acc_extra:.3f}")
            print(f"    {strategia_migliore_base:<16} = {acc_migliore_base:.3f}  (migliore tra le base)")
            print(f"    differenza       = {segno}{differenza:.3f}")

    print("\nBaseline CALAMITA (zero-shot, modelli open-weight, arXiv:2512.04759):")
    print("  LLaMA3.1-70B : 0.58  (migliore)")
    print("  LLaMA3.1-8B  : 0.43")
    print("  Anita-8B     : 0.43")
    print("  Minerva-7B   : 0.27  (peggiore)")
    print(f"  Caso casuale : 0.25")

    return riepilogo


def controlla_modelli_disponibili(modelli):
    """Verifica che Ollama sia in esecuzione e che i modelli richiesti siano
    già stati scaricati."""
    try:
        disponibili = [m["model"] for m in ollama.list()["models"]]
    except Exception as e:
        print("ERRORE: non riesco a contattare Ollama.")
        print("Assicurati che l'app/servizio Ollama sia avviato, poi riprova.")
        print(f"Dettaglio errore: {e}")
        return False

    mancanti = [m for m in modelli if not any(m in d for d in disponibili)]
    if mancanti:
        print("ERRORE: i seguenti modelli non risultano scaricati:")
        for m in mancanti:
            print(f"  - {m}  ->  scaricalo con: ollama pull {m}")
        return False

    print("Tutti i modelli richiesti sono disponibili:", modelli)
    return True


def parse_argomenti():
    parser = argparse.ArgumentParser(
        description="Valutazione MACID con modelli locali via Ollama. "
                     "Senza argomenti, esegue tutti i modelli del catalogo "
                     "con le 4 strategie base."
    )
    parser.add_argument(
        "--modelli", nargs="+", default=None, metavar="NOME_MODELLO",
        help=(
            "Modelli da eseguire in questa run, separati da spazio. "
        )
    )
    parser.add_argument(
        "--strategie", nargs="+", default=None, metavar="NOME_STRATEGIA",
        choices=list(STRATEGIE_DISPONIBILI.keys()),
        help=(
            "Strategie da eseguire in questa run, separate da spazio. "
            f"Disponibili: {', '.join(STRATEGIE_DISPONIBILI.keys())}. "
            "Default: solo le 4 strategie base (esclude few-shot+cot)."
        )
    )
    parser.add_argument(
        "--lista-modelli", action="store_true",
        help="Stampa il catalogo dei modelli noti e quelli scaricati su Ollama, poi esce."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argomenti()

    if args.lista_modelli:
        print("Modelli nel catalogo (MODELLI_DISPONIBILI):")
        for m in MODELLI_DISPONIBILI:
            print(f"  - {m}")
        print()
        try:
            scaricati = [m["model"] for m in ollama.list()["models"]]
            print("Modelli attualmente scaricati su Ollama:")
            for m in scaricati:
                print(f"  - {m}")
        except Exception as e:
            print(f"Impossibile contattare Ollama: {e}")
        raise SystemExit(0)


    modelli_run = args.modelli if args.modelli is not None else MODELLI_DISPONIBILI

    if args.strategie is not None:
        strategie_run = {nome: STRATEGIE_DISPONIBILI[nome] for nome in args.strategie}
    else:
        strategie_run = STRATEGIE_BASE

    print(f"Modelli da testare in questa run: {modelli_run}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Strategie da testare in questa run: {list(strategie_run.keys())}")
    print()

    if not controlla_modelli_disponibili(modelli_run):
        raise SystemExit(1)

    try:
        df_risultati = valuta(modelli_run, strategie_run)
    except KeyboardInterrupt:
        print("\n\nInterrotto manualmente (Ctrl+C).")
        print(f"Nessun problema: tutti i risultati ottenuti finora sono già "
              f"salvati in {OUTPUT_PATH}.")
        print("Rilancia lo script per riprendere da dove si è fermato.")
        raise SystemExit(0)

    riepilogo = riassunto(df_risultati)

    riepilogo.to_csv("riepilogo_accuratezza.csv")
    print(f"\nRisultati completi salvati in: {OUTPUT_PATH}")
    print(f"Riepilogo salvato in: riepilogo_accuratezza.csv")
