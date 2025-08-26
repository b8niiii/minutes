Perfetto ✅ Ecco il file **README.md** già pronto, formattato in Markdown e utilizzabile direttamente nel tuo repository.

---

````markdown
# 📑 Minutes Pipeline

Questo progetto permette di generare automaticamente **trascrizioni, minute e azioni** da un file video/meeting.  
Il pipeline gestisce l’estrazione audio, la trascrizione (con diarizzazione dei parlanti) e la creazione di minute in più formati.

---

## ⚙️ 1) Preparazione ambiente

### 1.1 Installa **ffmpeg**

- **macOS (Homebrew):**
  ```bash
  brew install ffmpeg
  ffmpeg -version
````

* **Ubuntu/Debian:**

  ```bash
  sudo apt update
  sudo apt install -y ffmpeg
  ffmpeg -version
  ```
* **Windows (PowerShell con winget):**

  ```powershell
  winget install Gyan.FFmpeg
  ffmpeg -version
  ```

  > Se non funziona, scarica la versione full build da [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) e aggiungi la cartella `bin` al **PATH**.

---

### 1.2 Crea un **virtualenv** e installa le dipendenze

Apri **VS Code** nella cartella del repo (`minutes`) e lancia il **Terminal**.

* **macOS / Linux:**

  ```bash
  cd minutes
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

* **Windows (PowerShell):**

  ```powershell
  cd minutes
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

> In VS Code seleziona l’interprete `.venv` (in basso a destra), così i comandi useranno l’ambiente corretto.

---

### 1.3 Imposta la **API Key OpenAI**

Puoi esportare la chiave come variabile d’ambiente:

* **macOS / Linux:**

  ```bash
  export OPENAI_API_KEY="sk-...la_tua_chiave..."
  ```
* **Windows (PowerShell):**

  ```powershell
  $env:OPENAI_API_KEY="sk-...la_tua_chiave..."
  ```

Oppure passala direttamente al comando con `--openai-api-key` (meno consigliato).

---

## ▶️ 2) Esecuzione del pipeline

Esempio base:

```bash
python pipeline.py \
  --video "meeting.mp4" \
  --chunk-seconds 180 \
  --max-speakers 6 \
  --attendees "Alessandro,Marta,Vasco,Marco"
```

### Opzioni utili

* Usa audio già estratto:

  ```bash
  python pipeline.py --audio "meeting.wav"
  ```
* Forza la lingua:

  ```bash
  python pipeline.py --video "meeting.mp4" --language "it"
  ```
* Specifica modello e cartella output:

  ```bash
  python pipeline.py --video "meeting.mp4" --model "whisper-1" --output-dir "./out"
  ```

---

## 📂 3) Risultati

Dopo l’esecuzione troverai i file nella cartella `out/`:

* `meeting.wav` → audio estratto
* `transcript_diarized.json` → trascrizione con diarizzazione
* `chunks/chunk_XXX.json` → trascrizioni a blocchi
* `minutes_merged.json` → merge dei chunk
* `minutes.md` → minute in Markdown
* `minutes.docx` → minute in Word
* `actions.csv` → azioni/decisioni estratte

---

## ✅ 4) Mini-check rapido

1. `ffmpeg -version` → funziona?
2. `python --version` o `py --version` → deve mostrare Python ≥ 3.9
3. Prompt con `(.venv)` → significa che il virtualenv è attivo
4. Variabile `OPENAI_API_KEY` impostata correttamente

---

## 🛠️ 5) Problemi comuni

* **`python3: not found` su Windows** → usa `py` o `python` al posto di `python3`.
* **`ffmpeg` non trovato** → reinstallalo e aggiungi al PATH.
* **`ModuleNotFoundError`** → assicurati di aver attivato `.venv` e installato i requirements.
* **Errore OpenAI API key** → ricontrolla spazi/apici nella chiave.
* **Percorsi con spazi** → usa sempre virgolette intorno ai file.

---

```

---

Vuoi che ti prepari anche un **badge di stato** (tipo `![Python](https://img.shields.io/badge/python-3.9+-blue.svg)`) e una sezione “Contributi” per renderlo più professionale su GitHub?
```
