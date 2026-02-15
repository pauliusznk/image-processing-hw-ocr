# IMAGE-PROCESSING-HW-OCR
# image-processing-hw-ocr

OCR pagrindu veikiantis dokumentų analizės projektas, skirtas:
- teksto išgavimui iš vaizdų,
- dokumento tipo klasifikavimui,
- struktūrizuotų laukų ištraukimui,
- rezultatų įrašymui į JSON/CSV bei metrikų generavimui.

---

## 1. Apžvalga

Šis projektas įgyvendina dokumentų apdorojimo pipeline, skirtą skenuotiems ar fotografuotiems dokumentams:
- **email**
- **invoice**
- **news**
- **receipts**

Pipeline tikslas:
1. nuskaityti tekstą iš dokumento vaizdo (OCR),
2. nustatyti dokumento klasę,
3. ištraukti svarbiausius laukus (pvz., invoice number, total amount, subject, date ir pan.),
4. išsaugoti rezultatą struktūrizuotai (`results/json/*.json`),
5. (pasirinktinai) sukurti anotuotą vaizdą su OCR dėžutėmis,
6. batch režime įvertinti kokybę (accuracy, confusion matrix, predictions CSV).

---

## 2. Metodai

### 2.1 OCR sluoksnis
OCR realizuotas su **EasyOCR** (`src/ocr.py`):
- grąžina pilną tekstą (`ocr_text`),
- grąžina aptikimo dėžutes (`boxes`), kurios naudojamos anotacijoms.

### 2.2 Dokumento klasifikacija
Klasifikacija (`src/classifier.py`) veikia dviem režimais:
1. **LLM režimas** (per Ollama, pvz. `phi3`) – kai `use_llm=True`;
2. **Rule-based fallback** – jei LLM nepasiekiamas arba atsakymas netinkamas.

Naudojamos klasės:
- `email`, `invoice`, `news`, `receipts`.

### 2.3 Laukų ištraukimas (Field Extraction)
`src/extractor.py`:
- dokumento tipui pritaikyti promptai,
- „focused text“ strategija mažesniems lokaliems modeliams,
- regex fallback pagal dokumento tipą.

Pavyzdiniai laukai:
- **email**: from, to, cc, subject, date
- **invoice**: invoice_number, date, seller, buyer, total_amount, vat_amount, currency
- **receipts**: store, date, total, currency, payment_method
- **news**: title, author, content

### 2.4 Batch evaluacija
`src/eval.py` generuoja:
- `predictions.csv`
- `summary.txt`
- `confusion_matrix.png`

Papildomai skaičiuojama:
- accuracy,
- bendras ir vidutinis apdorojimo laikas.

---

## 3. Sąranka

### 3.1 Reikalavimai
- Python 3.10+ (rekomenduojama 3.11/3.12)
- `pip`
- Linux/macOS/Windows aplinka su veikiančiu OpenCV/EasyOCR
- (pasirinktinai) **Ollama** + modelis `phi3`, jei norima LLM režimo

### 3.2 Diegimas

Sukurti virtualią aplinką ir įdiegti priklausomybes:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Jei naudojamas LLM (Ollama):

1) Įdiegti Ollama: https://ollama.com  
2) Parsisiųsti modelį:

```bash
ollama pull phi3
```

Patikrinti, ar modelis atsisiųstas:

```bash
ollama list
```

> Pastaba: LLM režimas naudoja lokalią Ollama API (`http://localhost:11434`). Jei Ollama neveikia, rekomenduojama naudoti `--no-llm`.

---

## 4. Vykdymo instrukcijos

### 4.1 Vieno failo apdorojimas

Bazinė komanda:

```bash
python main.py path/to/image.jpg
```

Naudingi argumentai:
- `--outdir results` – kur saugoti output
- `--model phi3` – Ollama modelis
- `--no-llm` – išjungti LLM (naudoti taisykles)
- `--lang en` – OCR kalba (pvz. `en`, `lt`, `en+lt`)
- `--annotate` – išsaugoti OCR dėžučių anotuotą vaizdą

Pavyzdys:

```bash
python main.py dataset/invoice/batch1-0002.jpg --annotate --model phi3 --lang en
```

### 4.2 Batch režimas

Apdoroti visą dataset:

```bash
python main.py --batch dataset
```

Su limitu (testavimui):

```bash
python main.py --batch dataset --limit 50
```

Be LLM:

```bash
python main.py --batch dataset --no-llm
```

---

## 5. Rezultatai ir output struktūra

Po paleidimo rezultatai saugomi `results/`:

- `results/json/*.json` – struktūrizuoti rezultatai kiekvienam dokumentui
- `results/annotated_images/*_boxes.jpg` – vaizdai su OCR dėžutėmis
- `results/metrics/predictions.csv` – batch klasifikacijos rezultatai
- `results/metrics/summary.txt` – accuracy + laiko statistika
- `results/metrics/confusion_matrix.png` – klaidų matrica

JSON įrašai taip pat turi `meta` informaciją:
- klasifikacijos metodą (`llm`, `rules`, `rules_fallback`),
- confidence,
- OCR engine,
- apdorojimo laiką.

---

## 6. Žinomos problemos

1. Praktinis apribojimas: šiuo metu sprendimas stabiliai veikia tik su `.jpg` failais.  
   Nors kode yra platesnis formatų sąrašas, realiame naudojime rekomenduojama naudoti JPG įvestis.
3. `--lang` argumento aprašyme anksčiau buvo minimas Tesseract stilius (`eng`), nors pipeline naudoja EasyOCR (`en`, `lt`, `en+lt`).
4. LLM režimas priklauso nuo Ollama prieinamumo; jam neveikiant pereinama į fallback taisykles.
5. Batch režimas su LLM gali būti lėtas (ypač CPU aplinkoje).
6. `invoice` ir `receipts` klasės gali persidengti triukšminguose ar trumpuose dokumentuose.
7. OCR kokybė smarkiai priklauso nuo vaizdo kokybės (pasukimas, blur, mažas kontrastas).

---
