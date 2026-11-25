# FIFA Player Intelligence Chatbot

A local, dataset-backed football assistant for exploring FIFA player profiles and match-performance statistics. The project combines a Streamlit chat interface, direct CSV lookup logic, and Ollama's local `llama3.2:3b` model. It also contains a FLAN-T5 fine-tuning pipeline and saved fine-tuned model artifacts for experimentation.

> **Dataset status:** The repository's player and match data is provided as a project/synthetic dataset. Its provenance should be independently verified before describing it as official FIFA data.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Fine-Tuned FLAN-T5 Artifacts](#fine-tuned-flan-t5-artifacts)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Setting Up Ollama](#setting-up-ollama)
- [Example Questions](#example-questions)
- [Training the SLM](#training-the-slm)
- [Project Limitations](#project-limitations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## Features

- Interactive Streamlit chat interface
- Direct lookup of player age, nationality, team, position, height, weight, foot, club, and market value
- Match-specific lookup for statistics such as goals, assists, minutes, shots, passes, tackles, interceptions, cards, saves, xG, and xA
- Player-name recognition from natural-language questions
- Local LLM fallback for general football concepts through Ollama
- Input restriction that rejects clearly non-football questions
- Chat history within the Streamlit session
- A FLAN-T5-Small fine-tuning notebook with training, validation, testing, tokenization, and model export cells
- Saved T5 model, tokenizer, and training configuration files

## How It Works

The application uses a hybrid question-answering flow:

1. **Load data:** Streamlit loads `dataset/raw/fifa_world_cup_2026_player_performance.csv` when the app starts.
2. **Validate scope:** Short football-keyword matching determines whether a question appears football-related.
3. **Identify the player:** The question engine searches for player names present in the dataset.
4. **Answer deterministically:** If the question asks for a profile field or match statistic in the dataset, the app returns the exact CSV value.
5. **Use Ollama for general knowledge:** If no exact dataset answer is matched, the app sends the question to the locally running `llama3.2:3b` model.
6. **Preserve session chat:** User and assistant messages are stored in Streamlit session state.

For exact player statistics, the application does not rely on model memory. It first looks up the value in the CSV file. If no exact match is found, it may fall back to Ollama, whose system prompt explicitly instructs it not to invent player statistics.

## Project Architecture

```text
User question
     |
     v
Streamlit chat interface (app/app.py)
     |
     +--> Football scope check
     |
     +--> Player-name and match-ID detection
     |        |
     |        +--> Exact dataset lookup
     |
     +--> Ollama fallback
              |
              +--> llama3.2:3b
```

## Repository Structure

```text
final slm project/
├── app/
│   ├── app.py                         # Main Streamlit application
│   ├── app.py.txt                     # Placeholder/empty text file
│   └── FIFA_PLAYER_DATABASE.csv       # Separate app database copy
├── dataset/
│   ├── FIFA_CLEANED_DATASET.csv       # Cleaned full 75-column dataset
│   ├── FIFA_PLAYER_PROFILE_QA.csv     # Profile QA records
│   ├── raw/
│   │   └── fifa_world_cup_2026_player_performance.csv
│   ├── training/
│   │   ├── FIFA_TRAIN.csv
│   │   └── FIFA_TRAIN_QA.csv
│   ├── validation/
│   │   ├── FIFA_VALIDATION.csv
│   │   └── FIFA_VALIDATION_QA.csv
│   └── testing/
│       ├── FIFA_TEST.csv
│       └── FIFA_TEST_QA.csv
├── model/
│   ├── final_model/                   # Model metadata placeholder README
│   └── tokenizer/                     # Exported tokenizer JSON and config
├── modal/
│   ├── final_model/                   # Saved fine-tuned model files
│   ├── FIFA_SLM_FINAL/                # Saved model copy
│   └── tokenizer/                     # Additional tokenizer copy
├── notebook/
│   ├── FIFA_SLM_Training1.ipynb       # Dataset preparation and model training notebook
│   └── FIFA_FLAN_T5_Training_files/   # Notebook HTML/export support files
├── ollama/
│   ├── ollama_commands.txt            # Ollama setup commands
│   ├── README.md                      # Ollama notes
│   └── supporting report files
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The primary raw dataset contains **54,600 player-match records**, **75 columns**, and the following main categories:

- **Identity:** player ID, name, age, nationality, team, jersey number, position
- **Profile:** height, weight, preferred foot, club, market value
- **Match context:** match ID/date, stadium, city, opponent, stage, result
- **Team performance:** goals scored and conceded
- **In-game performance:** minutes, goals, assists, shots, passing, dribbles, defensive actions
- **Advanced metrics:** expected goals, expected assists, player rating, stamina, consistency, and creativity
- **Tournament aggregates:** total goals, assists, minutes, player-of-match awards, tournament rating

### Dataset and QA split sizes

| File or split | Rows | Columns |
|---|---:|---:|
| Raw/cleaned dataset | 54,600 | 75 |
| Training records | 38,169 | 75 |
| Validation records | 8,134 | 75 |
| Testing records | 8,297 | 75 |
| Profile QA | 11,221 | 2 |
| Training QA | 236,871 | 2 |
| Validation QA | 50,487 | 2 |
| Testing QA | 51,474 | 2 |

The QA files contain `input_text` and `target_text` columns. They include generated questions for fields such as age, nationality, team, position, height, weight, preferred foot, club, and market value, as well as match-performance questions.

## Fine-Tuned FLAN-T5 Artifacts

The notebook is designed to fine-tune `google/flan-t5-small`, a small encoder-decoder language model.

The training pipeline:

- Loads the raw CSV and inspects the schema
- Cleans/removes unused or missing information
- Splits the source data into training, validation, and testing sets
- Generates natural-language question-answer records
- Loads FLAN-T5-Small and its tokenizer
- Converts pandas records to Hugging Face `Dataset` objects
- Tokenizes inputs to a maximum length of 128 tokens
- Tokenizes targets to a maximum length of 64 tokens
- Uses `Seq2SeqTrainer` and `DataCollatorForSeq2Seq`
- Trains for three epochs with a learning rate of `3e-4`
- Uses batch size 4 for training and evaluation
- Saves the final model and tokenizer

The final model is a T5 conditional-generation model with a 512-token model context and approximately 300 MB in the included weight file. The repository also includes two local copies under `modal/`:

- `modal/FIFA_SLM_FINAL/`
- `modal/final_model/`

## Requirements

### For the Streamlit app

- Windows 10/11 or another environment that can run Python and Ollama
- Python 3.10 or later recommended
- Ollama
- Internet access only if Ollama or Hugging Face models must be downloaded

Python dependencies declared in `requirements.txt`:

```text
streamlit
pandas
numpy
transformers
torch
ollama
```

### For notebook training

The notebook additionally uses packages not explicitly listed in `requirements.txt`, including:

- `datasets`
- `accelerate`
- `sentencepiece`
- `evaluate`
- `scikit-learn`

GPU acceleration is optional, but a CUDA-compatible GPU will substantially reduce training time.

## Installation

Open PowerShell in the project directory.

```powershell
cd "C:\Users\YourName\OneDrive\Desktop\final slm project"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install datasets accelerate sentencepiece evaluate scikit-learn
```

If PowerShell blocks virtual-environment activation, activate it with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Running the Streamlit Application

1. Install and start Ollama.
2. Pull the required base model:

   ```powershell
   ollama pull llama3.2:3b
   ```

3. Start the Streamlit app:

   ```powershell
   cd "C:\Users\YourName\OneDrive\Desktop\final slm project"
   streamlit run app/app.py
   ```

4. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The app loads the raw dataset relative to `app/app.py`, so it can be launched from other working directories as long as the repository structure remains unchanged.

## Setting Up Ollama

The app directly calls Ollama using:

```python
ollama.chat(model="llama3.2:3b", messages=[...])
```

Useful commands:

```powershell
ollama --version
ollama pull llama3.2:3b
ollama list
ollama run llama3.2:3b
ollama ps
```

The repository's `ollama/README.md` describes a custom model named `fifa-player-assistant`, but no `Modelfile` is currently present in the repository. The Streamlit app therefore uses the standard `llama3.2:3b` model directly.

## Example Questions

### Exact profile questions

```text
What is the age of Rodri Fati?
What nationality is Lyle Makgopa?
Which club does Pedro Cubarsi play for?
What is Ferran Cubarsi's position?
What is the height of Dani Gavi?
What is the preferred foot of Ansu Fati?
```

### Match-statistic questions

Use a match ID in the format `M00001`:

```text
How many goals did Gavi Ramos score in M00001?
What were the assists for Pedro Cubarsi in M00001?
How many minutes did Rodri Fati play in M00001?
What was Dani Gavi's pass accuracy in M00001?
```

### General football concepts

```text
Explain the difference between goals and expected goals.
What is an assist?
What does pass accuracy measure?
Explain the difference between a striker and a winger.
```

Questions that do not contain recognized football keywords are rejected before being sent to Ollama.

## Training the SLM

Open `notebook/FIFA_SLM_Training1.ipynb` in JupyterLab or Google Colab.

Before training:

1. Ensure `notebook/FIFA_SLM_Training1.ipynb` has access to the repository's `dataset/` directory.
2. Install the notebook dependencies.
3. Confirm that Hugging Face model downloads are permitted.
4. Use a GPU runtime when possible.

A minimal notebook setup is:

```powershell
python -m pip install jupyterlab
jupyter lab
```

Run the cells sequentially. The notebook creates the QA splits, downloads `google/flan-t5-small`, trains the model, saves it to `model/final_model/` or the configured final-model directory, and can generate sample answers.

The trained FLAN-T5 model is **not currently loaded by `app/app.py`**. The running Streamlit application uses direct CSV lookup followed by Ollama. The FLAN-T5 artifacts are separate experimental/model-training outputs.

## Project Limitations

- The Streamlit app does not send the full CSV or matched record to Ollama; it only uses Ollama for questions not answered by direct lookup.
- General scope detection is keyword-based and may reject unusual football phrasing or accept ambiguous terms.
- Player identification is based on substring matching and may be ambiguous if names share text.
- Profile answers use the first matching player record.
- Match lookup requires the exact `M#####` match-ID format.
- The app supports profile fields and a defined set of match fields; unsupported questions fall back to Ollama.
- The Ollama system prompt reduces hallucination risk but cannot guarantee factual correctness for concepts outside the exact dataset.
- Dataset provenance and official/FIFA affiliation are not documented in sufficient detail to call the data authoritative.
- A custom Ollama `Modelfile` is mentioned in documentation but is not included in the repository.
- The fine-tuned FLAN-T5 weight file is ignored by some local Git workflows and is provided in the working tree under `modal/`.
- There is no automated test suite or CI configuration in the repository.

## Troubleshooting

### `Dataset loading failed`

Run Streamlit from the repository root and verify that this file exists:

```text
dataset/raw/fifa_world_cup_2026_player_performance.csv
```

### `Ollama connection failed`

Check that the Ollama service is running:

```powershell
ollama list
ollama ps
ollama run llama3.2:3b
```

Then pull the model if necessary:

```powershell
ollama pull llama3.2:3b
```

### Missing Python package

Install the package or restore all dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install ollama datasets accelerate sentencepiece evaluate scikit-learn
```

### Player information is unavailable

Check that:

- The player name is spelled exactly as it appears in the CSV
- The question contains a recognized field keyword
- Match questions include the player's name and a valid `M#####` ID

### Multiple player matches

If the app detects more than one player with an ambiguous name match, use the player's full name.

## Future Improvements

- Integrate the fine-tuned FLAN-T5 model directly into the Streamlit interface
- Add a single `Modelfile` to package the Ollama setup reproducibly
- Replace substring scope detection with intent classification or named-entity extraction
- Improve name matching with fuzzy matching and ID-aware disambiguation
- Add exact match data to Ollama prompts for grounded general responses
- Add unit tests for lookup functions and CSV validation
- Add model and answer evaluation metrics such as exact match, ROUGE, and BLEU
- Add deployment configuration for Streamlit Cloud or Docker
- Document dataset provenance and data-generation methodology
- Add an explicit license and contribution guidelines
