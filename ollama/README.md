# FIFA Player Performance Assistant — Ollama Files

## Files

| File | Purpose |
|---|---|
| `ollama_commands.txt` | Commands to download, create, run, test, and manage the Ollama model |
| `Modelfile` | Ollama configuration for the custom assistant |
| `README.md` | Project documentation |

## Model

This setup uses **`llama3.2:3b`** as the Ollama base model and creates a custom model named **`fifa-player-assistant`**.

## Setup

1. Install and start Ollama.
2. Open Command Prompt or PowerShell.
3. Navigate to this folder (edit the path for your computer):

   ```cmd
   cd "C:\Users\YourName\OneDrive\Desktop\final slm project\ollama"
   ```

4. Download the base model:

   ```cmd
   ollama pull llama3.2:3b
   ```

5. Create the custom model:

   ```cmd
   ollama create fifa-player-assistant -f Modelfile
   ```

6. Run it:

   ```cmd
   ollama run fifa-player-assistant
   ```

## Important: Ollama model vs. fine-tuned FLAN-T5 model

The `Modelfile` customizes the behavior of the Ollama **Llama 3.2 3B** model. It does **not** import, convert, or contain the fine-tuned **FLAN-T5-Small** weights.

The FIFA dataset and player-statistics lookup logic must be connected through the Python/Streamlit application to answer exact dataset questions. A system prompt by itself does not give Ollama access to the CSV file.

## Dataset note

The project uses `fifa_world_cup_2026_player_performance.csv`. Refer to it as the project's dataset or simulated dataset unless its provenance has been independently verified. Do not call it official FIFA data without evidence.

## Test prompt

> Explain the difference between goals, assists, expected goals (xG), and expected assists (xA).

For exact player or match statistics, use the dataset-backed application rather than relying on the language model to guess.
