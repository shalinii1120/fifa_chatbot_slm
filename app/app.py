
import streamlit as st
import pandas as pd
import ollama
import re
from pathlib import Path

# PROJECT PATHS
ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT / "dataset" / "raw" /
    "fifa_world_cup_2026_player_performance.csv"
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="FIFA Intelligence Chatbot",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA Player Intelligence Chatbot")
st.caption("Powered by FIFA Dataset and Ollama Llama 3.2 3B")

# LOAD DATASET
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

try:
    fifa = load_dataset()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# CHECK REQUIRED COLUMNS
required_columns = ["player_name", "player_id"]

missing_columns = [
    col for col in required_columns
    if col not in fifa.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

# SIDEBAR
st.sidebar.title("FIFA Assistant")
st.sidebar.success("Dataset connected")
st.sidebar.write(f"Total records: {len(fifa):,}")
st.sidebar.write(
    f"Unique players: {fifa['player_id'].nunique():,}"
)
st.sidebar.write("Ollama model: llama3.2:3b")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# FOOTBALL KEYWORDS
football_keywords = [
    "football", "soccer", "fifa", "world cup",
    "player", "team", "club", "match", "goal",
    "assist", "striker", "defender", "midfielder",
    "goalkeeper", "offside", "penalty", "corner",
    "tackle", "dribble", "passing", "league",
    "coach", "formation", "tournament",
    "footballer", "red card", "yellow card",
    "shot", "save", "nationality", "position",
    "statistics", "stats", "performance", "age",
    "height", "weight", "minutes", "market value",
    "xg", "xa", "interception", "pass accuracy"
]

def is_football_question(question):
    q = question.lower()
    return any(word in q for word in football_keywords)

# PLAYER IDENTIFICATION
def identify_player(question):
    q = question.lower()

    names = fifa["player_name"].dropna().unique()

    matched_names = [
        name for name in names
        if str(name).lower() in q
    ]

    if not matched_names:
        return None

    # Prefer the longest matching full name
    matched_names.sort(
        key=lambda name: len(str(name)),
        reverse=True
    )

    if (
        len(matched_names) > 1
        and len(str(matched_names[0])) ==
        len(str(matched_names[1]))
    ):
        return "MULTIPLE"

    return matched_names[0]

# PLAYER PROFILE
def get_player_profile(player_name):

    records = fifa[
        fifa["player_name"].astype(str).str.lower()
        == str(player_name).lower()
    ]

    if records.empty:
        return None

    return records.iloc[0]

# FIFA QUESTION ENGINE
def fifa_question_engine(question):

    player_name = identify_player(question)

    if player_name is None:
        return None

    if player_name == "MULTIPLE":
        return "Multiple players found. Please specify the full name."

    q = question.lower()
    player = get_player_profile(player_name)

    if player is None:
        return None

    # PROFILE QUESTIONS
    profile_fields = {
        "age": ["age", "old"],
        "nationality": ["nationality", "country"],
        "team": ["national team", "represent"],
        "position": ["position", "play as"],
        "height_cm": ["height"],
        "weight_kg": ["weight"],
        "preferred_foot": ["preferred foot"],
        "club_name": ["club"],
        "market_value_eur": ["market value"]
    }

    for field, keywords in profile_fields.items():

        if any(word in q for word in keywords):

            if field in fifa.columns:

                value = player[field]

                if pd.notna(value):
                    return f"{player_name}: {value}"

                return f"{field.replace('_', ' ').title()} is unavailable."

    # MATCH-SPECIFIC QUESTIONS
    match_found = re.search(
        r"\bM\d+\b", question, re.IGNORECASE
    )

    if match_found:

        match_id = match_found.group(0).upper()

        records = fifa[
            (
                fifa["player_name"].astype(str).str.lower()
                == str(player_name).lower()
            )
            &
            (
                fifa["match_id"].astype(str).str.upper()
                == match_id
            )
        ]

        if records.empty:
            return "No record found for this player in that match."

        if len(records) > 1:
            return "Multiple records found for this player and match."

        record = records.iloc[0]

        stat_keywords = {
            "goals": ["goals", "scored"],
            "assists": ["assists"],
            "minutes_played": ["minutes"],
            "shots": ["shots"],
            "shots_on_target": ["shots on target"],
            "tackles": ["tackles"],
            "interceptions": ["interceptions"],
            "yellow_cards": ["yellow cards"],
            "red_cards": ["red cards"],
            "expected_goals_xg": ["xg", "expected goals"],
            "expected_assists_xa": ["xa", "expected assists"],
            "key_passes": ["key passes"],
            "successful_passes": ["successful passes"],
            "total_passes": ["total passes"],
            "pass_accuracy": ["pass accuracy"],
            "saves": ["saves"]
        }

        for field, keywords in stat_keywords.items():

            if any(word in q for word in keywords):

                if field not in record.index:
                    return f"{field} is unavailable in the dataset."

                value = record[field]

                if pd.isna(value):
                    return f"{field.replace('_', ' ')} is unavailable."

                return (
                    f"{player_name} recorded {value} "
                    f"{field.replace('_', ' ')} "
                    f"in match {match_id}."
                )

        # If the user asks generally about this match
        return (
            f"Found match {match_id} for {player_name}. "
            "Please specify which statistic you want to know."
        )

    return None

# OLLAMA RESPONSE
def ask_ollama(question):

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a football knowledge assistant. "
                    "Answer only football-related questions. "
                    "Explain football concepts clearly. "
                    "Never invent exact player statistics. "
                    "For exact dataset statistics, rely only on "
                    "information explicitly supplied to you."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY PREVIOUS MESSAGES
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT
question = st.chat_input(
    "Ask anything about FIFA or football..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your question..."):

            if not is_football_question(question):

                answer = (
                    "Sorry! I can only answer questions "
                    "related to football and FIFA."
                )

            else:

                answer = fifa_question_engine(question)

                if answer is None:

                    try:
                        answer = ask_ollama(question)

                    except Exception as e:

                        answer = (
                            "Ollama connection failed. "
                            "Please check whether Ollama is running "
                            "and llama3.2:3b is installed.\n\n"
                            f"Error: {e}"
                        )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

