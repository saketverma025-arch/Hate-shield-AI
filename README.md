# HATE-SHIELD AI web app

A Streamlit interface for the supplied Class XII hate-speech classification project.

## Files required in the same folder
- `app.py` — web interface
- `twitter_data.csv` — supplied dataset, with `tweet` and `class` columns
- `hate_speech_model.joblib` — optional saved model from your existing Python script

## Run on a computer
1. Install Python.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Open the local URL Streamlit prints (usually `http://localhost:8501`).

If `hate_speech_model.joblib` is present, the app loads it. If it is missing, the app trains a model from the CSV the first time and saves it. The first run can take a little while.

## Important
The classifier is an educational prototype. Its predictions and confidence scores can be wrong and should not be treated as definitive judgements about a person or message.
