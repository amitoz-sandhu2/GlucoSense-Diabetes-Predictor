# GlucoSense - Diabetes Predictor

A Streamlit web app that predicts whether a person may have diabetes from a few
health details, and then recommends foods and habits that help.

## Run
    pip install -r requirements.txt
    streamlit run app.py

The model is already trained and saved in `model/`. To retrain it:

    python train.py

## Project structure
    app.py                 Streamlit interface (welcome page, form, result, diet guide)
    diet.py                Diet, food and lifestyle recommendations
    train.py               Data cleaning and model training
    scripts/make_assets.py Generates the illustrations in assets/
    assets/                hero.png and plate.png used by the interface
    data/                  Pima Indians Diabetes dataset (raw and cleaned)
    model/                 Saved model (model.joblib)

Educational tool only. It is not a medical diagnosis.
