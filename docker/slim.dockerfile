FROM python:3.11.0



RUN pip install --upgrade pip && \
    pip install pandas scipy numpy matplotlib seaborn spacy scikit-learn tokenizers text-generation statsmodels jinja2 krippendorff pingouin simpledorff openpyxl deepsig streamlit && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg

WORKDIR /app