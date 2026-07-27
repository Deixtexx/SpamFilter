import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re
from pathlib import Path

def main():
    proj_root = Path(__file__).parent.parent
    data_path = proj_root / 'data' / 'spam.csv'

    with open(data_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    labels = []
    texts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        first_comma = line.find(',')
        if first_comma == -1:
            continue

        label_part = line[:first_comma].strip().strip('"')
        text_part = line[first_comma + 1:].strip()

        if ',,,' in text_part:
            text_part = text_part.split(',,,')[0].strip()

        elif ';' in text_part:
            text_part = text_part.split(';')[0].strip()

        text_part = text_part.strip('"')

        if label_part.lower() == 'spam':
            labels.append(1)
        elif label_part.lower() == 'ham':
            labels.append(0)
        else:
            continue

        texts.append(text_part)

    df = pd.DataFrame({
        'label': labels,
        'text': texts
    })

    X = df['text']
    y = df['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(C=1.0, max_iter=1000))
    ])

    pipeline.fit(X, y)
    print(f"Precision: {pipeline.score(X, y):.4f}")

    joblib.dump(pipeline, proj_root / 'models/logistic_model.joblib')
    print("the model has been saved")

if __name__ == '__main__':
    main()