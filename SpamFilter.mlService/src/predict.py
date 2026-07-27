import re
import joblib
from pathlib import Path

proj_root = Path(__file__).parent.parent

class SpamFilter:
    def __init__(self, model_path=proj_root / 'models/logistic_model.joblib'):
        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        prob = self.pipeline.predict_proba([clean_text])[0][1]
        is_spam = prob > 0.5

        return {
            'is_spam': is_spam,
            'confidence': prob,
        }