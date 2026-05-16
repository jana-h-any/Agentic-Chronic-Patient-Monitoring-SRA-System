import requests

def generate_clinical_note(top_features):
    prompt = f"The patient is at high risk of deterioration. Top contributing factors: {', '.join(top_features)}. Write a concise 2-3 sentence clinical note explaining this risk."
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "biomistral:7b",
            "prompt": prompt,
            "stream": False
        }, timeout=10)
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception("Ollama API error")
    except Exception as e:
        return f"[Fallback] Patient shows signs of clinical deterioration driven by anomalies in {', '.join(top_features)}. Please monitor vital signs closely and prepare for intervention."