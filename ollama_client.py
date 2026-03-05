import requests


def ollama_generate(prompt: str, model: str = "mistral") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError("Ollama request failed")

    return response.json().get("response", "")