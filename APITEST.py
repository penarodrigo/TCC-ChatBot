from dotenv import load_dotenv, find_dotenv
import os
import requests

# Encontrar o arquivo .env
dotenv_path = find_dotenv()
print(f"Arquivo .env encontrado: {dotenv_path}")

# Carregar o arquivo .env
load_dotenv(dotenv_path)

# Verificar o valor da chave GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY carregada (limpa): {repr(api_key)}")

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

headers = {
    "Content-Type": "application/json"
}

data = {
    "prompt": "Explain how AI works in a few words"
}

response = requests.post(f"{url}?key={api_key}", headers=headers, json=data)

if response.status_code == 200:
    print("Resposta da API:", response.json())
else:
    print(f"Erro {response.status_code}: {response.text}")