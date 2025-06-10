import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def ask_openai_raw(api_key: str, question: str, model: str = "gpt-4o") -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(url, headers=headers, json=json_data, verify=False)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# Usage
api_key = "ENTER API"
question = "how are you?"
answer = ask_openai_raw(api_key, question)
print("Answer:", answer)

