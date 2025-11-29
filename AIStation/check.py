import requests

BASE_URL = "http://localhost:8000"

def add(org_id: str, text: str):
    r = requests.post(f"{BASE_URL}/add", json={
        "org_id": org_id,
        "text": text,
    })
    print(r.status_code, r.json())

if __name__ == "__main__":
    org = "my_university"

    add(org, "Столовая: работает с 9:00 до 18:00.")
    add(org, "Кабинет 204 — прием документов с 9:00 до 17:00.")
    add(org, "Кабинет 105 — выдача справок с 10:00 до 16:00.")