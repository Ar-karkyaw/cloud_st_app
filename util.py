import base64, io, json, time
import requests
from pydub import AudioSegment
from google.oauth2 import service_account
from google.auth.transport.requests import Request

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"

def _get_access_token(sa_info: dict) -> str:
    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=_SCOPES)
    creds.refresh(Request())
    return creds.token

def encode_audio(audio_bytes: bytes) -> str:
    # WAV をモノラル化（サンプリングレートは維持。WAV/FLACはヘッダで自動判定してくれる）
    seg = AudioSegment.from_wav(io.BytesIO(audio_bytes)).set_channels(1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_response(encoded_audio: str, *, sa_info: dict = None, api_key: str = None):
    # WAV の場合は encoding / sampleRate を送らない（自動判定）← 公式仕様
    payload = {
        "config": {
            "languageCode": "ja-JP",
            "enableWordTimeOffsets": True,
            "audioChannelCount": 1
        },
        "audio": {"content": encoded_audio}
    }

    headers = {"Content-Type": "application/json; charset=utf-8"}
    url = _STT_ENDPOINT

    if sa_info:
        token = _get_access_token(sa_info)
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        # APIキーで呼ぶ場合（動作は環境依存になりがち。推奨はサービスアカウント）
        url = f"{url}?key={api_key}"
    else:
        raise RuntimeError("No credentials provided. Provide service account info or API key.")

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp

def extract_words(data: dict):
    alts = data["results"][0]["alternatives"][0]
    words = alts.get("words", [])
    return [
        {
            "word": w["word"],
            "startTime": float(w["startTime"].rstrip("s")),
            "endTime": float(w["endTime"].rstrip("s")),
        }
        for w in words
    ]
