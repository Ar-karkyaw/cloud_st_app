#import base64, io, json, time
#import requests
#from pydub import AudioSegment
#from google.oauth2 import service_account
#from google.auth.transport.requests import Request

#_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
#_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"

#def _get_access_token(sa_info: dict) -> str:
    #creds = service_account.Credentials.from_service_account_info(sa_info, scopes=_SCOPES)
    #creds.refresh(Request())
    #return creds.token

#def encode_audio(audio_bytes: bytes) -> str:
    # WAV をモノラル化（サンプリングレートは維持。WAV/FLACはヘッダで自動判定してくれる）
    #seg = AudioSegment.from_wav(io.BytesIO(audio_bytes)).set_channels(1)
    #buf = io.BytesIO()
    #seg.export(buf, format="wav")
    #return base64.b64encode(buf.getvalue()).decode("utf-8")

#def get_response(encoded_audio: str, *, sa_info: dict = None, api_key: str = None):
    # WAV の場合は encoding / sampleRate を送らない（自動判定）← 公式仕様
    #payload = {
        #"config": {
            #"languageCode": "ja-JP",
            #"enableWordTimeOffsets": True,
            #"audioChannelCount": 1
        #},
        #"audio": {"content": encoded_audio}
    #}

    #headers = {"Content-Type": "application/json; charset=utf-8"}
    #url = _STT_ENDPOINT

    #if sa_info:
        #token = _get_access_token(sa_info)
        #headers["Authorization"] = f"Bearer {token}"
    #elif api_key:
        # APIキーで呼ぶ場合（動作は環境依存になりがち。推奨はサービスアカウント）
       # url = f"{url}?key={api_key}"
    #else:
     #   raise RuntimeError("No credentials provided. Provide service account info or API key.")

    #resp = requests.post(url, headers=headers, json=payload, timeout=60)
    #return resp

#def extract_words(data: dict):
 #   alts = data["results"][0]["alternatives"][0]
  #  words = alts.get("words", [])
   # return [
    #    {
     #       "word": w["word"],
      #      "startTime": float(w["startTime"].rstrip("s")),
       #     "endTime": float(w["endTime"].rstrip("s")),
        #}
        #for w in words
    #]


# util.py
# Robust audio encoding (any input -> WAV mono 16k LINEAR16) + Google STT call + word extraction
# Dependencies: av, soundfile, numpy, google-auth, requests

import io
import base64
import json
import numpy as np
import requests
import av
import soundfile as sf
from google.oauth2 import service_account
from google.auth.transport.requests import Request

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"


def _get_access_token(sa_info: dict) -> str:
    """Exchange a GCP service account JSON (dict) for an OAuth access token."""
    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=_SCOPES)
    creds.refresh(Request())
    return creds.token


def _decode_to_mono16k(audio_bytes: bytes) -> bytes:
    """
    Decode arbitrary container/codec (webm/opus, m4a/aac, mp3, wav, etc.) using PyAV,
    resample to mono 16 kHz PCM16, and return WAV bytes.
    Handles the case where resampler returns a list of frames.
    """
    frames_arrays = []

    with av.open(io.BytesIO(audio_bytes)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("No audio stream found in the uploaded file.")

        # Resample to 16-bit PCM, mono, 16 kHz
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)

        for packet in container.demux(stream):
            for frame in packet.decode():
                res = resampler.resample(frame)  # can be None, a Frame, or a list[Frame]
                if res is None:
                    continue
                res_frames = res if isinstance(res, list) else [res]
                for rf in res_frames:
                    arr = rf.to_ndarray()  # int16; shape (channels, samples) or (samples,)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)  # (1, samples)
                    frames_arrays.append(arr)

    if not frames_arrays:
        raise RuntimeError("No audio frames decoded after resampling (file may be empty or unsupported).")

    # Concatenate along time axis → (samples,)
    i16 = np.concatenate(frames_arrays, axis=1).squeeze()  # dtype=int16
    # Write WAV (PCM_16). soundfile can take int16 directly.
    with io.BytesIO() as buf:
        sf.write(buf, i16, 16000, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def encode_audio(audio_bytes: bytes) -> str:
    """
    Convert whatever the user uploaded/recorded into WAV (mono, 16k, LINEAR16) bytes,
    then return base64-encoded string ready for Google STT.
    """
    wav = _decode_to_mono16k(audio_bytes)
    return base64.b64encode(wav).decode("utf-8")


def get_response(encoded_audio: str, *, sa_info: dict | None = None, api_key: str | None = None):
    """
    Call Google Speech-to-Text v1.
    Prefer service account (OAuth) auth; API key also supported.
    """
    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": "ja-JP",
            "enableWordTimeOffsets": True,
            "audioChannelCount": 1,
        },
        "audio": {"content": encoded_audio},
    }

    headers = {"Content-Type": "application/json; charset=utf-8"}
    url = _STT_ENDPOINT

    if sa_info:
        token = _get_access_token(sa_info)
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        url = f"{url}?key={api_key}"
    else:
        raise RuntimeError("No credentials provided. Provide service account info (sa_info) or api_key.")

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    return resp


def _to_seconds(t: str | None) -> float:
    if not t:
        return 0.0
    t = t.rstrip("s")
    try:
        return float(t)
    except Exception:
        return 0.0


def extract_words(data: dict):
    """
    Convert Google STT response into a list of {word, startTime, endTime}.
    Returns [] if nothing found.
    """
    out = []
    for result in data.get("results", []):
        for alt in result.get("alternatives", []):
            for w in alt.get("words", []):
                out.append(
                    {
                        "word": w.get("word", ""),
                        "startTime": _to_seconds(w.get("startTime")),
                        "endTime": _to_seconds(w.get("endTime")),
                    }
                )
    return out
