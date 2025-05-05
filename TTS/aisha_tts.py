import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('AISHA_API_KEY')

def tts(api_key: str):
    url = "https://back.aisha.group/api/v1/tts/post/"
    headers = {
        "x-api-key": api_key,
        "X-Channels": "stereo",
        "X-Quality": "64k",
        "X-Rate": "16000",
        "X-Format": "mp3"
    }
    files = {
        "transcript": (None, "Assalomu Alaikum"),
        "language": (None, "uz"),
        "model": (None, "gulnoza")
    }

    response = requests.post(url, headers=headers, files=files)
    print(response)

    if response.status_code == 200:
        audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        play(audio)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    tts(api_key)