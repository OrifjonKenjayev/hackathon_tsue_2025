from google import genai
from google.genai import types
import base64
import os
import json
import time
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

# Ovoz yozish (.wav formatda)
def record_audio(duration=7, fs=44100):
    print("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    
    wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(wav_temp.name, fs, audio)

    return wav_temp.name

# Gemini transkripsiya
def generate(audiofile):
    client = genai.Client(api_key=gemini_api_key)

    transcript = ""

    with open(audiofile, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # WAV format uchun mime_type
    audio_part = types.Part.from_bytes(data=base64.b64decode(audio_base64), mime_type="audio/wav")

    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                audio_part,
                types.Part.from_text(text=".")
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        system_instruction=[types.Part.from_text(text="transcribe given audio to uzbek language")]
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        transcript += parse_transcription(chunk)
    return transcript

def parse_transcription(response):
    transcription = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            transcription += part.text + " "
    return transcription.strip()

# Ishga tushirish
if __name__ == "__main__":
    wav_file = record_audio(duration=7)
    result = generate(wav_file)
    print("\n🔊 Transkriptsiya:\n", result)