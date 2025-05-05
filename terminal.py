import pandas as pd
import joblib
import requests
import re
import os
import google.generativeai as genai
import base64
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import pygame
import time
from io import BytesIO
from together import Together

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
tts_api_key = os.getenv('TTS_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')

# Initialize Together client
client = Together(api_key=together_api_key)

# Initialize pygame for audio playback
pygame.mixer.init()

# Load bank information from text file
def load_bank_info(file_path='uploads/general_info.txt'):
    if not os.path.exists(file_path):
        return "Bank haqida ma'lumot fayli topilmadi."
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

bank_info = load_bank_info()

# Load the saved model and test data
model = joblib.load('uploads/linear_regression_model.pkl')
test_data = pd.read_csv('uploads/test_data2.csv')
features = ['Income', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married', 'Ethnicity', 'Balance']

# Function to predict credit limit for a given ID
def predict_limit_by_id(input_id):
    if input_id not in test_data['ID'].values:
        return f"ID {input_id} topilmadi."
    input_data = test_data[test_data['ID'] == input_id][features]
    predicted_limit = model.predict(input_data)[0]
    return predicted_limit

# Function to detect query type
def is_credit_query(message):
    credit_keywords = ['kredit', 'qarz', 'limit', 'pul olish', 'kredit olish', 'kredit limiti']
    return any(keyword.lower() in message.lower() for keyword in credit_keywords)

# Function to detect greetings
def is_greeting(message):
    greeting_keywords = ['salom', 'assalom', 'assalomu alaykum', 'assalomu aleykum']
    return any(keyword.lower() in message.lower() for keyword in greeting_keywords)

# Function to detect thanks
def is_thanks(message):
    thanks_keywords = ['rahmat', 'tashakkur']
    return any(keyword.lower() in message.lower() for keyword in thanks_keywords)

# Function to detect queries about credit amount reason
def is_credit_reason_query(message):
    reason_keywords = ['nima uchun', 'negadir', 'nima sababdan', 'qanday qilib', 'why', 'how']
    return any(keyword.lower() in message.lower() for keyword in reason_keywords) and 'kredit' in message.lower()

# Function to detect queries about bot's name or developer
def is_bot_info_query(message):
    bot_keywords = ['isming', 'kim', 'nomi', 'quruvchi', 'ishlab chiqaruvchi', 'developer']
    return any(keyword.lower() in message.lower() for keyword in bot_keywords)

# Function to generate response using Together API
def generate_response(prompt, chat_history):
    messages = chat_history + [{"role": "user", "content": f"Answer concisely (max 50 words) in Uzbek, using natural and polite language: {prompt}"}]
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            temperature=0.7,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Uzr, hozirda javob bera olmayman."

# Function to transliterate Krill to Latin Uzbek
def krill_to_latin(text):
    krill_to_latin_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'j', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'ъ': '', 'ы': 'i',
        'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya', 'қ': 'q', 'ғ': 'g\'', 'ҳ': 'h',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
        'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
        'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
        'Ф': 'F', 'Х': 'X', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Ъ': '', 'Ы': 'I',
        'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya', 'Қ': 'Q', 'Ғ': 'G\'', 'Ҳ': 'H'
    }
    return ''.join(krill_to_latin_map.get(char, char) for char in text)

# Function to clean transcription output
def clean_transcription(text):
    text = krill_to_latin(text.lower().strip())
    if any(keyword in text for keyword in ['appears', 'unclear', 'similar to', 'transcription in']):
        return None
    match = re.search(r'(mening id raqamim\s+)?([\w\s\'’]+)', text)
    if match:
        return match.group(2).strip()
    return text if text else None

# Function to parse Uzbek number words or digits from a sentence
def uzbek_text_to_number(text):
    number_map = {
        'nol': 0, 'bir': 1, 'ikki': 2, 'uch': 3, 'to\'rt': 4, 'besh': 5,
        'olti': 6, 'yetti': 7, 'sakkiz': 8, 'to\'qqiz': 9,
        'o\'n': 10, 'yigirma': 20, 'o\'ttiz': 30, 'qirq': 40, 'ellik': 50,
        'oltmish': 60, 'yetmish': 70, 'sakson': 80, 'to\'qson': 90,
        'yuz': 100, 'yuzi': 100, 'ming': 1000, 'million': 1000000
    }
    text = krill_to_latin(text.lower().replace('va', '').strip())
    numeric_match = re.search(r'\b\d+\b', text)
    if numeric_match:
        try:
            return int(numeric_match.group())
        except ValueError:
            pass
    words = text.split()
    number_words = []
    for word in words:
        if word in number_map:
            number_words.append(word)
        elif re.match(r'[\d]', word):
            continue
        else:
            break
    if not number_words:
        return None
    if len(number_words) == 1:
        return number_map[number_words[0]]
    total = 0
    current = 0
    for word in number_words:
        num = number_map[word]
        if num in [100, 1000, 1000000]:
            if current == 0:
                current = 1
            total += current * num
            current = 0
        else:
            current += num
    total += current
    return total if total > 0 else None

# Audio recording function
def record_audio(duration=7, fs=44100):
    print("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(wav_temp.name, fs, audio)
    return wav_temp.name

# TTS function using the provided API
def text_to_speech(text):
    url = "https://back.aisha.group/api/v1/tts/post/"
    headers = {
        "x-api-key": tts_api_key,
        "X-Channels": "stereo",
        "X-Quality": "64k",
        "X-Rate": "16000",
        "X-Format": "mp3"
    }
    data = {
        "transcript": text,
        "language": "uz",
        "model": "gulnoza"
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code in (200, 201):
            response_data = response.json()
            audio_url = response_data.get("audio_path")
            if audio_url:
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    return audio_response.content
                else:
                    print(f"Error downloading audio: {audio_response.status_code}")
                    return None
            else:
                print("Error: No audio_path found in response")
                return None
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# Function to play audio
def play_audio(audio_content):
    try:
        audio_file = BytesIO(audio_content)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error playing audio: {e}")

# Gemini transcription function
def generate_transcription(audiofile):
    try:
        genai.configure(api_key=gemini_api_key)
        with open(audiofile, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        audio_part = {
            "inline_data": {
                "mime_type": "audio/wav",
                "data": audio_base64
            }
        }
        prompt = "Transcribe the audio to Uzbek (Latin or Cyrillic) text only. Do not include explanations, metadata, or commentary."
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content([prompt, audio_part])
        return clean_transcription(response.text)
    except Exception as e:
        return None

# Chatbot class to manage state and interactions
class BankChatbot:
    def __init__(self):
        self.chat_history = []
        self.waiting_for_id = False
        self.last_credit_amount = None

    def process_message(self, user_input):
        if not user_input:
            return "Iltimos, aniq ovozli xabar yuboring yoki matn kiriting."
        user_input = krill_to_latin(user_input.lower().strip())
        self.chat_history.append({"role": "user", "content": user_input})

        # Handle greetings
        if is_greeting(user_input):
            response = "Vaalaykum assalom! Sizga qanday yordam bera olaman?"
        # Handle thanks
        elif is_thanks(user_input):
            response = "Arzimaydi! Sizga yordam bera olganimdan xursandman."
        # Handle bot info queries
        elif is_bot_info_query(user_input):
            if 'ism' in user_input or 'nomi' in user_input:
                response = "Mening ismim Epsilion."
            elif 'quruvchi' in user_input or 'ishlab chiqaruvchi' in user_input or 'developer' in user_input:
                response = "Meni ishlab chiqaruvchimning taxallusi Neo."
            else:
                response = "Men Epsilion, Neo tomonidan yaratilganman."
        # Handle credit reason queries
        elif is_credit_reason_query(user_input) and self.last_credit_amount is not None:
            response = "Mening kredit scoring modelim buni bashorat qildi."
        # Handle credit context
        elif self.waiting_for_id or any(is_credit_query(msg["content"]) or "topilmadi" in msg["content"] for msg in self.chat_history[-4:]):
            parsed_id = uzbek_text_to_number(user_input)
            if parsed_id is not None:
                prediction = predict_limit_by_id(parsed_id)
                if isinstance(prediction, str):
                    response = f"ID {parsed_id} topilmadi. Iltimos, boshqa ID kiriting."
                    self.waiting_for_id = True
                else:
                    self.last_credit_amount = prediction
                    response = f"Sizga bir yil muddatga {prediction:.2f} dollar miqdorida kredit bera olamiz."
                    self.waiting_for_id = False
            else:
                response = "To'g'ri ID raqamini kiriting (masalan, '1', 'bir', '127')."
                self.waiting_for_id = True
        else:
            if is_credit_query(user_input):
                self.waiting_for_id = True
                response = "Kredit limiti uchun ID raqamingizni kiriting (masalan, '1', 'bir', '127')."
            else:
                prompt = f"Quyidagi ma'lumot asosida qisqa (50 so'zdan kam) va muloyim javob bering:\n{bank_info}\n\nSavol: {user_input}"
                response = generate_response(prompt, self.chat_history)

        self.chat_history.append({"role": "assistant", "content": response})
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        audio_content = text_to_speech(response)
        if audio_content:
            play_audio(audio_content)
        
        return response

# Main function to run the chatbot with voice input
def main():
    chatbot = BankChatbot()
    print("Ipak Yo'li Bank Chatbotiga xush kelibsiz! Ovozli xabar (7 soniya) yoki matn yozing.")
    print("Chiqish uchun 'exit' yozing yoki ayting.")

    while True:
        choice = input("Ovozli xabar (o) yoki matn (m)? [o/m]: ").lower()
        while choice not in ['o', 'm']:
            print("Iltimos, faqat 'o' yoki 'm' tanlang.")
            choice = input("Ovozli xabar (o) yoki matn (m)? [o/m]: ").lower()
        if choice == 'o':
            wav_file = record_audio(duration=7)
            user_input = generate_transcription(wav_file)
            print(f"\n🔊 Transkriptsiya: {user_input or 'Xato: aniqroq gapiring.'}")
            os.remove(wav_file)
            if not user_input:
                print("Bot: Iltimos, aniqroq gapiring va qayta urinib ko'ring.")
                continue
        else:
            user_input = input("Siz: ")

        if user_input.lower() == 'exit':
            print("Xayr, yana ko'rishamiz!")
            break

        response = chatbot.process_message(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()

