import pandas as pd
import joblib
import requests
import re
import os
import time
from together import Together
import uuid
from google.colab import files  # For file upload in Colab

# Initialize Together client (replace with your API key)
client = Together(api_key="8494e37999569861cb84f3e9a07df5b7f5bf3bb538fedca8cda75c571f8224b2")  # Replace with your Together API key

# API keys for STT and TTS
STT_API_KEY = "V0cMBJOY.EZABBVcZP49UEVfBjbkTKnCIugdrx5XL"  # Replace with your STT API key
TTS_API_KEY = "V0cMBJOY.EZABBVcZP49UEVfBjbkTKnCIugdrx5XL"  # Replace with your TTS API key

# Load bank information from text file
def load_bank_info(file_path='general_info.txt'):
    if not os.path.exists(file_path):
        return "Bank haqida ma'lumot fayli topilmadi."
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

bank_info = load_bank_info()

# Load the saved model and test data
model = joblib.load('linear_regression_model.pkl')
test_data = pd.read_csv('test_data2.csv')
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

# Function to generate response using Together API
def generate_response(prompt, chat_history):
    messages = chat_history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages
    )
    return response.choices[0].message.content

# Function to parse Uzbek number words or digits from a sentence
def uzbek_text_to_number(text):
    number_map = {
        'nol': 0, 'bir': 1, 'ikki': 2, 'uch': 3, 'to\'rt': 4, 'besh': 5,
        'olti': 6, 'yetti': 7, 'sakkiz': 8, 'to\'qqiz': 9,
        'o\'n': 10, 'yigirma': 20, 'o\'ttiz': 30, 'qirq': 40, 'ellik': 50,
        'oltmish': 60, 'yetmish': 70, 'sakson': 80, 'to\'qson': 90,
        'yuz': 100, 'yuzi': 100, 'ming': 1000, 'million': 1000000
    }
    text = text.lower().replace('va', '').strip()
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
            if number_words:
                break
    if not number_words:
        return None
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

# Function to upload audio file in Colab
def upload_audio():
    print("Please upload an OGG audio file.")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return None
    audio_file = list(uploaded.keys())[0]
    if not audio_file.endswith('.ogg'):
        print("Please upload an OGG file.")
        return None
    return audio_file

# Function to perform STT
def speech_to_text(audio_file):
    url = "https://back.aisha.group/api/v1/stt/post/"
    headers = {"x-api-key": STT_API_KEY}
    data = {
        "title": f"recording_{uuid.uuid4()}",
        "has_diarization": "false",
        "language": "uz"
    }
    files = {"audio": open(audio_file, "rb")}
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code == 200:
            result = response.json()
            print("STT Result:", result)
            return result.get("transcript", "")
        else:
            print(f"STT Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"STT Error: {e}")
        return None
    finally:
        files["audio"].close()

# Function to perform TTS
def text_to_speech(text):
    url = "https://back.aisha.group/api/v1/tts/post/"
    headers = {
        "x-api-key": TTS_API_KEY,
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
                    output_file = f"output_{uuid.uuid4()}.mp3"
                    with open(output_file, "wb") as f:
                        f.write(audio_response.content)
                    print(f"TTS Success! Audio saved as '{output_file}'")
                    return output_file
                else:
                    print(f"Error downloading audio: {audio_response.status_code}")
                    print(audio_response.text)
            else:
                print("Error: No audio_path found in response")
                print(response_data)
        else:
            print(f"TTS Error: {response.status_code}")
            print(response.text)
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# Function to play audio (Colab: download instead of play)
def play_audio(file_path):
    print(f"Audio generated: {file_path}")
    # Colab doesn't support direct audio playback; download the file
    files.download(file_path)

# Function to list files in directory
def list_directory():
    print("Files in current directory:")
    for f in os.listdir():
        print(f" - {f}")

# Chatbot class with STT and TTS integration
class BankChatbot:
    def __init__(self):
        self.chat_history = []
        self.waiting_for_id = False

    def process_message(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        if self.waiting_for_id:
            parsed_id = uzbek_text_to_number(user_input)
            if parsed_id is not None:
                prediction = predict_limit_by_id(parsed_id)
                if isinstance(prediction, str):
                    response = prediction
                else:
                    response = f"ID {parsed_id} uchun taxminiy kredit limiti: {prediction:.2f} USD. Sizga bir yil muddatga ushbu miqdorda kredit berishimiz mumkin."
                self.waiting_for_id = False
            else:
                response = "Iltimos, to'g'ri ID raqamini kiriting (raqamlar yoki so'zlar bilan, masalan, '127' yoki 'bir yuz yigirma yetti')."
        else:
            if is_credit_query(user_input):
                self.waiting_for_id = True
                response = "Kredit limiti haqida ma'lumot olish uchun ID raqamingizni kiriting (raqamlar yoki so'zlar bilan, masalan, '127' yoki 'bir yuz yigirma yetti')."
            else:
                prompt = f"Quyidagi ma'lumot asosida savolga javob bering:\n{bank_info}\n\nSavol: {user_input}"
                response = generate_response(prompt, self.chat_history)
        self.chat_history.append({"role": "assistant", "content": response})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        return response

# Main function to run the voice-based chatbot
def main():
    chatbot = BankChatbot()
    print("Ipak Yo'li Bank Voice Chatbotiga xush kelibsiz!")
    print("Upload an OGG audio file to start. Say 'exit' to quit.")

    while True:
        # Upload audio
        audio_file = upload_audio()
        if not audio_file:
            print("Bot: Please upload a valid OGG file.")
            audio_response = text_to_speech("Iltimos, to'g'ri OGG faylini yuklang.")
            if audio_response:
                play_audio(audio_response)
                list_directory()  # Show files after TTS
            continue

        # Perform STT
        transcript = speech_to_text(audio_file)
        if not transcript:
            print("Bot: Ovozni aniqlashda xatolik yuz berdi. Iltimos, qayta urinib ko'ring.")
            audio_response = text_to_speech("Ovozni aniqlashda xatolik yuz berdi. Iltimos, qayta urinib ko'ring.")
            if audio_response:
                play_audio(audio_response)
                list_directory()  # Show files after TTS
            continue

        print(f"User (STT): {transcript}")
        if "exit" in transcript.lower():
            print("Bot: Xayr, yana ko'rishamiz!")
            audio_response = text_to_speech("Xayr, yana ko'rishamiz!")
            if audio_response:
                play_audio(audio_response)
                list_directory()  # Show files after TTS
            break

        # Process the transcript with the chatbot
        response = chatbot.process_message(transcript)
        print(f"Bot: {response}")

        # Generate and play TTS response
        audio_response = text_to_speech(response)
        if audio_response:
            play_audio(audio_response)
            list_directory()  # Show files after TTS

        # Clean up only the uploaded OGG file (keep MP3 files)
        try:
            os.remove(audio_file)
        except:
            pass

if __name__ == "__main__":
    main()