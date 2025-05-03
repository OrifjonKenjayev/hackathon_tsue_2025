import pandas as pd
import joblib
import requests
import re
import os
import uuid
from together import Together
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from mutagen.oggvorbis import OggVorbis
import os
import uuid
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Together client
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env file")
client = Together(api_key=TOGETHER_API_KEY)

# API keys for STT and TTS
STT_API_KEY = os.getenv('STT_API_KEY')
TTS_API_KEY = os.getenv('TTS_API_KEY')
if not STT_API_KEY:
    raise ValueError("STT_API_KEY not found in .env file")
if not TTS_API_KEY:
    raise ValueError("TTS_API_KEY not found in .env file")

# Load bank information
def load_bank_info(file_path=r'uploads/general_info.txt'):
    if not os.path.exists(file_path):
        return "Bank haqida ma'lumot fayli topilmadi."
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

bank_info = load_bank_info()

# Load model and test data
model = joblib.load(r'uploads/linear_regression_model.pkl')
test_data = pd.read_csv(r'uploads/test_data2.csv')
features = ['Income', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married', 'Ethnicity', 'Balance']

# Predict credit limit
def predict_limit_by_id(input_id):
    if input_id not in test_data['ID'].values:
        return f"ID {input_id} topilmadi."
    input_data = test_data[test_data['ID'] == input_id][features]
    predicted_limit = model.predict(input_data)[0]
    return predicted_limit

# Detect credit query
def is_credit_query(message):
    credit_keywords = ['kredit', 'qarz', 'limit', 'pul olish', 'kredit olish', 'kredit limiti']
    return any(keyword.lower() in message.lower() for keyword in credit_keywords)

# Generate response using Together API
def generate_response(prompt, chat_history):
    messages = chat_history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages
    )
    return response.choices[0].message.content

# Parse Uzbek number words
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

# Get OGG file metadata
def get_ogg_metadata(audio_path):
    try:
        audio = OggVorbis(audio_path)
        duration = audio.info.length
        sample_rate = audio.info.sample_rate
        channels = audio.info.channels
        return {'duration': duration, 'sample_rate': sample_rate, 'channels': channels}
    except Exception as e:
        print(f"Metadata error for {audio_path}: {str(e)}")
        return None

# STT function
def speech_to_text(audio_path):
    # Log MP3 file details
    print(f"Sending MP3 file: {audio_path}, size: {os.path.getsize(audio_path)} bytes")
    # Note: MP3 metadata extraction requires a library like mutagen.mp3
    # Since you want to avoid pydub, we'll skip detailed metadata for now
    # If metadata is critical, consider using mutagen.mp3 (not pydub) later

    url = "https://back.aisha.group/api/v1/stt/post/"
    headers = {"x-api-key": STT_API_KEY}
    data = {
        "title": f"recording_{uuid.uuid4()}",
        "has_diarization": "false",
        "language": "uz"
    }
    files = {"audio": open(audio_path, "rb")}
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        print(f"STT Response Status: {response.status_code}")
        print(f"STT Response Content: {response.text}")
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("transcript", "")
            if not transcript:
                print(f"STT Warning: Empty transcript received")
                return None, "No speech detected in audio"
            return transcript, None
        else:
            error_msg = response.json().get("error", "Unknown STT error")
            print(f"STT Error: Status {response.status_code}, Response: {response.text}")
            return None, error_msg
    except Exception as e:
        print(f"STT Exception: {str(e)}")
        return None, str(e)
    finally:
        files["audio"].close()

# TTS function
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
        print(f"TTS Response Status: {response.status_code}, Content: {response.text}")
        if response.status_code in (200, 201):
            response_data = response.json()
            audio_url = response_data.get("audio_path")
            if not audio_url:
                print("TTS Error: No audio_path in response")
                return None
            audio_response = requests.get(audio_url)
            if audio_response.status_code == 200:
                output_file = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{uuid.uuid4()}.mp3")
                with open(output_file, "wb") as f:
                    f.write(audio_response.content)
                print(f"TTS Output saved: {output_file}")
                return output_file
            else:
                print(f"TTS Audio Fetch Error: Status {audio_response.status_code}")
                return None
        else:
            print(f"TTS Error: Status {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        return None

# Chatbot class
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
                    response = f"ID {parsed_id} uchun taxminiy kredit limiti: {prediction:.2f} dollar. Sizga bir yil muddatga ushbu miqdorda kredit berishimiz mumkin."
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

# Initialize chatbot
chatbot = BankChatbot()

# Flask routes
@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    filename = secure_filename(f"recording_{uuid.uuid4()}.mp3")  # Changed to .mp3
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(audio_path)
    print(f"Saved MP3 file: {audio_path}, size: {os.path.getsize(audio_path)} bytes")

    # STT
    transcript, error = speech_to_text(audio_path)
    if not transcript:
        response_text = error or "Ovozni aniqlashda xatolik yuz berdi. Iltimos, aniq va baland ovozda gapiring."
        tts_path = text_to_speech(response_text)
        if not tts_path:
            return jsonify({
                'transcript': transcript,
                'response': response_text,
                'audio_url': None,
                'error': 'TTS failed to generate audio'
            })
        return jsonify({
            'transcript': transcript,
            'response': response_text,
            'audio_url': f"/uploads/{os.path.basename(tts_path)}"
        })

    # Process with chatbot
    response_text = chatbot.process_message(transcript)
    tts_path = text_to_speech(response_text)
    if not tts_path:
        return jsonify({
            'transcript': transcript,
            'response': response_text,
            'audio_url': None,
            'error': 'TTS failed to generate audio'
        })

    return jsonify({
        'transcript': transcript,
        'response': response_text,
        'audio_url': f"/uploads/{os.path.basename(tts_path)}"
    })

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text_input = data['text']
    response_text = chatbot.process_message(text_input)
    tts_path = text_to_speech(response_text)
    if not tts_path:
        return jsonify({
            'transcript': text_input,
            'response': response_text,
            'audio_url': None,
            'error': 'TTS failed to generate audio'
        })
    return jsonify({
        'transcript': text_input,
        'response': response_text,
        'audio_url': f"/uploads/{os.path.basename(tts_path)}"
    })

@app.route('/uploads/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)