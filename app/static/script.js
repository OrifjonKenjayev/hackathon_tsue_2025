const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const sendBtn = document.getElementById('send-btn');
const textInput = document.getElementById('text-input');
const status = document.getElementById('status');
const chatHistory = document.getElementById('chat-history');
const canvas = document.getElementById('waveform');
const ctx = canvas.getContext('2d');
const ttsPlayer = document.getElementById('tts-player');

let mediaRecorder;
let audioContext;
let analyser;
let dataArray;
let audioChunks = [];
let recordStartTime;

function setupWaveform() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    drawWaveform();
}

function drawWaveform() {
    requestAnimationFrame(drawWaveform);
    analyser.getByteTimeDomainData(dataArray);
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#3b82f6';
    ctx.beginPath();
    const sliceWidth = canvas.width / dataArray.length;
    let x = 0;
    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

function addToChatHistory(role, message) {
    const div = document.createElement('div');
    div.className = `p-2 ${role === 'user' ? 'text-blue-600' : 'text-green-600'}`;
    div.innerHTML = `<strong>${role === 'user' ? 'You' : 'Bot'}:</strong> ${message}`;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function getMediaRecorder(stream) {
    const mimeTypes = ['audio/ogg', 'audio/webm', ''];
    for (let mimeType of mimeTypes) {
        if (mimeType && MediaRecorder.isTypeSupported(mimeType)) {
            return new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 128000 });
        } else if (!mimeType) {
            return new MediaRecorder(stream, { audioBitsPerSecond: 128000 });
        }
    }
    throw new Error('No supported MIME type for MediaRecorder');
}

recordBtn.addEventListener('click', async () => {
    status.textContent = 'Requesting microphone access...';
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = getMediaRecorder(stream);
        audioChunks = [];
        recordStartTime = Date.now();

        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        mediaRecorder.ondataavailable = (e) => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const duration = (Date.now() - recordStartTime) / 1000;
            if (duration < 3) {
                status.textContent = 'Recording too short. Please record at least 3 seconds of clear speech.';
                return;
            }

            const mimeType = mediaRecorder.mimeType || 'audio/ogg';
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            const formData = new FormData();
            formData.append('audio', audioBlob, `recording.${mimeType.split('/')[1] || 'ogg'}`);

            status.textContent = 'Processing audio...';
            try {
                const response = await fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (result.error) {
                    status.textContent = `Error: ${result.error}. Try speaking clearly or use text input.`;
                    return;
                }

                addToChatHistory('user', result.transcript || 'No transcript');
                addToChatHistory('assistant', result.response);
                status.textContent = '';

                if (result.audio_url) {
                    ttsPlayer.src = result.audio_url.toLowerCase(); // Ensure lowercase URL
                    ttsPlayer.play().catch(err => {
                        status.textContent = 'Error playing audio. Check TTS API key or try text input.';
                        console.error('TTS playback error:', err);
                    });

                    // Waveform for playback
                    const audio = new Audio(result.audio_url.toLowerCase());
                    const source = audioContext.createMediaElementSource(audio);
                    source.connect(analyser);
                    analyser.connect(audioContext.destination);
                    audio.play().catch(err => console.error('Waveform audio error:', err));
                } else {
                    status.textContent = 'No audio response received. TTS may have failed.';
                }
            } catch (err) {
                status.textContent = 'Error processing audio. Try typing your query.';
                console.error('Fetch error:', err);
            }
        };

        mediaRecorder.onerror = (event) => {
            status.textContent = `MediaRecorder error: ${event.error.message}`;
            console.error('MediaRecorder error:', event.error);
        };

        mediaRecorder.start();
        recordBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        status.textContent = 'Recording... Speak clearly for at least 3 seconds.';
    } catch (err) {
        let errorMessage = 'Unknown error accessing microphone';
        if (err.name === 'NotAllowedError') {
            errorMessage = 'Please allow microphone access in your browser settings';
        } else if (err.name === 'NotFoundError') {
            errorMessage = 'No microphone found. Please connect a microphone';
        } else if (err.name === 'NotSupportedError') {
            errorMessage = 'Browser does not support MediaRecorder. Try Chrome or Firefox';
        } else {
            errorMessage = `Microphone error: ${err.message}`;
        }
        status.textContent = errorMessage;
        console.error('Microphone error:', err);
    }
});

stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    recordBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    status.textContent = '';
});

sendBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) {
        status.textContent = 'Please enter a query';
        return;
    }
    status.textContent = 'Processing text...';
    try {
        const response = await fetch('/process_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const result = await response.json();

        if (result.error) {
            status.textContent = `Error: ${result.error}. Check TTS API key or try again.`;
            return;
        }

        addToChatHistory('user', result.transcript);
        addToChatHistory('assistant', result.response);
        status.textContent = '';
        textInput.value = '';

        if (result.audio_url) {
            ttsPlayer.src = result.audio_url.toLowerCase(); // Ensure lowercase URL
            ttsPlayer.play().catch(err => {
                status.textContent = 'Error playing audio. Check TTS API key or try again.';
                console.error('TTS playback error:', err);
            });

            // Waveform for playback
            const audio = new Audio(result.audio_url.toLowerCase());
            const source = audioContext.createMediaElementSource(audio);
            source.connect(analyser);
            analyser.connect(audioContext.destination);
            audio.play().catch(err => console.error('Waveform audio error:', err));
        } else {
            status.textContent = 'No audio response received. TTS may have failed.';
        }
    } catch (err) {
        status.textContent = 'Error processing text. Check server or API keys.';
        console.error('Fetch error:', err);
    }
});

setupWaveform();