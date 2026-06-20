let mediaStream = null;
let audioContext = null;
let playbackContext = null;
let processor = null;
let ws = null;
let config = {};

chrome.runtime.onMessage.addListener(async (message) => {
  if (message.target !== 'offscreen') return;
  
  if (message.type === 'init-recording') {
    config = message.config;
    await startRecording(message.streamId);
  }
  
  if (message.type === 'update-config') {
    config = message.config;
    sendConfigToBackend();
  }
});

async function startRecording(streamId) {
  try {
    // Proactively clean up any previous recording resources/connections
    cleanup();
    
    // Wait 200ms to allow Chrome to release previous streams
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // 1. Capture stream with retry logic
    const maxRetries = 3;
    let attempt = 0;
    
    while (attempt < maxRetries) {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            mandatory: {
              chromeMediaSource: 'tab',
              chromeMediaSourceId: streamId
            }
          }
        });
        break; // Success, exit retry loop
      } catch (err) {
        attempt++;
        console.warn(`Attempt ${attempt} to capture tab audio failed:`, err);
        if (attempt >= maxRetries) {
          throw err; // Re-throw error if all retries failed
        }
        // Wait 300ms before retrying
        await new Promise(resolve => setTimeout(resolve, 300));
      }
    }
    
    // 2. Play original audio stream back to user so they hear it
    playbackContext = new AudioContext();
    const playbackSource = playbackContext.createMediaStreamSource(mediaStream);
    playbackSource.connect(playbackContext.destination);
    
    // 3. Connect to WebSocket backend
    connectWebSocket();
    
    // 4. Downsample captured stream to 16kHz for SenseVoice ASR
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);
    
    // Buffer size of 4096 frames
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    source.connect(processor);
    processor.connect(audioContext.destination); // Required to trigger onaudioprocess
    
    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      
      const inputData = e.inputBuffer.getChannelData(0); // Float32 Array
      
      // Convert Float32 to 16-bit PCM (Int16)
      const int16Buffer = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        let s = Math.max(-1, Math.min(1, inputData[i]));
        int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      
      // Send binary data over WS
      ws.send(int16Buffer.buffer);
    };
    
  } catch (err) {
    console.error("Offscreen capture failure:", err);
    chrome.runtime.sendMessage({
      target: 'background',
      type: 'offscreen-error',
      error: err.message
    });
  }
}

function connectWebSocket() {
  // Hardcoded to localhost backend (as we want offline/local security)
  ws = new WebSocket('ws://127.0.0.1:8000/stream');
  
  ws.onopen = () => {
    console.log("WebSocket backend connected successfully.");
    chrome.runtime.sendMessage({
      target: 'background',
      type: 'websocket-connected'
    });
    
    // Send configuration instantly after connection opens
    sendConfigToBackend();
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.event === 'subtitle') {
        // Forward ASR & translation subtitles to background worker
        chrome.runtime.sendMessage({
          target: 'background',
          type: 'subtitle-data',
          data: data
        });
      }
    } catch (e) {
      console.warn("Failed to parse backend message:", e);
    }
  };
  
  ws.onclose = () => {
    console.warn("WebSocket closed.");
    chrome.runtime.sendMessage({
      target: 'background',
      type: 'websocket-disconnected'
    });
  };
  
  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

function sendConfigToBackend() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      event: 'config',
      ollama_url: config.ollamaUrl,
      model_name: config.modelName,
      deepseek_key: config.deepseekKey,
      min_silence: config.minSilence,
      max_speech: config.maxSpeech,
      source_lang: config.sourceLang
    }));
  }
}

// Ensure cleanup on window unload
window.addEventListener('unload', () => {
  cleanup();
});

function cleanup() {
  console.log("Cleaning up offscreen contexts...");
  
  if (processor) {
    try {
      processor.disconnect();
    } catch (e) {
      console.warn("Error disconnecting processor:", e);
    }
    processor = null;
  }
  
  if (audioContext) {
    try {
      audioContext.close();
    } catch (e) {
      console.warn("Error closing audioContext:", e);
    }
    audioContext = null;
  }
  
  if (playbackContext) {
    try {
      playbackContext.close();
    } catch (e) {
      console.warn("Error closing playbackContext:", e);
    }
    playbackContext = null;
  }
  
  if (mediaStream) {
    try {
      mediaStream.getTracks().forEach(track => {
        try {
          track.stop();
        } catch (err) {
          console.warn("Error stopping track:", err);
        }
      });
    } catch (e) {
      console.warn("Error stopping mediaStream tracks:", e);
    }
    mediaStream = null;
  }
  
  if (ws) {
    try {
      ws.close();
    } catch (e) {
      console.warn("Error closing WebSocket:", e);
    }
    ws = null;
  }
}
