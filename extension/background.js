let isCapturing = false;
let isConnected = false;
let activeTabId = null;

// Listen to messages
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'get-status') {
    sendResponse({ isCapturing, isConnected });
    return true;
  }
  
  if (message.type === 'start-capture') {
    startCapture(message.streamId, message.tabId).then(() => sendResponse({ success: true }));
    return true;
  }
  
  if (message.type === 'stop-capture') {
    stopCapture().then(() => sendResponse({ success: true }));
    return true;
  }
  
  if (message.type === 'update-config') {
    // Forward config updates to offscreen if active
    if (isCapturing) {
      chrome.runtime.sendMessage({
        type: 'update-config',
        target: 'offscreen',
        config: message.config
      });
      // 即時通知 content.js 切換雙語/單語顯示
      if (activeTabId) {
        chrome.tabs.sendMessage(activeTabId, {
          type: 'toggle-bilingual',
          showBilingual: message.config.showBilingual
        }).catch(() => {});
      }
    }
    sendResponse({ success: true });
    return true;
  }
  
  // Messages from Offscreen
  if (message.target === 'background') {
    if (message.type === 'websocket-connected') {
      isConnected = true;
    }
    
    if (message.type === 'websocket-disconnected') {
      isConnected = false;
    }
    
    if (message.type === 'subtitle-data') {
      // Forward subtitle translation to content script in the active tab
      if (activeTabId) {
        chrome.storage.local.get(['showBilingual'], (result) => {
          const showBilingual = result.showBilingual !== false;
          chrome.tabs.sendMessage(activeTabId, {
            type: 'render-subtitle',
            data: message.data,
            showBilingual: showBilingual
          }).catch(err => {
            console.warn("Failed to send subtitle to content script (tab might have been closed or reloaded):", err);
          });
        });
      }
    }
    
    if (message.type === 'offscreen-error') {
      console.error("Offscreen capture error:", message.error);
      stopCapture();
    }
  }
});

async function startCapture(streamId, tabId) {
  if (isCapturing) return;
  
  try {
    activeTabId = tabId;
    
    // 3. Load config from storage
    const storage = await chrome.storage.local.get(['ollamaUrl', 'modelName', 'deepseekKey', 'minSilence', 'maxSpeech', 'showBilingual', 'sourceLang']);
    const config = {
      ollamaUrl: storage.ollamaUrl || 'http://localhost:11434',
      modelName: storage.modelName || 'qwen2.5:3b-instruct',
      deepseekKey: storage.deepseekKey || '',
      minSilence: storage.minSilence !== undefined ? storage.minSilence : 0.5,
      maxSpeech: storage.maxSpeech !== undefined ? storage.maxSpeech : 6.0,
      sourceLang: storage.sourceLang || 'auto'
    };
    const showBilingual = storage.showBilingual !== false;
    
    // 4. Create Offscreen Document if it doesn't exist
    const hasDocument = await chrome.offscreen.hasDocument();
    if (!hasDocument) {
      await chrome.offscreen.createDocument({
        url: 'offscreen.html',
        reasons: ['USER_MEDIA'],
        justification: 'Capture tab audio for real-time speech transcription'
      });
    }
    
    // 5. Tell Content Script to show/reset subtitles overlay
    chrome.tabs.sendMessage(activeTabId, { 
      type: 'show-subtitles',
      showBilingual: showBilingual
    }).catch(() => {
      // Content script might not be loaded yet, ignoring
    });
    
    // 6. Tell Offscreen Document to start recording and connect WebSocket
    // We add a tiny delay to ensure the offscreen document is ready to listen
    setTimeout(() => {
      chrome.runtime.sendMessage({
        type: 'init-recording',
        target: 'offscreen',
        streamId: streamId,
        config: config
      });
    }, 300);
    
    isCapturing = true;
    console.log(`Started tab audio capture on tab: ${activeTabId}`);
    
  } catch (err) {
    console.error("Failed to start capture:", err);
    isCapturing = false;
    isConnected = false;
    activeTabId = null;
  }
}

async function stopCapture() {
  if (!isCapturing) return;
  
  try {
    // 1. Close Offscreen Document
    const hasDocument = await chrome.offscreen.hasDocument();
    if (hasDocument) {
      await chrome.offscreen.closeDocument();
    }
    
    // 2. Tell Content Script to hide subtitles
    if (activeTabId) {
      chrome.tabs.sendMessage(activeTabId, { type: 'hide-subtitles' }).catch(() => {});
    }
    
  } catch (err) {
    console.error("Failed to stop capture safely:", err);
  } finally {
    isCapturing = false;
    isConnected = false;
    activeTabId = null;
    console.log("Stopped tab audio capture");
  }
}
