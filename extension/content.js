let subtitleContainer = null;
let rawTextEl = null;
let zhTextEl = null;
let clearTimer = null;

// Initialize subtitle overlay
function initSubtitleOverlay() {
  if (document.getElementById('studio0808-subtitle-container')) return;
  
  subtitleContainer = document.createElement('div');
  subtitleContainer.id = 'studio0808-subtitle-container';
  
  // Create raw text element (English/Japanese)
  rawTextEl = document.createElement('div');
  rawTextEl.id = 'studio0808-subtitle-raw';
  
  // Create translated Chinese text element
  zhTextEl = document.createElement('div');
  zhTextEl.id = 'studio0808-subtitle-zh';
  
  subtitleContainer.appendChild(rawTextEl);
  subtitleContainer.appendChild(zhTextEl);
  
  // Inject styling directly into the DOM
  const style = document.createElement('style');
  style.textContent = `
    #studio0808-subtitle-container {
      position: fixed;
      bottom: 8%;
      left: 50%;
      transform: translateX(-50%);
      z-index: 2147483647;
      width: 85%;
      max-width: 750px;
      padding: 12px 18px;
      background: var(--subtitle-bg, rgba(10, 15, 25, 0.82));
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
      text-align: center;
      pointer-events: auto;
      cursor: grab;
      user-select: none;
      display: none;
      opacity: 0;
      transition: opacity 0.25s ease, transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      font-family: system-ui, -apple-system, sans-serif;
    }
    
    #studio0808-subtitle-container:active {
      cursor: grabbing;
    }
    
    #studio0808-subtitle-container.visible {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    
    #studio0808-subtitle-raw {
      font-size: 14px;
      color: rgba(220, 225, 235, 0.7);
      margin-bottom: 4px;
      line-height: 1.4;
      text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
      font-weight: 400;
    }
    
    #studio0808-subtitle-zh {
      font-size: 19px;
      color: var(--subtitle-color, #ffffff);
      line-height: 1.4;
      text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9);
      font-weight: 700;
    }
  `;
  
  document.head.appendChild(style);
  document.body.appendChild(subtitleContainer);
  
  // Load initial appearance settings from storage
  chrome.storage.local.get(['bgColor', 'textColor'], (result) => {
    const bg = result.bgColor || '#0a0f19';
    const text = result.textColor || '#ffffff';
    applySubtitleStyles(bg, text);
  });
  
  // Drag and drop logic
  let isDragging = false;
  let startX, startY;
  let initialX, initialY;
  
  subtitleContainer.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // Left click only
    
    isDragging = true;
    const rect = subtitleContainer.getBoundingClientRect();
    
    // Switch to exact pixel positions for dragging to prevent offset issues with TranslateX
    subtitleContainer.style.transform = 'none';
    subtitleContainer.style.left = rect.left + 'px';
    subtitleContainer.style.top = rect.top + 'px';
    subtitleContainer.style.bottom = 'auto';
    
    startX = e.clientX;
    startY = e.clientY;
    initialX = rect.left;
    initialY = rect.top;
    
    e.preventDefault();
  });
  
  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    
    subtitleContainer.style.left = (initialX + dx) + 'px';
    subtitleContainer.style.top = (initialY + dy) + 'px';
  });
  
  document.addEventListener('mouseup', () => {
    isDragging = false;
  });
  
  // Double click to reset to default position
  subtitleContainer.addEventListener('dblclick', () => {
    subtitleContainer.style.transform = 'translateX(-50%)';
    subtitleContainer.style.left = '50%';
    subtitleContainer.style.top = 'auto';
    
    const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
    if (fullscreenElement) {
      subtitleContainer.style.position = 'absolute';
      subtitleContainer.style.bottom = '10%';
    } else {
      subtitleContainer.style.position = 'fixed';
      subtitleContainer.style.bottom = '8%';
    }
  });

  // Listen to Fullscreen Change Event
  document.addEventListener('fullscreenchange', handleFullscreenChange);
  document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
}

function handleFullscreenChange() {
  if (!subtitleContainer) return;
  
  const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
  
  if (fullscreenElement) {
    // If browser entered fullscreen, attach subtitle container to the fullscreen wrapper
    fullscreenElement.appendChild(subtitleContainer);
    subtitleContainer.style.position = 'absolute';
    subtitleContainer.style.bottom = '10%';
  } else {
    // Return back to body
    document.body.appendChild(subtitleContainer);
    subtitleContainer.style.position = 'fixed';
    subtitleContainer.style.bottom = '8%';
  }
}

// Listen to runtime messages from background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'show-subtitles') {
    initSubtitleOverlay();
    subtitleContainer.style.display = 'block';
    // Small delay to trigger fade-in transition smoothly
    setTimeout(() => {
      subtitleContainer.classList.add('visible');
    }, 50);
    
    const showBilingual = message.showBilingual !== false;
    if (showBilingual) {
      rawTextEl.style.display = 'block';
      rawTextEl.textContent = '語音系統已連線，準備辨識中...';
    } else {
      rawTextEl.style.display = 'none';
      rawTextEl.textContent = '';
    }
    zhTextEl.textContent = '';
    
    if (clearTimer) clearTimeout(clearTimer);
    clearTimer = setTimeout(clearSubtitle, 3000);
  }
  
  if (message.type === 'hide-subtitles') {
    if (subtitleContainer) {
      subtitleContainer.classList.remove('visible');
      setTimeout(() => {
        subtitleContainer.style.display = 'none';
      }, 250);
    }
  }
  
  if (message.type === 'render-subtitle') {
    console.log("LiveCaption: Received subtitle to render:", message.data);
    initSubtitleOverlay();
    
    const data = message.data;
    const showBilingual = message.showBilingual !== false;
    
    // Only display raw text if bilingual mode is active and raw text is different from translated text
    if (showBilingual && data.text_raw !== data.text_zh) {
      rawTextEl.style.display = 'block';
      rawTextEl.textContent = data.text_raw;
    } else {
      rawTextEl.style.display = 'none';
      rawTextEl.textContent = '';
    }
    
    zhTextEl.textContent = data.text_zh;
    
    subtitleContainer.style.display = 'block';
    subtitleContainer.classList.add('visible');
    
    // Automatically clear subtitles when speech segment expires (add a buffer of 1.5 seconds)
    if (clearTimer) clearTimeout(clearTimer);
    const duration = Math.max(3000, (data.duration || 3) * 1000 + 1500);
    clearTimer = setTimeout(clearSubtitle, duration);
  }
  
  if (message.type === 'toggle-bilingual') {
    initSubtitleOverlay();
    const showBilingual = message.showBilingual !== false;
    if (showBilingual) {
      rawTextEl.style.display = 'block';
    } else {
      rawTextEl.style.display = 'none';
      rawTextEl.textContent = '';
    }
  }
  
  if (message.type === 'update-styles') {
    applySubtitleStyles(message.bgColor, message.textColor);
  }
});

function applySubtitleStyles(bg, text) {
  if (!subtitleContainer) return;
  
  // Appends 'd1' (Hex for 82% opacity) to 6-digit hex colors to make it semi-transparent
  let finalBg = bg;
  if (bg.startsWith('#') && bg.length === 7) {
    finalBg = bg + 'd1';
  }
  
  subtitleContainer.style.setProperty('--subtitle-bg', finalBg);
  subtitleContainer.style.setProperty('--subtitle-color', text);
  subtitleContainer.style.setProperty('--subtitle-color-fade', text); // Disable gradient fade to match exact text color chosen
}

function clearSubtitle() {
  if (rawTextEl && zhTextEl) {
    rawTextEl.textContent = '';
    zhTextEl.textContent = '';
  }
}
