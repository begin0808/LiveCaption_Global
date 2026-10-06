let subtitleContainer = null;
let subtitleScroll = null;   // 內層捲動容器：字幕行放在這裡，超過高度自己捲動
let subtitleJump = null;     // 「↓ 最新」按鈕：使用者往上翻讀歷史時才出現
let clearTimer = null;
let subtitleHistory = [];
let maxHistoryLines = 0; // 0 = only show latest, 1 = latest + 1 history, 2 = latest + 2 history
let teleprompter = false;  // 提詞面板模式：保留全部歷史、可上下捲動、不自動清空
let pinnedToBottom = true; // 使用者是否停在最新處（決定要不要自動跟隨）

// 面板縮放狀態：面板是疊在影片上的，能自由調大小才不會擋住畫面。
// null 代表「還沒手動調過」，沿用 CSS 的預設值（寬 85%/max 750px、高 58vh）。
let panelW = null;
let panelH = null;
let fontScale = 1;         // Ctrl+滾輪的字級倍率，乘在面板「字號」設定之上

// 提詞面板的歷史上限：夠翻回去找，又不會無限制吃記憶體
const MAX_KEEP_LINES = 300;
const TELEPROMPTER_VALUE = 'panel';

const MIN_PANEL_W = 220;
const MIN_PANEL_H = 80;
const MIN_FONT_SCALE = 0.6;
const MAX_FONT_SCALE = 2.0;

// Initialize subtitle overlay
function initSubtitleOverlay() {
  // 舊版遺留的 DOM（例如重新載入擴充但沒重載頁面）沒有捲動區 → 拆掉重建，
  // 否則會出現「面板能用但捲不動」的怪狀態。
  const stale = document.getElementById('studio0808-subtitle-container');
  if (stale && (!stale.querySelector('.studio0808-subtitle-scroll')
                || !stale.querySelector('.studio0808-resize-handle'))) {
    stale.remove();
  }
  if (document.getElementById('studio0808-subtitle-container')) return;
  
  subtitleContainer = document.createElement('div');
  subtitleContainer.id = 'studio0808-subtitle-container';
  
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
      /* 縱向彈性：拖曳縮放時由外層定高、內層捲動區自動填滿剩餘高度 */
      display: none;
      flex-direction: column;
      box-sizing: border-box;
      min-width: 220px;
      overflow: hidden;
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
    
    .studio0808-subtitle-line {
      margin-bottom: 12px;
      transition: opacity 0.25s ease, transform 0.25s ease;
    }
    
    .studio0808-subtitle-line:last-child {
      margin-bottom: 0;
    }
    
    /* 字幕捲動區：超過高度就自己捲動，這是提詞面板的基礎。
       行放在這裡（而不是直接放容器）→ 面板不會被撐破，下面的按鈕也不會被捲走。 */
    .studio0808-subtitle-scroll {
      flex: 1 1 auto;
      min-height: 0;
      max-height: min(58vh, 560px);
      overflow-y: auto;
      overscroll-behavior: contain;
      scrollbar-width: thin;
      scrollbar-color: rgba(255, 255, 255, 0.28) transparent;
    }
    
    .studio0808-subtitle-scroll::-webkit-scrollbar {
      width: 6px;
    }
    
    .studio0808-subtitle-scroll::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.28);
      border-radius: 3px;
    }
    
    .studio0808-subtitle-scroll::-webkit-scrollbar-track {
      background: transparent;
    }
    
    /* ↓ 最新：往上翻讀歷史時才出現，點一下回到即時字幕 */
    .studio0808-jump-latest {
      display: none;
      margin: 10px auto 0;
      padding: 3px 12px;
      width: fit-content;
      font-size: 12px;
      font-family: inherit;
      color: #fff;
      background: rgba(255, 255, 255, 0.16);
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 999px;
      cursor: pointer;
    }
    
    .studio0808-jump-latest.show {
      display: block;
    }
    
    .studio0808-jump-latest:hover {
      background: rgba(255, 255, 255, 0.26);
    }
    
    /* 右下角縮放把手：拖曳自由調寬高。常駐半透明（不然沒人找得到），
       滑過面板時變亮。雙擊還原預設尺寸與字級。 */
    .studio0808-resize-handle {
      position: absolute;
      right: 0;
      bottom: 0;
      width: 22px;
      height: 22px;
      cursor: nwse-resize;
      opacity: 0.4;
      transition: opacity 0.15s ease;
      z-index: 2;
    }
    
    #studio0808-subtitle-container:hover .studio0808-resize-handle {
      opacity: 0.9;
    }
    
    .studio0808-resize-handle::after {
      content: '';
      position: absolute;
      right: 7px;
      bottom: 6px;
      width: 9px;
      height: 9px;
      border-right: 2px solid rgba(255, 255, 255, 0.9);
      border-bottom: 2px solid rgba(255, 255, 255, 0.9);
      border-bottom-right-radius: 3px;
    }
    
    /* Faded style for history lines */
    /* 提詞面板要看得到上一句，所以只壓低透明度、不再縮放（縮放會讓整排字大小不一） */
    .studio0808-subtitle-line.history-line {
      opacity: 0.72;
      margin-bottom: 10px;
    }
    
    .studio0808-subtitle-raw-item {
      font-size: calc(var(--subtitle-font-size-raw, 14px) * 0.95 * var(--subtitle-scale, 1));
      color: rgba(220, 225, 235, 0.75);
      margin-bottom: 3px;
      line-height: 1.4;
      text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
      font-weight: 400;
    }
    
    .studio0808-subtitle-zh-item {
      font-size: calc(var(--subtitle-font-size-zh, 19px) * var(--subtitle-scale, 1));
      color: var(--subtitle-color, #ffffff);
      line-height: 1.4;
      text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9);
      font-weight: 700;
    }
    
    /* Streaming indicator: a blinking caret while the translation is being generated */
    .studio0808-subtitle-line.streaming .studio0808-subtitle-zh-item::after {
      content: '▍';
      margin-left: 2px;
      font-weight: 400;
      opacity: 0.85;
      animation: studio0808-caret-blink 0.9s steps(1, end) infinite;
    }
    
    @keyframes studio0808-caret-blink {
      0%, 50% { opacity: 0.9; }
      51%, 100% { opacity: 0.1; }
    }
  `;
  
  document.head.appendChild(style);
  document.body.appendChild(subtitleContainer);

  // 字幕行實際放在這個捲動區裡；使用者往上翻時，新字幕不會把他拉回底部
  subtitleScroll = document.createElement('div');
  subtitleScroll.className = 'studio0808-subtitle-scroll';
  subtitleContainer.appendChild(subtitleScroll);

  subtitleJump = document.createElement('button');
  subtitleJump.type = 'button';
  subtitleJump.className = 'studio0808-jump-latest';
  subtitleJump.textContent = '↓ 最新';
  subtitleContainer.appendChild(subtitleJump);

  // 只要不在底部就停止自動跟隨 —— 這是「邊看上一句、下面繼續更新」的關鍵
  subtitleScroll.addEventListener('scroll', () => {
    const gap = subtitleScroll.scrollHeight - subtitleScroll.scrollTop - subtitleScroll.clientHeight;
    pinnedToBottom = gap < 12;
    updateJumpButton();
  });

  // 別讓點按鈕變成拖動整個面板
  subtitleJump.addEventListener('mousedown', (e) => e.stopPropagation());
  subtitleJump.addEventListener('click', (e) => {
    e.stopPropagation();
    scrollToLatest();
  });

  // ── 縮放把手 ────────────────────────────────────────────────────────
  // 面板疊在影片上，寬高不能是死的是「不擋畫面」的前提。
  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'studio0808-resize-handle';
  resizeHandle.title = '拖曳調整大小 · 雙擊還原 · Ctrl+滾輪縮放字級';
  subtitleContainer.appendChild(resizeHandle);

  let isResizing = false;
  let resizeStartX = 0, resizeStartY = 0, resizeStartW = 0, resizeStartH = 0;

  resizeHandle.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    // 一定要擋住冒泡：外層的 mousedown 是「拖動整個面板」，不擋會邊縮邊跑
    e.stopPropagation();
    e.preventDefault();
    isResizing = true;
    const rect = subtitleContainer.getBoundingClientRect();
    resizeStartX = e.clientX;
    resizeStartY = e.clientY;
    resizeStartW = rect.width;
    // 高度基準要取「設定值」而不是目前渲染高度：
    // 內容不足時捲動區只有幾十 px，用渲染高度當基準拖起來會黏在底部。
    resizeStartH = (panelH !== null)
      ? panelH
      : (Math.round(parseFloat(getComputedStyle(subtitleScroll).maxHeight)) || 560);
  });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    applyPanelSize(
      resizeStartW + (e.clientX - resizeStartX),
      resizeStartH + (e.clientY - resizeStartY)
    );
  });

  document.addEventListener('mouseup', () => {
    if (!isResizing) return;
    isResizing = false;
    savePanelSize();
  });

  resizeHandle.addEventListener('dblclick', (e) => {
    e.stopPropagation();   // 否則會同時觸發外層的「還原位置」
    resetPanelSize();
  });

  // Ctrl/⌘ + 滾輪：等比縮放字級。面板調小了字也該跟著小，否則一樣擠爆。
  subtitleContainer.addEventListener('wheel', (e) => {
    if (!e.ctrlKey && !e.metaKey) return;   // 普通滾輪要留給捲動字幕
    e.preventDefault();
    e.stopPropagation();
    applyFontScale(fontScale * (e.deltaY < 0 ? 1.08 : 1 / 1.08));
    savePanelSize();
  }, { passive: false });

  // 視窗尺寸變了要重新夾一次，否則面板可能超出畫面
  window.addEventListener('resize', () => {
    if (panelW !== null || panelH !== null) applyPanelSize(panelW, panelH);
  });
  
  // Load initial appearance settings from storage
  chrome.storage.local.get(
    ['bgColor', 'textColor', 'fontSize', 'historyLines', 'panelWidth', 'panelHeight', 'subtitleScale'],
    (result) => {
      const bg = result.bgColor || '#0a0f19';
      const text = result.textColor || '#ffffff';
      const size = result.fontSize || 'medium';
      applyHistorySetting(result.historyLines);
      applySubtitleStyles(bg, text, size);
      // 還原上次手動調過的尺寸與字級（存在 storage 裡，換頁也還在）
      if (result.subtitleScale) applyFontScale(result.subtitleScale);
      if (result.panelWidth || result.panelHeight) {
        applyPanelSize(result.panelWidth, result.panelHeight);
      }
    });
  
  // Drag and drop logic
  let isDragging = false;
  let startX, startY;
  let initialX, initialY;
  
  subtitleContainer.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // Left click only
    isDragging = true;
    const rect = subtitleContainer.getBoundingClientRect();
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
  
  document.addEventListener('mouseup', () => { isDragging = false; });
  
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
    fullscreenElement.appendChild(subtitleContainer);
    subtitleContainer.style.position = 'absolute';
    subtitleContainer.style.bottom = '10%';
  } else {
    document.body.appendChild(subtitleContainer);
    subtitleContainer.style.position = 'fixed';
    subtitleContainer.style.bottom = '8%';
  }
  // 全螢幕切換會換掉可用畫面大小，手動調過的尺寸要重新夾一次上限
  if (panelW !== null || panelH !== null) applyPanelSize(panelW, panelH);
}

// Render history subtitles based on configurations
function renderHistorySubtitles(targetLang, showBilingual) {
  if (!subtitleContainer) return;
  
  // 只清「字幕行」，捲動容器與「↓ 最新」按鈕本身要留著。
  // 重建前先記下閱讀位置：使用者正在往上翻時，不能被新字幕搶回底部。
  const container = subtitleScroll || subtitleContainer;
  const prevTop = container.scrollTop;
  const wasPinned = pinnedToBottom;
  const hadLines = container.childElementCount > 0;

  container.innerHTML = ''; // Clear previous elements
  
  subtitleHistory.forEach((item, index) => {
    const isLatest = (index === subtitleHistory.length - 1);
    
    const lineWrapper = document.createElement('div');
    lineWrapper.className = 'studio0808-subtitle-line';
    if (!isLatest) {
      lineWrapper.classList.add('history-line');
    }
    // 記下起始時間，串流更新時才能精準定位到同一行做原地改字
    if (item.start !== undefined) {
      lineWrapper.dataset.start = String(item.start);
    }
    if (item.streaming) {
      lineWrapper.classList.add('streaming');
    }
    
    if (targetLang === 'none') {
      // Show only raw text with primary styling (larger and bolder)
      const rawEl = document.createElement('div');
      rawEl.className = 'studio0808-subtitle-zh-item';
      rawEl.textContent = item.text_raw || item.text_zh;
      lineWrapper.appendChild(rawEl);
    } else {
      // Translation is active
      if (showBilingual) {
        // Show raw (top) + translated (bottom)
        if (item.text_raw && item.text_raw !== item.text_zh) {
          const rawEl = document.createElement('div');
          rawEl.className = 'studio0808-subtitle-raw-item';
          rawEl.textContent = item.text_raw;
          lineWrapper.appendChild(rawEl);
        }
        const zhEl = document.createElement('div');
        zhEl.className = 'studio0808-subtitle-zh-item';
        zhEl.textContent = item.text_zh;
        lineWrapper.appendChild(zhEl);
      } else {
        // Show only translated text with primary styling (larger and bolder)
        const zhEl = document.createElement('div');
        zhEl.className = 'studio0808-subtitle-zh-item';
        zhEl.textContent = item.text_zh;
        lineWrapper.appendChild(zhEl);
      }
    }
    
    container.appendChild(lineWrapper);
  });
  
  subtitleContainer.style.display = 'flex';   // 縱向 flex：捲動區自動填滿剩餘高度
  subtitleContainer.classList.add('visible');

  // 自動跟隨：只有使用者本來就停在最新處（或首次繪製）才捲到底；
  // 已經往上翻的就停在原地，讓新字幕在下面繼續長，不打斷閱讀。
  if (wasPinned || !hadLines) {
    scrollToLatest();
  } else {
    container.scrollTop = prevTop;
    updateJumpButton();
  }
}

// 串流／滾動更新：只改既有那一行的文字，避免整塊重建造成閃爍與 transition 重播
// 回傳 true 表示有找到對應的行並改完；false 表示這行還不存在，需呼叫方整塊重繪
function patchSubtitleLine(start, textRaw, textZh, streaming) {
  if (!subtitleContainer) return false;

  const key = String(start);
  const lines = subtitleContainer.querySelectorAll('.studio0808-subtitle-line');
  for (const line of lines) {
    if (line.dataset.start !== key) continue;

    // 滾動字幕的原文會先到、譯文後到，兩者都要能就地更新
    const rawEl = line.querySelector('.studio0808-subtitle-raw-item');
    if (rawEl && textRaw) rawEl.textContent = textRaw;

    const zhEl = line.querySelector('.studio0808-subtitle-zh-item');
    if (zhEl) {
      zhEl.textContent = (textZh === undefined || textZh === null) ? '' : textZh;
    } else if (textZh) {
      // 這行先前只顯示原文（沒有譯文元素），現在有譯文了 → 交由整塊重繪補上
      return false;
    }

    line.classList.toggle('streaming', !!streaming);
    subtitleContainer.style.display = 'flex';
    subtitleContainer.classList.add('visible');
    // 最新那行會越長越高；只有在使用者沒往上翻時才跟著捲，
    // 否則他正在讀的那段歷史會被推走。
    if (pinnedToBottom && subtitleScroll) {
      subtitleScroll.scrollTop = subtitleScroll.scrollHeight;
    }
    return true;
  }
  return false;
}

// Listen to runtime messages from background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'show-subtitles') {
    initSubtitleOverlay();
    subtitleHistory = []; // Reset history queue
    pinnedToBottom = true;
    if (subtitleScroll) subtitleScroll.innerHTML = '';
    
    const targetLang = message.targetLang || 'none';
    const showBilingual = message.showBilingual !== false;
    
    // Push initial status placeholder
    subtitleHistory.push({
      text_raw: targetLang === 'none' ? '語音系統已連線，準備辨識中...' : '',
      text_zh: '語音系統已連線，準備辨識中...'
    });
    
    renderHistorySubtitles(targetLang, showBilingual);
    
    scheduleClear(4000);
  }
  
  if (message.type === 'hide-subtitles') {
    clearSubtitle();
  }
  
  if (message.type === 'render-subtitle') {
    initSubtitleOverlay();
    
    const data = message.data;
    const targetLang = message.targetLang || 'none';
    const showBilingual = message.showBilingual !== false;
    const eventType = data.event || 'subtitle';
    
    if (eventType !== 'subtitle_chunk') {
      console.log("LiveCaption: Received subtitle to render:", data);
    }
    
    // Clear initial status placeholders if any
    subtitleHistory = subtitleHistory.filter(item => item.text_zh !== '語音系統已連線，準備辨識中...');
    
    // 同一句以 start（起始時間）為身分識別。
    // 後端實測 VAD 的 current_segment.start 與定段後的 front.start 完全相同，
    // 所以「滾動臨時字幕」與「定稿」落在同一個 start → 可以原地改字、無縫替換。
    const existingIndex = subtitleHistory.findIndex(item => item.start === data.start);

    const limit = historyLimit();
    const trimHistory = () => {
      if (subtitleHistory.length > limit) {
        subtitleHistory = subtitleHistory.slice(subtitleHistory.length - limit);
      }
    };

    // ── 串流中間態（定稿翻譯逐字回推）：只做「原地改字」，不重建 DOM ──
    //    整塊重建會讓 CSS transition 反覆重播，且每秒數十次並不經濟。
    if (eventType === 'subtitle_chunk' && existingIndex !== -1) {
      subtitleHistory[existingIndex].text_raw = data.text_raw;
      subtitleHistory[existingIndex].text_zh = data.text_zh;
      subtitleHistory[existingIndex].streaming = true;
      if (!patchSubtitleLine(data.start, data.text_raw, data.text_zh, true)) {
        trimHistory();
        renderHistorySubtitles(targetLang, showBilingual);
      }
      return;
    }

    // ── 滾動臨時字幕：說話中就持續刷新同一行，定段後由 subtitle 定稿覆蓋 ──
    //    這一步是首字延遲從 5~9 秒（等說話人停頓）降到約 1 秒的來源。
    //    文字會在定稿時被修正 —— 這與 Zoom / Google Meet 的字幕行為一致。
    if (eventType === 'subtitle_partial') {
      const rec = {
        text_raw: data.text_raw,
        text_zh: data.text_zh,
        duration: data.duration,
        start: data.start,
        streaming: true
      };
      if (existingIndex !== -1) {
        subtitleHistory[existingIndex] = rec;
      } else {
        subtitleHistory.push(rec);
      }
      if (!patchSubtitleLine(data.start, data.text_raw, data.text_zh, true)) {
        trimHistory();
        renderHistorySubtitles(targetLang, showBilingual);
      }
      // 說話持續中就先給足時間，別讓臨時字幕中途淡出
      scheduleClear(15000);
      return;
    }
    
    const record = {
      text_raw: data.text_raw,
      text_zh: data.text_zh,
      duration: data.duration,
      start: data.start,
      // subtitle_begin 代表「翻譯尚未回來」，先顯示閃爍游標示意正在生成
      streaming: eventType === 'subtitle_begin'
    };
    
    if (existingIndex !== -1) {
      subtitleHistory[existingIndex] = record;
    } else {
      subtitleHistory.push(record);
    }
    
    trimHistory();
    
    renderHistorySubtitles(targetLang, showBilingual);
    
    // Set auto-fade timer（提詞面板模式下不會清空，見 scheduleClear）
    if (eventType === 'subtitle_begin') {
      // 串流進行中先給足時間，避免翻譯還沒吐完字幕就淡出；
      // 最終結果（subtitle）到達時會依實際語音長度重設。
      scheduleClear(15000);
    } else {
      const duration = Math.max(3000, (data.duration || 3) * 1000 + 1500);
      scheduleClear(duration);
    }
  }
  
  if (message.type === 'toggle-bilingual' || message.type === 'update-subtitle-mode') {
    initSubtitleOverlay();
    const targetLang = message.targetLang || 'none';
    const showBilingual = message.showBilingual !== false;
    renderHistorySubtitles(targetLang, showBilingual);
  }
  
  if (message.type === 'update-styles') {
    initSubtitleOverlay();
    applySubtitleStyles(message.bgColor, message.textColor, message.fontSize);
  }
  
  if (message.type === 'update-history-lines') {
    applyHistorySetting(message.historyLines);
    // Prune queue immediately if new limit is smaller
    const limit = historyLimit();
    if (subtitleHistory.length > limit) {
      subtitleHistory = subtitleHistory.slice(subtitleHistory.length - limit);
    }
    chrome.storage.local.get(['targetLang', 'showBilingual'], (result) => {
      const targetLang = result.targetLang || 'none';
      const showBilingual = result.showBilingual !== false;
      renderHistorySubtitles(targetLang, showBilingual);
    });
  }
});

function applySubtitleStyles(bg, text, fontSize) {
  if (!subtitleContainer) return;
  
  // Appends 'd1' (Hex for 82% opacity) to 6-digit hex colors to make it semi-transparent
  let finalBg = bg;
  if (bg.startsWith('#') && bg.length === 7) {
    finalBg = bg + 'd1';
  }
  
  subtitleContainer.style.setProperty('--subtitle-bg', finalBg);
  subtitleContainer.style.setProperty('--subtitle-color', text);
  subtitleContainer.style.setProperty('--subtitle-color-fade', text); 
  
  // Map selectors to specific font size scales
  let rawSize = '14px';
  let zhSize = '19px';
  
  if (fontSize === 'small') {
    rawSize = '12px';
    zhSize = '16px';
  } else if (fontSize === 'medium') {
    rawSize = '14px';
    zhSize = '19px';
  } else if (fontSize === 'large') {
    rawSize = '18px';
    zhSize = '24px';
  } else if (fontSize === 'xlarge') {
    rawSize = '22px';
    zhSize = '28px';
  }
  
  subtitleContainer.style.setProperty('--subtitle-font-size-raw', rawSize);
  subtitleContainer.style.setProperty('--subtitle-font-size-zh', zhSize);
}

function clearSubtitle() {
  subtitleHistory = [];
  pinnedToBottom = true;
  if (subtitleContainer) {
    subtitleContainer.classList.remove('visible');
    setTimeout(() => {
      // Check if another segment hasn't triggered visibility before hiding
      if (!subtitleContainer.classList.contains('visible')) {
        subtitleContainer.style.display = 'none';
        // 只清字幕行；捲動容器與「↓ 最新」按鈕要留著
        if (subtitleScroll) subtitleScroll.innerHTML = '';
        if (subtitleJump) subtitleJump.classList.remove('show');
      }
    }, 250);
  }
}

// ── 提詞面板的輔助函式 ────────────────────────────────────────────────

// 'panel' = 提詞面板：保留全部歷史、可上下捲動、不自動清空
function applyHistorySetting(value) {
  if (String(value) === TELEPROMPTER_VALUE) {
    teleprompter = true;
    maxHistoryLines = MAX_KEEP_LINES;
    return;
  }
  teleprompter = false;
  let n = (value === undefined || value === null) ? 0 : parseInt(value, 10);
  if (isNaN(n) || n < 0) n = 0;
  maxHistoryLines = n;
}

// 目前實際要保留的行數（含最新那行）
function historyLimit() {
  return teleprompter ? MAX_KEEP_LINES : maxHistoryLines + 1;
}

// 提詞面板常駐不清空；其餘模式維持原本的「閒置即淡出」
function scheduleClear(ms) {
  if (clearTimer) {
    clearTimeout(clearTimer);
    clearTimer = null;
  }
  if (teleprompter) return;
  clearTimer = setTimeout(clearSubtitle, ms);
}

// 捲到最新一行，並恢復自動跟隨
function scrollToLatest() {
  if (!subtitleScroll) return;
  subtitleScroll.scrollTop = subtitleScroll.scrollHeight;
  pinnedToBottom = true;
  updateJumpButton();
}

// 「↓ 最新」按鈕只在「有東西可捲」且「使用者不在底部」時出現
function updateJumpButton() {
  if (!subtitleJump || !subtitleScroll) return;
  const scrollable = subtitleScroll.scrollHeight - subtitleScroll.clientHeight > 12;
  subtitleJump.classList.toggle('show', scrollable && !pinnedToBottom);
}

// ── 面板縮放 ──────────────────────────────────────────────────────────
// 寬度定在容器上、高度定在容器上（由內層捲動區 flex 填滿）。
// 兩者都可傳 null/undefined 表示「這一個維度不動」，方便只調單邊。

function applyPanelSize(w, h) {
  if (!subtitleContainer || !subtitleScroll) return;

  if (w !== undefined && w !== null && !isNaN(w)) {
    panelW = Math.max(MIN_PANEL_W, Math.min(Math.round(w), window.innerWidth - 24));
    subtitleContainer.style.width = panelW + 'px';
    // 預設有 max-width: 750px，不解除就永遠拖不寬
    subtitleContainer.style.maxWidth = 'none';
  }

  if (h !== undefined && h !== null && !isNaN(h)) {
    panelH = Math.max(MIN_PANEL_H, Math.min(Math.round(h), Math.round(window.innerHeight * 0.92)));
    subtitleContainer.style.height = panelH + 'px';
    // 容器已定高，捲動區再吃 max-height 會讓底部留一段空白
    subtitleScroll.style.maxHeight = 'none';
  }

  // 高度變了會影響「是否可捲動」的判斷
  if (subtitleScroll) {
    const gap = subtitleScroll.scrollHeight - subtitleScroll.scrollTop - subtitleScroll.clientHeight;
    pinnedToBottom = gap < 12;
    updateJumpButton();
  }
}

function applyFontScale(scale) {
  fontScale = Math.max(MIN_FONT_SCALE, Math.min(MAX_FONT_SCALE, Number(scale) || 1));
  if (subtitleContainer) {
    subtitleContainer.style.setProperty('--subtitle-scale', String(fontScale));
  }
}

// 雙擊把手：尺寸與字級一起還原（不然字被縮到很小就沒救了）
function resetPanelSize() {
  panelW = null;
  panelH = null;
  if (subtitleContainer) {
    subtitleContainer.style.width = '';
    subtitleContainer.style.maxWidth = '';
    subtitleContainer.style.height = '';
  }
  if (subtitleScroll) subtitleScroll.style.maxHeight = '';
  applyFontScale(1);
  savePanelSize();
}

// 縮放結果要跨頁面/跨場次留存，不然每開一次新影片就得重調
function savePanelSize() {
  try {
    chrome.storage.local.set({
      panelWidth: panelW,
      panelHeight: panelH,
      subtitleScale: fontScale
    });
  } catch (e) {
    // 擴充被停用時 chrome.storage 會拋錯，不影響當下顯示
  }
}
