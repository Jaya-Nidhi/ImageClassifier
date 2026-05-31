/**
 * NeuralEye — Image Classification App
 * Powered by TensorFlow.js + MobileNetV2
 */

/* ─── State ─────────────────────────────────────────────── */
let model = null;
let currentImage = null;
let classificationResults = null;
let webcamStream = null;
let chart = null;

/* ─── DOM Refs ───────────────────────────────────────────── */
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const previewWrap   = document.getElementById('previewWrap');
const previewImg    = document.getElementById('previewImg');
const imageMeta     = document.getElementById('imageMeta');
const scanLine      = document.getElementById('scanLine');
const clearBtn      = document.getElementById('clearBtn');
const classifyBtn   = document.getElementById('classifyBtn');

const webcamBtn     = document.getElementById('webcamBtn');
const webcamContainer = document.getElementById('webcamContainer');
const webcamVideo   = document.getElementById('webcamVideo');
const captureBtn    = document.getElementById('captureBtn');
const stopWebcamBtn = document.getElementById('stopWebcamBtn');

const resultsIdle    = document.getElementById('resultsIdle');
const resultsLoading = document.getElementById('resultsLoading');
const resultsOutput  = document.getElementById('resultsOutput');
const loadingText    = document.getElementById('loadingText');

const topName        = document.getElementById('topName');
const topConfidence  = document.getElementById('topConfidence');
const topFill        = document.getElementById('topFill');
const resultsList    = document.getElementById('resultsList');
const resultsMeta    = document.getElementById('resultsMeta');
const vizPanel       = document.getElementById('vizPanel');

const downloadBtn    = document.getElementById('downloadBtn');
const reclassifyBtn  = document.getElementById('reclassifyBtn');

/* ─── Upload & File Handling ─────────────────────────────── */
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadImageFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadImageFile(fileInput.files[0]);
});

function loadImageFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    showPreview(e.target.result, {
      name: file.name,
      size: formatBytes(file.size),
      type: file.type,
    });
  };
  reader.readAsDataURL(file);
}

function showPreview(src, meta = {}) {
  currentImage = src;
  previewImg.src = src;
  previewWrap.style.display = 'block';
  uploadZone.style.display = 'none';

  if (meta.name) {
    imageMeta.textContent = `${meta.name}  ·  ${meta.size}  ·  ${meta.type}`;
  }

  scanLine.classList.remove('active');
  void scanLine.offsetWidth;
  scanLine.classList.add('active');

  classifyBtn.disabled = false;

  // Reset results
  showState('idle');
  vizPanel.style.display = 'none';
  classificationResults = null;
}

clearBtn.addEventListener('click', () => {
  currentImage = null;
  previewImg.src = '';
  previewWrap.style.display = 'none';
  uploadZone.style.display = 'block';
  classifyBtn.disabled = true;
  fileInput.value = '';
  showState('idle');
  vizPanel.style.display = 'none';
  classificationResults = null;
  if (chart) { chart.destroy(); chart = null; }
});

/* ─── Webcam ─────────────────────────────────────────────── */
webcamBtn.addEventListener('click', async () => {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    webcamVideo.srcObject = webcamStream;
    webcamContainer.style.display = 'block';
    webcamBtn.style.display = 'none';
  } catch (err) {
    alert('Camera access denied or not available. ' + err.message);
  }
});

captureBtn.addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  canvas.width  = webcamVideo.videoWidth;
  canvas.height = webcamVideo.videoHeight;
  canvas.getContext('2d').drawImage(webcamVideo, 0, 0);
  const dataUrl = canvas.toDataURL('image/png');
  stopWebcam();
  showPreview(dataUrl, { name: 'Webcam Capture', size: '—', type: 'image/png' });
});

stopWebcamBtn.addEventListener('click', stopWebcam);

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  webcamContainer.style.display = 'none';
  webcamBtn.style.display = 'flex';
}

/* ─── Classification ─────────────────────────────────────── */
classifyBtn.addEventListener('click', async () => {
  if (!currentImage) return;
  await classify();
});

async function classify() {
  showState('loading');
  classifyBtn.disabled = true;

  const t0 = performance.now();

  try {
    // Step 1: Init TF
    setStep(1);
    await tf.ready();

    // Step 2: Load model (cached after first load)
    setStep(2);
    if (!model) {
      loadingText.textContent = 'Loading MobileNetV2 (~14 MB)…';
      model = await mobilenet.load({ version: 2, alpha: 1.0 });
    }

    // Step 3: Preprocess
    setStep(3);
    loadingText.textContent = 'Preprocessing image…';
    await delay(200);

    // Step 4: Inference
    setStep(4);
    loadingText.textContent = 'Running inference…';
    const imgEl = new Image();
    imgEl.src = currentImage;
    await new Promise(res => { imgEl.onload = res; });

    const predictions = await model.classify(imgEl, 10);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    classificationResults = { predictions, elapsed };
    showResults(predictions, elapsed);

  } catch (err) {
    console.error(err);
    showState('idle');
    alert('Classification failed: ' + err.message);
    classifyBtn.disabled = false;
  }
}

function setStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step${i}`);
    el.className = 'step';
    if (i < n)  el.classList.add('done');
    if (i === n) el.classList.add('active');
  }
}

function showResults(predictions, elapsed) {
  showState('results');
  classifyBtn.disabled = false;

  const top = predictions[0];
  topName.textContent = formatLabel(top.className);
  topConfidence.textContent = `${(top.probability * 100).toFixed(1)}% confidence`;

  setTimeout(() => {
    topFill.style.width = `${top.probability * 100}%`;
  }, 80);

  resultsList.innerHTML = '';
  predictions.slice(0, 10).forEach((p, i) => {
    const pct = (p.probability * 100).toFixed(1);
    const delay = i * 60;
    const item = document.createElement('div');
    item.className = 'result-item';
    item.style.animationDelay = `${delay}ms`;
    item.innerHTML = `
      <div class="result-name-wrap">
        <span class="result-name">${formatLabel(p.className)}</span>
        <div class="result-bar-bg">
          <div class="result-bar-fill" style="width:0" data-width="${pct}"></div>
        </div>
      </div>
      <span class="result-score">${pct}%</span>
    `;
    resultsList.appendChild(item);
    setTimeout(() => {
      item.querySelector('.result-bar-fill').style.width = pct + '%';
    }, delay + 100);
  });

  resultsMeta.innerHTML = `
    <span>⏱ Inference: ${elapsed}s</span>
    <span>📦 Model: MobileNetV2 (ImageNet)</span>
    <span>🔢 Top-10 of 1000 classes</span>
  `;

  vizPanel.style.display = 'block';
  setTimeout(() => renderChart(predictions.slice(0, 8)), 200);
}

/* ─── Chart ──────────────────────────────────────────────── */
function renderChart(preds) {
  const ctx = document.getElementById('confidenceChart').getContext('2d');
  if (chart) chart.destroy();

  const labels = preds.map(p => formatLabel(p.className, 20));
  const data   = preds.map(p => +(p.probability * 100).toFixed(2));
  const colors = preds.map((_, i) => {
    const hue = 180 + i * 22;
    return `hsla(${hue},100%,65%,0.75)`;
  });

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Confidence (%)',
        data,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.75', '1')),
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      animation: { duration: 900, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#8888aa', font: { family: 'Space Mono', size: 10 }, maxRotation: 30 },
          grid: { color: 'rgba(255,255,255,0.04)' }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#8888aa', font: { family: 'Space Mono', size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.06)' }
        }
      }
    }
  });
}

/* ─── Export ─────────────────────────────────────────────── */
downloadBtn.addEventListener('click', () => {
  if (!classificationResults) return;
  const { predictions, elapsed } = classificationResults;
  const rows = predictions.map((p, i) =>
    `${i + 1},${p.className},"${formatLabel(p.className)}",${(p.probability * 100).toFixed(4)}%`
  );
  const csv = [
    'Rank,Raw Label,Formatted Label,Confidence',
    ...rows,
    `,,Inference Time,${elapsed}s`,
    `,,Model,MobileNetV2 (ImageNet)`,
    `,,Date,"${new Date().toISOString()}"`,
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'neuraleye-results.csv';
  a.click();
  URL.revokeObjectURL(url);
});

reclassifyBtn.addEventListener('click', classify);

/* ─── State Switcher ─────────────────────────────────────── */
function showState(state) {
  resultsIdle.style.display    = state === 'idle'    ? 'flex' : 'none';
  resultsLoading.style.display = state === 'loading' ? 'flex' : 'none';
  resultsOutput.style.display  = state === 'results' ? 'flex' : 'none';
}

/* ─── Helpers ────────────────────────────────────────────── */
function formatLabel(label, maxLen = 999) {
  const formatted = label
    .split(',')[0]
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, c => c.toUpperCase());
  return formatted.length > maxLen ? formatted.slice(0, maxLen) + '…' : formatted;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function delay(ms) {
  return new Promise(res => setTimeout(res, ms));
}

/* ─── Eagerly load model in background ──────────────────── */
(async () => {
  await delay(1500);
  try {
    await tf.ready();
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
    console.log('NeuralEye: Model pre-loaded and ready.');
  } catch (e) {
    console.warn('NeuralEye: Model pre-load failed, will retry on classify.', e);
  }
})();

/* ─── Chart.js CDN load ─────────────────────────────────── */
const chartScript = document.createElement('script');
chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
document.head.appendChild(chartScript);
