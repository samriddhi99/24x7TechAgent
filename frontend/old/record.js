// Call-style recorder: Start -> hold to talk (pause/resume) -> Stop -> MP3 auto-download

const $ = (id) => document.getElementById(id);
const log = (m) => { const el = $('log'); el.textContent += m + '\n'; el.scrollTop = el.scrollHeight; };
const setStatus = (t) => $('callStatus').textContent = t;

let callBtn, talkBtn, timerEl, vuEl, bitrateEl;

let callActive = false;
let stream, mediaRecorder, chunks = [];
let audioCtx, analyser, dataArray, raf;
let startedAt = 0, ticker = null;

// Press-and-hold Talk = resume/pause on a single recorder
function pressStart(e) {
  e.preventDefault();
  if (!callActive) return;
  try { mediaRecorder && mediaRecorder.state === 'paused' && mediaRecorder.resume(); } catch {}
  talkBtn.textContent = '● Release to Stop';
  talkBtn.style.opacity = '0.95';
  setStatus('Recording… release to stop');
}

function pressEnd(e) {
  e.preventDefault();
  if (!callActive) return;
  try { mediaRecorder && mediaRecorder.state === 'recording' && mediaRecorder.pause(); } catch {}
  talkBtn.textContent = '● Hold to Talk';
  talkBtn.style.opacity = '1';
  setStatus('Connected. Hold to talk.');
}

// Initialize when DOM is ready
function initializeUI() {
  callBtn   = $('callBtn');
  talkBtn   = $('talkBtn');
  timerEl   = $('timer');
  vuEl      = $('vu');
  bitrateEl = $('bitrate');

  // Attach event listeners
  callBtn.addEventListener('click', handleCallClick);
  talkBtn.addEventListener('mousedown', pressStart);
  talkBtn.addEventListener('mouseup', pressEnd);
  talkBtn.addEventListener('mouseleave', pressEnd);
  talkBtn.addEventListener('touchstart', pressStart, { passive:false });
  talkBtn.addEventListener('touchend', pressEnd);
}

function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
}

function startTimer() {
  startedAt = performance.now();
  ticker = setInterval(() => timerEl.textContent = formatTime(performance.now() - startedAt), 250);
}
function stopTimer() {
  clearInterval(ticker); ticker = null;
  timerEl.textContent = '00:00';
}

async function ensureLameEncoder() {
  // Check if lamejs Mp3Encoder is available
  if (window.lamejs && typeof window.lamejs.Mp3Encoder === 'function') {
    log('lamejs encoder ready');
    return true;
  }

  // Wait for lamejs to be available on window
  let attempts = 0;
  while (!(window.lamejs && typeof window.lamejs.Mp3Encoder === 'function') && attempts < 100) {
    await new Promise(resolve => setTimeout(resolve, 50));
    attempts++;
  }

  if (!(window.lamejs && typeof window.lamejs.Mp3Encoder === 'function')) {
    throw new Error('lamejs MP3 encoder failed to load');
  }

  log('lamejs encoder loaded');
  return true;
}

async function getMic() {
  if (stream) return stream;
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return stream;
}

function setupVu(s) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(s);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  dataArray = new Uint8Array(analyser.frequencyBinCount);
  source.connect(analyser);

  const loop = () => {
    analyser.getByteTimeDomainData(dataArray);
    let sum = 0;
    for (let i=0;i<dataArray.length;i++) {
      const v = (dataArray[i]-128)/128;
      sum += v*v;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    vuEl.style.width = `${Math.min(100, Math.round(rms*160))}%`;
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
}
function teardownVu() {
  cancelAnimationFrame(raf); raf = null;
  if (audioCtx) audioCtx.close().catch(()=>{});
  audioCtx = null;
  vuEl.style.width = '0%';
}

async function handleCallClick() {
  if (!callActive) {
    try {
      await ensureLameEncoder();
      await getMic();
      setupVu(stream);

      // Prepare one recorder for the whole call
      const mime = MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm';
      mediaRecorder = new MediaRecorder(stream, { mimeType: mime });

      chunks = [];
      mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      mediaRecorder.onstart = () => log('Recorder started');
      mediaRecorder.onpause = () => log('Recorder paused');
      mediaRecorder.onresume = () => log('Recorder resumed');
      
      // Handle upload when recording stops
      mediaRecorder.onstop = async () => {
        log('Recorder stopped, processing audio...');
        setStatus('Processing audio...');
        
        try {
          const webmBlob = new Blob(chunks, { type: mime });
          const mp3Blob = await toMp3(webmBlob);
          
          const formData = new FormData();
          const ts = new Date().toISOString().replace(/[:.]/g,'-');
          formData.append('file', mp3Blob, `caller-${ts}.mp3`);
          
          setStatus('Uploading...');
          const response = await fetch('http://127.0.0.1:5000/upload', { 
            method: 'POST', 
            body: formData 
          });
          
          if (response.ok) {
            const result = await response.json();
            setStatus('Upload complete!');
            log('Upload successful: ' + JSON.stringify(result));
            
            if (result.transcript) {
              log('Transcript: ' + result.transcript);
              const transcriptEl = document.getElementById('transcript');
              if (transcriptEl) transcriptEl.textContent = result.transcript;
            }
            
            if (result.file_path) {
              const audioUrl = `http://127.0.0.1:5000/uploads/${result.file_path}`;
              
              try {
                const audioResponse = await fetch(audioUrl);
                const audioBlob = await audioResponse.blob();
                const blobUrl = URL.createObjectURL(audioBlob);
                
                const audio = new Audio(blobUrl);
                audio.play();
                
                const link = document.createElement('a');
                link.href = blobUrl;
                link.textContent = 'Download Audio';
                link.download = result.file_path;
                document.body.appendChild(link);
                
                log('Audio playing from blob URL');
              } catch (audioErr) {
                log('Error loading audio: ' + audioErr.message);
              }
            }
          } else {
            const result = await response.json().catch(() => ({}));
            setStatus('Upload failed: ' + (result.error || response.statusText));
            log('Upload failed: ' + JSON.stringify(result));
          }
        } catch (err) {
          setStatus('Processing error: ' + err.message);
          log('Processing error: ' + err.message);
        }
      };

      mediaRecorder.start();

      callActive = true;
      talkBtn.disabled = false;
      callBtn.textContent = 'End Call';
      callBtn.setAttribute('aria-pressed', 'true');
      setStatus('Connected. Hold to talk.');
      startTimer();
    } catch (e) {
      setStatus('Error: ' + e.message);
      log('Error: ' + e.message);
    }
  } else {
    // Hang up: stop recorder and clean up
    try { 
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop(); 
      }
    } catch {}
    
    try { 
      if (stream) {
        stream.getTracks().forEach(t => t.stop()); 
      }
    } catch {}
    
    stream = null;
    talkBtn.disabled = true;
    callBtn.textContent = 'Start Call';
    callBtn.setAttribute('aria-pressed', 'false');
    stopTimer();
    teardownVu();
    callActive = false;
    
    // Status will be updated by onstop handler
  }
}

async function toMp3(webmBlob, bitrate = '64k') {
  log('Converting audio to MP3...');
  
  const kbps = parseInt(bitrate, 10) || 64;

  const arrayBuffer = await webmBlob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  let audioBuffer;
  try {
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  } catch (err) {
    log('decodeAudioData failed: ' + err.message);
    throw new Error('Could not decode recorded audio in this browser');
  }

  const channelData = audioBuffer.numberOfChannels > 0
    ? audioBuffer.getChannelData(0)
    : new Float32Array(audioBuffer.length);

  const samples = new Int16Array(channelData.length);
  for (let i = 0; i < channelData.length; i++) {
    let s = Math.max(-1, Math.min(1, channelData[i]));
    samples[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  const Mp3Encoder = window.lamejs && window.lamejs.Mp3Encoder;
  if (!Mp3Encoder) throw new Error('Mp3Encoder not available');

  const sampleRate = audioBuffer.sampleRate || 44100;
  const mp3encoder = new Mp3Encoder(1, sampleRate, kbps);

  const blockSize = 1152;
  const mp3Chunks = [];
  for (let i = 0; i < samples.length; i += blockSize) {
    const slice = samples.subarray(i, i + blockSize);
    const mp3buf = mp3encoder.encodeBuffer(slice);
    if (mp3buf.length > 0) mp3Chunks.push(new Uint8Array(mp3buf));
  }
  const endBuf = mp3encoder.flush();
  if (endBuf.length > 0) mp3Chunks.push(new Uint8Array(endBuf));

  let totalLen = 0;
  for (const c of mp3Chunks) totalLen += c.length;
  const mp3Data = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of mp3Chunks) {
    mp3Data.set(c, offset);
    offset += c.length;
  }

  log('MP3 bytes: ' + mp3Data.length);
  return new Blob([mp3Data], { type: 'audio/mpeg' });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing UI');
    initializeUI();
  });
} else {
  console.log('DOM already loaded, initializing UI');
  initializeUI();
}