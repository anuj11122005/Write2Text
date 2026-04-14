
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadPrompt = document.getElementById('upload-prompt');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultArea = document.getElementById('result-area');
    const predictionText = document.getElementById('prediction');
    const confVal = document.getElementById('conf-val');
    const timeVal = document.getElementById('time-val');
    const loader = document.getElementById('loader');
    const btnText = document.getElementById('btn-text');
    const beamSearchToggle = document.getElementById('beam_search_toggle');
    const modelEpoch = document.getElementById('model-epoch');
    const historyList = document.getElementById('history-list');

    let history = [];

    // Initialize Model Info
    async function fetchModelInfo() {
        try {
            const response = await fetch('/model_info');
            const data = await response.json();
            modelEpoch.innerText = data.epoch;
        } catch (e) { console.error(e); }
    }
    fetchModelInfo();

    let selectedFile = null;

    dropZone.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });

    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--primary)'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'var(--glass-border)'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadPrompt.style.display = 'none';
                predictBtn.disabled = false;
                clearBtn.style.display = 'block';
                resultArea.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    }

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.style.display = 'none';
        uploadPrompt.style.display = 'block';
        predictBtn.disabled = true;
        clearBtn.style.display = 'none';
        resultArea.style.display = 'none';
        dropZone.style.borderColor = 'var(--glass-border)';
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        predictBtn.disabled = true;
        loader.style.display = 'inline-block';
        btnText.innerText = 'Analyzing Neural Path...';

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('beam_search', beamSearchToggle.checked);

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.status === 'success') {
                displayResult(data);
                addToHistory(data.prediction, data.confidence, imagePreview.src);
            } else {
                alert('Analysis Error: ' + data.error);
            }
        } catch (error) {
            console.error(error);
            alert('Critial Neural Link Failure.');
        } finally {
            predictBtn.disabled = false;
            loader.style.display = 'none';
            btnText.innerText = 'Execute Prediction';
        }
    });

    function displayResult(data) {
        predictionText.innerText = data.prediction || "[NULL]";
        confVal.innerText = data.confidence + "%";
        timeVal.innerText = data.time + "ms";
        resultArea.style.display = 'block';
        
        // Dynamic color based on confidence
        if (data.confidence > 80) confVal.style.color = '#10b981';
        else if (data.confidence > 50) confVal.style.color = '#f59e0b';
        else confVal.style.color = '#ef4444';
    }

    function addToHistory(word, conf, imgSrc) {
        if (history.length === 0) historyList.innerHTML = '';
        
        const item = { word, conf, time: new Date().toLocaleTimeString() };
        history.unshift(item);
        if (history.length > 10) history.pop();

        const html = `
            <div class="history-item">
                <img src="${imgSrc}" class="history-thumb">
                <div class="history-info">
                    <div class="history-word">${word}</div>
                    <div class="history-meta">${item.time} • ${conf}% Match</div>
                </div>
            </div>
        `;
        historyList.insertAdjacentHTML('afterbegin', html);
    }
});
