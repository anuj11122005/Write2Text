
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadPrompt = document.getElementById('upload-prompt');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultContainer = document.getElementById('result-container');
    const predictionText = document.getElementById('prediction');
    const loader = document.getElementById('loader');
    const btnText = document.getElementById('btn-text');

    const beamSearchToggle = document.getElementById('beam_search_toggle');
    const modelEpoch = document.getElementById('model-epoch');

    // Fetch model info on load
    async function fetchModelInfo() {
        try {
            const response = await fetch('/model_info');
            const data = await response.json();
            modelEpoch.innerText = data.epoch;
            // Optionally update title or other info with word_acc/cer
        } catch (e) {
            console.error('Could not fetch model info', e);
        }
    }
    fetchModelInfo();

    let selectedFile = null;

    // Trigger file input on click
    dropZone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    // Drag and drop handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragging');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragging');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
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
                resultContainer.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    }

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        imagePreview.style.display = 'none';
        uploadPrompt.style.display = 'block';
        predictBtn.disabled = true;
        clearBtn.style.display = 'none';
        resultContainer.style.display = 'none';
    });

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading state
        predictBtn.disabled = true;
        loader.style.display = 'inline-block';
        btnText.innerText = 'Analyzing...';
        resultContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('beam_search', beamSearchToggle.checked);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                predictionText.innerText = data.prediction;
                resultContainer.style.display = 'block';
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction.');
        } finally {
            predictBtn.disabled = false;
            loader.style.display = 'none';
            btnText.innerText = 'Predict Word';
        }
    });
});
