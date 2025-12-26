// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// UI elements
const clearBtn = document.getElementById('clearBtn');
const recognizeBtn = document.getElementById('recognizeBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');

// Initialize canvas
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

// Get mouse/touch position relative to canvas
function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

// Start drawing
function startDrawing(e) {
    isDrawing = true;
    const pos = getPosition(e);
    lastX = pos.x;
    lastY = pos.y;
    e.preventDefault();
}

// Draw on canvas
function draw(e) {
    if (!isDrawing) return;

    const pos = getPosition(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    lastX = pos.x;
    lastY = pos.y;

    e.preventDefault();
}

// Stop drawing
function stopDrawing() {
    isDrawing = false;
}

// Clear canvas
function clearCanvas() {
    initCanvas();
    hideAllResults();
}

// Hide all result sections
function hideAllResults() {
    loading.classList.add('hidden');
    error.classList.add('hidden');
    document.getElementById('resultsTabs').classList.add('hidden');
}

// Show error message
function showError(message) {
    hideAllResults();
    error.classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message;
}

// Tab switching functionality
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Remove active class from all buttons and panes
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });

            // Add active class to clicked button and corresponding pane
            btn.classList.add('active');
            if (targetTab === 'prediction') {
                document.getElementById('predictionTab').classList.add('active');
            } else if (targetTab === 'activations') {
                document.getElementById('activationsTab').classList.add('active');
            }
        });
    });
}

// Display prediction results
function displayResults(data) {
    hideAllResults();

    // Show tabs container
    document.getElementById('resultsTabs').classList.remove('hidden');

    // Update prediction data
    document.getElementById('predictedDigit').textContent = data.digit;
    document.getElementById('confidence').textContent =
        `${(data.confidence * 100).toFixed(2)}%`;

    // Show probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';

    // Sort probabilities by digit
    const sortedProbs = Object.entries(data.probabilities)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

    sortedProbs.forEach(([digit, prob]) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'probability-bar';

        const label = document.createElement('div');
        label.className = 'probability-label';
        label.textContent = digit;

        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar-container';

        const barFill = document.createElement('div');
        barFill.className = 'probability-bar-fill';
        const percentage = (prob * 100).toFixed(1);
        barFill.style.width = `${percentage}%`;

        const valueInside = document.createElement('span');
        valueInside.className = 'probability-value';
        valueInside.textContent = percentage >= 25 ? `${percentage}%` : '';

        const valueOutside = document.createElement('span');
        valueOutside.className = 'probability-value-outside';
        valueOutside.textContent = `${percentage}%`;

        barFill.appendChild(valueInside);
        barContainer.appendChild(barFill);
        barDiv.appendChild(label);
        barDiv.appendChild(barContainer);
        barDiv.appendChild(valueOutside);
        probabilityBars.appendChild(barDiv);
    });

    // Display neural network visualization if available
    if (data.network_activations) {
        displayNetworkVisualization(data.network_activations);
    }
}

// Color mapping function - converts activation value to color
function getActivationColor(activation) {
    // Use a gradient from dark blue (low) to bright yellow (high)
    // Normalize activation (typically 0-1 for ReLU, but can vary)
    const normalized = Math.max(0, Math.min(1, activation));

    if (normalized < 0.01) {
        // Very low/inactive - dark gray
        return `rgb(50, 50, 50)`;
    } else if (normalized < 0.3) {
        // Low activation - blue to purple
        const intensity = normalized / 0.3;
        return `rgb(${Math.floor(100 * intensity)}, ${Math.floor(50 * intensity)}, ${Math.floor(150 + 105 * intensity)})`;
    } else if (normalized < 0.7) {
        // Medium activation - purple to orange
        const intensity = (normalized - 0.3) / 0.4;
        return `rgb(${Math.floor(100 + 155 * intensity)}, ${Math.floor(50 + 100 * intensity)}, ${Math.floor(255 - 155 * intensity)})`;
    } else {
        // High activation - orange to bright yellow
        const intensity = (normalized - 0.7) / 0.3;
        return `rgb(255, ${Math.floor(150 + 105 * intensity)}, ${Math.floor(100 - 100 * intensity)})`;
    }
}

// Display neural network visualization
function displayNetworkVisualization(networkActivations) {
    const networkCanvas = document.getElementById('networkCanvas');

    if (!networkCanvas) {
        console.error('networkCanvas element not found');
        return;
    }

    networkCanvas.innerHTML = '';

    const container = document.createElement('div');
    container.className = 'network-container';

    networkActivations.forEach(layerData => {
        const layerDiv = document.createElement('div');
        layerDiv.className = 'network-layer';

        const label = document.createElement('div');
        label.className = 'layer-label';
        const layerName = layerData.layer.replace('_', ' ').toUpperCase();
        const layerSize = layerData.activations.length;
        label.textContent = `${layerName} (${layerSize})`;

        const nodesContainer = document.createElement('div');
        nodesContainer.className = 'nodes-container';

        layerData.activations.forEach(activation => {
            const node = document.createElement('div');
            node.className = 'node';
            node.style.backgroundColor = getActivationColor(activation);
            node.setAttribute('data-activation', activation.toFixed(4));
            nodesContainer.appendChild(node);
        });

        layerDiv.appendChild(label);
        layerDiv.appendChild(nodesContainer);
        container.appendChild(layerDiv);
    });

    networkCanvas.appendChild(container);
}

// Send canvas image to server for prediction
async function recognizeDigit() {
    // Check if canvas is empty (all white)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    let isEmpty = true;

    for (let i = 0; i < pixels.length; i += 4) {
        // Check if any pixel is not white
        if (pixels[i] !== 255 || pixels[i + 1] !== 255 || pixels[i + 2] !== 255) {
            isEmpty = false;
            break;
        }
    }

    if (isEmpty) {
        showError('Please draw a digit first!');
        return;
    }

    // Show loading
    hideAllResults();
    loading.classList.remove('hidden');

    try {
        // Convert canvas to base64 image
        const imageBase64 = canvas.toDataURL('image/png');

        // Send to server
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageBase64 })
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (err) {
        showError('Network error: ' + err.message);
    }
}

// Event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for touch
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Button event listeners
clearBtn.addEventListener('click', clearCanvas);
recognizeBtn.addEventListener('click', recognizeDigit);

// Page navigation
function initPageNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetPage = link.getAttribute('data-page');

            // Remove active class from all links and pages
            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));

            // Add active class to clicked link and corresponding page
            link.classList.add('active');
            if (targetPage === 'recognizer') {
                document.getElementById('recognizerPage').classList.add('active');
            } else if (targetPage === 'architecture') {
                document.getElementById('architecturePage').classList.add('active');
            }
        });
    });
}

// Initialize
initCanvas();
initTabs();
initPageNavigation();
