// Global variables
let socket;
let videoFeedInterval;
let isConnected = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSocketIO();
    initializeEventListeners();
    updateConfigDisplay();
});

// Initialize Socket.IO connection
function initializeSocketIO() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        isConnected = true;
        showMessage('Connected to server', 'success');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        isConnected = false;
        showMessage('Disconnected from server', 'error');
    });
    
    socket.on('detection_update', function(data) {
        updateDetections(data.detections);
        updateTracking(data.tracking);
        updateStats(data.stats);
    });
    
    socket.on('status', function(data) {
        showMessage(data.message, 'success');
    });
}

// Initialize event listeners
function initializeEventListeners() {
    // Camera controls
    document.getElementById('startCamera').addEventListener('click', startCamera);
    document.getElementById('stopCamera').addEventListener('click', stopCamera);
    document.getElementById('startDetection').addEventListener('click', startDetection);
    document.getElementById('stopDetection').addEventListener('click', stopDetection);
    
    // Configuration
    document.getElementById('confidenceSlider').addEventListener('input', updateConfigDisplay);
    document.getElementById('updateConfig').addEventListener('click', updateConfig);
    
    // Database
    document.getElementById('testDb').addEventListener('click', testDatabase);
    document.getElementById('loadRecords').addEventListener('click', loadDatabaseRecords);
}

// API call helper function
async function apiCall(endpoint, method = 'POST', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        const result = await response.json();
        
        if (result.success) {
            showMessage(result.message, 'success');
        } else {
            showMessage(result.message, 'error');
        }
        
        return result;
    } catch (error) {
        showMessage('Network error: ' + error.message, 'error');
        return { success: false, message: error.message };
    }
}

// Camera control functions
async function startCamera() {
    const result = await apiCall('/api/start_camera');
    if (result.success) {
        startVideoFeed();
    }
}

async function stopCamera() {
    const result = await apiCall('/api/stop_camera');
    if (result.success) {
        stopVideoFeed();
    }
}

async function startDetection() {
    await apiCall('/api/start_detection');
}

async function stopDetection() {
    await apiCall('/api/stop_detection');
}

// Video feed functions
function startVideoFeed() {
    const videoContainer = document.getElementById('videoContainer');
    const placeholder = document.getElementById('videoPlaceholder');
    
    // Remove placeholder
    if (placeholder) {
        placeholder.remove();
    }
    
    // Create video element
    let videoElement = document.getElementById('videoFeed');
    if (!videoElement) {
        videoElement = document.createElement('img');
        videoElement.id = 'videoFeed';
        videoElement.style.width = '100%';
        videoElement.style.height = '100%';
        videoElement.style.objectFit = 'contain';
        videoContainer.appendChild(videoElement);
    }
    
    // Start updating video feed
    updateVideoFeed();
    videoFeedInterval = setInterval(updateVideoFeed, 100); // Update every 100ms
}

function stopVideoFeed() {
    if (videoFeedInterval) {
        clearInterval(videoFeedInterval);
        videoFeedInterval = null;
    }
    
    const videoElement = document.getElementById('videoFeed');
    const videoContainer = document.getElementById('videoContainer');
    
    if (videoElement) {
        videoElement.remove();
    }
    
    // Add placeholder back
    const placeholder = document.createElement('div');
    placeholder.id = 'videoPlaceholder';
    placeholder.textContent = 'Camera stopped';
    placeholder.style.position = 'absolute';
    placeholder.style.top = '50%';
    placeholder.style.left = '50%';
    placeholder.style.transform = 'translate(-50%, -50%)';
    placeholder.style.color = 'white';
    placeholder.style.fontSize = '18px';
    placeholder.style.textAlign = 'center';
    
    videoContainer.appendChild(placeholder);
}

async function updateVideoFeed() {
    try {
        const response = await fetch('/video_feed');
        const frameData = await response.text();
        
        if (frameData && frameData.trim()) {
            const videoElement = document.getElementById('videoFeed');
            if (videoElement) {
                // The response should be just the base64 data
                videoElement.src = frameData.trim();
            }
        }
    } catch (error) {
        console.log('Video feed error:', error);
    }
}

// Configuration functions
function updateConfigDisplay() {
    const slider = document.getElementById('confidenceSlider');
    const display = document.getElementById('confidenceValue');
    display.textContent = slider.value;
}

async function updateConfig() {
    const confidence = document.getElementById('confidenceSlider').value;
    const threshold = document.getElementById('exitThreshold').value;
    
    await apiCall('/api/set_config', 'POST', {
        confidence: parseFloat(confidence),
        threshold: parseInt(threshold)
    });
}

// Database functions
async function testDatabase() {
    const dbConfig = {
        host: document.getElementById('dbHost').value || 'localhost',
        user: document.getElementById('dbUser').value || 'root',
        password: document.getElementById('dbPassword').value || '1234',
        database: document.getElementById('dbName').value || 'nrkindex_api'
    };
    
    await apiCall('/api/test_db', 'POST', dbConfig);
}

async function loadDatabaseRecords() {
    try {
        const response = await fetch('/api/get_db_records');
        const result = await response.json();
        
        if (result.success) {
            displayDatabaseRecords(result.data);
            showMessage('Database records loaded successfully', 'success');
        } else {
            showMessage(result.message, 'error');
        }
    } catch (error) {
        showMessage('Failed to load database records: ' + error.message, 'error');
    }
}

function displayDatabaseRecords(records) {
    const tbody = document.querySelector('#recordsTable tbody');
    tbody.innerHTML = '';
    
    if (records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6">No records found</td></tr>';
        return;
    }
    
    records.forEach(record => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${record.id}</td>
            <td>${record.objname}</td>
            <td>${record.dateandtime}</td>
            <td>${record.entrytime || '-'}</td>
            <td>${record.exittime || '-'}</td>
            <td><span class="status-${record.status}">${record.status.toUpperCase()}</span></td>
        `;
        tbody.appendChild(row);
    });
}

// Update display functions
function updateDetections(detections) {
    const container = document.getElementById('detectionsList');
    
    if (!detections || detections.length === 0) {
        container.innerHTML = '<p>No detections</p>';
        return;
    }
    
    let html = '';
    detections.forEach(detection => {
        html += `
            <div class="detection-item">
                <h4>${detection.class_name}</h4>
                <p>Confidence: ${(detection.confidence * 100).toFixed(1)}%</p>
                <p>Position: [${detection.bbox[0]}, ${detection.bbox[1]}, ${detection.bbox[2]}, ${detection.bbox[3]}]</p>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateTracking(tracking) {
    const container = document.getElementById('trackingList');
    
    if (!tracking || tracking.length === 0) {
        container.innerHTML = '<p>No objects being tracked</p>';
        return;
    }
    
    let html = '';
    tracking.forEach(item => {
        html += `
            <div class="tracking-item">
                <h4>${item.class_name}</h4>
                <p>Status: ${item.status}</p>
                <p>Duration: ${item.duration} seconds</p>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateStats(stats) {
    if (stats) {
        document.getElementById('totalEntries').textContent = stats.total_entries || 0;
        document.getElementById('totalExits').textContent = stats.total_exits || 0;
        document.getElementById('presentObjects').textContent = stats.present_objects || 0;
    }
}

// Utility functions
function showMessage(message, type = 'info') {
    const container = document.getElementById('statusMessages');
    const timestamp = new Date().toLocaleTimeString();
    
    // Clear previous message classes
    container.className = 'status-messages';
    
    // Add appropriate class
    if (type === 'success') {
        container.classList.add('status-success');
    } else if (type === 'error') {
        container.classList.add('status-error');
    }
    
    container.innerHTML = `[${timestamp}] ${message}`;
    
    // Auto-clear after 5 seconds
    setTimeout(() => {
        if (container.innerHTML === `[${timestamp}] ${message}`) {
            container.innerHTML = '';
            container.className = 'status-messages';
        }
    }, 8080);
}

// Auto-refresh database records every 30 seconds
setInterval(() => {
    const tbody = document.querySelector('#recordsTable tbody');
    if (tbody && tbody.innerHTML !== '<tr><td colspan="6">No records loaded</td></tr>') {
        loadDatabaseRecords();
    }
}, 30000);