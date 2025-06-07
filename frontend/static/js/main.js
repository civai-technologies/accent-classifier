// Accent Classifier - Frontend JavaScript

class AccentClassifier {
    constructor() {
        this.form = document.getElementById('accentForm');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = this.progressContainer.querySelector('.progress-bar');
        this.progressText = document.getElementById('progressText');
        this.resultsContent = document.getElementById('resultsContent');
        this.noResults = document.getElementById('noResults');
        
        this.initializeEventListeners();
        this.setupFileUpload();
        this.setupDynamicPlaceholder();
    }

    initializeEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        
        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => this.resetForm());
        
        // Download results button
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadResults());
        
        // Tab switching
        document.querySelectorAll('#inputTabs button[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => this.handleTabSwitch(e));
        });
        
        // URL input validation
        document.getElementById('urlInput').addEventListener('input', (e) => this.validateUrl(e));
        

        
        // Media preview buttons
        this.setupMediaPreview();
    }

    setupDynamicPlaceholder() {
        // Get current host and protocol
        const protocol = window.location.protocol;
        const host = window.location.host;
        
        // Construct the dynamic URL for the test MP4
        const dynamicUrl = `${protocol}//${host}/static/american_test.mp4`;
        
        // Set the placeholder
        const urlInput = document.getElementById('urlInput');
        urlInput.placeholder = dynamicUrl;
    }

    setupFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const filePanel = document.getElementById('file-panel');
        
        // File input change
        fileInput.addEventListener('change', (e) => this.handleFileSelection(e));
        
        // Drag and drop functionality
        filePanel.addEventListener('dragover', (e) => {
            e.preventDefault();
            filePanel.classList.add('dragover');
        });
        
        filePanel.addEventListener('dragleave', (e) => {
            e.preventDefault();
            filePanel.classList.remove('dragover');
        });
        
        filePanel.addEventListener('drop', (e) => {
            e.preventDefault();
            filePanel.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.handleFileSelection({ target: fileInput });
            }
        });
    }

    handleFileSelection(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Validate file size (100MB limit)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 100MB');
            event.target.value = '';
            return;
        }
        
        // Validate file type
        const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'video/mp4', 'video/quicktime', 'video/x-msvideo'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['mp3', 'wav', 'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4a'];
        
        if (!allowedExtensions.includes(fileExtension)) {
            this.showError('Invalid file type. Please upload an audio or video file.');
            event.target.value = '';
            return;
        }
        
        // Show file info
        const fileInfo = `Selected: ${file.name} (${this.formatFileSize(file.size)})`;
        this.showFileInfo(fileInfo);
    }

    validateUrl(event) {
        const url = event.target.value.trim();
        const urlInput = event.target;
        
        if (!url) {
            urlInput.classList.remove('is-valid', 'is-invalid');
            return;
        }
        
        try {
            new URL(url);
            // Check if URL looks like a video/audio file
            const supportedExtensions = ['.mp4', '.mp3', '.wav', '.mov', '.avi', '.mkv', '.webm', '.m4a'];
            const hasValidExtension = supportedExtensions.some(ext => url.toLowerCase().includes(ext));
            const isVideoService = ['youtube.com', 'youtu.be', 'loom.com', 'vimeo.com'].some(service => url.includes(service));
            
            if (hasValidExtension || isVideoService) {
                urlInput.classList.remove('is-invalid');
                urlInput.classList.add('is-valid');
            } else {
                urlInput.classList.remove('is-valid');
                urlInput.classList.add('is-invalid');
            }
        } catch {
            urlInput.classList.remove('is-valid');
            urlInput.classList.add('is-invalid');
        }
    }

    handleTabSwitch(event) {
        // Reset form validation states when switching tabs
        document.querySelectorAll('.form-control').forEach(input => {
            input.classList.remove('is-valid', 'is-invalid');
        });
        
        // Clear any file info or errors
        this.clearMessages();
    }

    async handleSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData();
        const activeTab = document.querySelector('#inputTabs .nav-link.active').id;
        
        // Determine which input method is being used
        switch (activeTab) {
            case 'file-tab':
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) {
                    this.showError('Please select a file to analyze.');
                    return;
                }
                formData.append('file', fileInput.files[0]);
                break;
                
            case 'url-tab':
                const urlInput = document.getElementById('urlInput');
                let url = urlInput.value.trim();
                
                // If empty, use the placeholder URL
                if (!url) {
                    url = urlInput.placeholder;
                    urlInput.value = url; // Update the input field to show the URL being used
                }
                
                if (!url) {
                    this.showError('Please enter a valid URL.');
                    return;
                }
                formData.append('url', url);
                break;
                
            case 'demo-tab':
                formData.append('use-default', 'true');
                break;
                
            default:
                this.showError('Please select an input method.');
                return;
        }
        
        // Start analysis
        await this.analyzeAudio(formData);
    }

    setupMediaPreview() {
        // Preview buttons (video + audio)
        document.querySelectorAll('[data-preview]').forEach(button => {
            console.log('Found preview button:', button, button.dataset);
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const sampleType = e.currentTarget.dataset.sample;
                const previewType = e.currentTarget.dataset.preview;
                console.log('Preview button clicked:', { sampleType, previewType });
                this.showMediaPreview(sampleType, previewType, 'video');
            });
        });
        
        // Audio-only buttons
        document.querySelectorAll('[data-audio]').forEach(button => {
            console.log('Found audio button:', button, button.dataset);
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const audioType = e.currentTarget.dataset.audio;
                const sampleType = audioType.split('_')[0]; // Extract accent type
                console.log('Audio button clicked:', { audioType, sampleType });
                this.showMediaPreview(sampleType, audioType, 'audio');
            });
        });
        
        // Analyze selected sample button
        document.getElementById('analyzeSelectedSample').addEventListener('click', () => {
            const sampleType = this.currentSample;
            if (sampleType) {
                this.testAccentSample(sampleType);
            }
        });
        
        // Clear preview button
        document.getElementById('clearPreview').addEventListener('click', () => {
            this.clearMediaPreview();
        });
    }

    showMediaPreview(sampleType, mediaType, playerType) {
        const mediaPreviewCard = document.getElementById('mediaPreviewCard');
        const previewTitle = document.getElementById('previewTitle');
        const videoPlayer = document.getElementById('videoPlayer');
        const audioPlayer = document.getElementById('audioPlayer');
        
        console.log('showMediaPreview called:', { sampleType, mediaType, playerType });
        
        // Store current sample type for analysis
        this.currentSample = sampleType;
        
        // Set title
        const titles = {
            'american': 'American Accent Sample',
            'british': 'British Accent Sample', 
            'french': 'French Accent Sample'
        };
        previewTitle.textContent = titles[sampleType] || 'Sample Preview';
        
        // Clear previous event handlers and sources
        videoPlayer.onloadstart = null;
        videoPlayer.oncanplay = null;
        videoPlayer.onerror = null;
        audioPlayer.onerror = null;
        
        // Hide both players first
        videoPlayer.style.display = 'none';
        audioPlayer.style.display = 'none';
        
        // Set up the appropriate player
        const mediaUrl = `/test-media/${mediaType}`;
        console.log('Setting media URL:', mediaUrl);
        
        if (playerType === 'video') {
            // Clear audio player completely
            audioPlayer.src = '';
            audioPlayer.load();
            
            // Set up video player
            videoPlayer.src = mediaUrl;
            videoPlayer.load(); // Force reload
            videoPlayer.style.display = 'block';
            console.log('Video player shown with src:', videoPlayer.src);
            
            // Add error handling only for video
            videoPlayer.onerror = function(e) {
                console.error('Video error:', e, videoPlayer.error);
                alert('Error loading video. Please check the console for details.');
            };
            
            videoPlayer.onloadstart = function() {
                console.log('Video load started');
            };
            
            videoPlayer.oncanplay = function() {
                console.log('Video can play');
            };
        } else {
            // Clear video player completely
            videoPlayer.src = '';
            videoPlayer.load();
            
            // Set up audio player
            audioPlayer.src = mediaUrl;
            audioPlayer.load(); // Force reload
            audioPlayer.style.display = 'block';
            console.log('Audio player shown with src:', audioPlayer.src);
            
            // Add error handling only for audio
            audioPlayer.onerror = function(e) {
                console.error('Audio error:', e, audioPlayer.error);
                alert('Error loading audio. Please check the console for details.');
            };
        }
        
        // Show the preview card
        mediaPreviewCard.style.display = 'block';
        
        // Scroll to preview
        mediaPreviewCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    clearMediaPreview() {
        const mediaPreviewCard = document.getElementById('mediaPreviewCard');
        const videoPlayer = document.getElementById('videoPlayer');
        const audioPlayer = document.getElementById('audioPlayer');
        
        // Hide preview card
        mediaPreviewCard.style.display = 'none';
        
        // Clear event handlers
        videoPlayer.onloadstart = null;
        videoPlayer.oncanplay = null;
        videoPlayer.onerror = null;
        audioPlayer.onerror = null;
        
        // Clear media sources
        videoPlayer.src = '';
        audioPlayer.src = '';
        videoPlayer.load();
        audioPlayer.load();
        
        // Clear current sample
        this.currentSample = null;
    }

    async testAccentSample(sampleType) {
        const formData = new FormData();
        formData.append('test-sample', sampleType);
        
        // Start analysis
        await this.analyzeAudio(formData);
    }

    async analyzeAudio(formData) {
        try {
            this.showProgress('Uploading and processing audio...');
            this.disableForm();
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            this.displayResults(result);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.hideProgress();
            this.enableForm();
        }
    }

    displayResults(result) {
        // Hide no results message
        this.noResults.style.display = 'none';
        
        // Update accent name and icon
        const accentName = document.getElementById('accentName');
        const accentBadge = document.getElementById('accentBadge');
        const accentIcon = accentBadge.querySelector('i');
        
        accentName.textContent = result.accent;
        
        // Set appropriate icon based on accent
        this.setAccentIcon(accentIcon, result.accent);
        
        // Update confidence
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceBadge = document.getElementById('confidenceBadge');
        
        confidenceBar.style.width = `${result.confidence}%`;
        confidenceBar.textContent = `${result.confidence}%`;
        confidenceBar.setAttribute('aria-valuenow', result.confidence);
        
        confidenceBadge.textContent = result.confidence_level;
        this.setConfidenceBadgeColor(confidenceBadge, result.confidence);
        
        // Update summary
        document.getElementById('summaryText').textContent = result.summary;
        
        // Display all predictions
        this.displayAllPredictions(result.all_predictions, result.accent);
        
        // Store results for download
        this.currentResults = result;
        
        // Show results with animation
        this.resultsContent.style.display = 'block';
        this.resultsContent.classList.add('fade-in');
        
        // Scroll to results
        this.resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    setAccentIcon(iconElement, accent) {
        const accentIcons = {
            'american': 'fas fa-flag-usa',
            'british': 'fas fa-flag',
            'australian': 'fas fa-globe-oceania',
            'canadian': 'fas fa-leaf',
            'irish': 'fas fa-clover',
            'scottish': 'fas fa-mountain',
            'indian': 'fas fa-globe-asia',
            'german': 'fas fa-beer',
            'french': 'fas fa-wine-glass',
            'spanish': 'fas fa-guitar',
            'russian': 'fas fa-snowflake',
            'chinese': 'fas fa-yin-yang',
            'japanese': 'fas fa-torii-gate'
        };
        
        const iconClass = accentIcons[accent.toLowerCase()] || 'fas fa-globe-americas';
        iconElement.className = iconClass;
    }

    setConfidenceBadgeColor(badgeElement, confidence) {
        // Remove existing color classes
        badgeElement.classList.remove('bg-success', 'bg-warning', 'bg-danger', 'bg-info', 'bg-secondary');
        
        if (confidence >= 90) {
            badgeElement.classList.add('bg-success');
        } else if (confidence >= 80) {
            badgeElement.classList.add('bg-info');
        } else if (confidence >= 60) {
            badgeElement.classList.add('bg-warning');
        } else {
            badgeElement.classList.add('bg-danger');
        }
    }

    displayAllPredictions(predictions, topAccent) {
        const container = document.getElementById('allPredictions');
        container.innerHTML = '';
        
        if (!predictions || Object.keys(predictions).length === 0) {
            container.innerHTML = '<p class="text-muted">No detailed predictions available.</p>';
            return;
        }
        
        // Sort predictions by confidence
        const sortedPredictions = Object.entries(predictions)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5); // Show top 5
        
        sortedPredictions.forEach(([accent, confidence]) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            
            if (accent === topAccent) {
                item.classList.add('top-prediction');
            }
            
            const percentage = Math.round(confidence * 100);
            
            item.innerHTML = `
                <span class="prediction-name">${accent}</span>
                <span class="prediction-score">${percentage}%</span>
            `;
            
            container.appendChild(item);
        });
    }

    downloadResults() {
        if (!this.currentResults) {
            this.showError('No results to download.');
            return;
        }
        
        const data = {
            accent: this.currentResults.accent,
            confidence: this.currentResults.confidence,
            confidence_level: this.currentResults.confidence_level,
            summary: this.currentResults.summary,
            all_predictions: this.currentResults.all_predictions,
            timestamp: new Date().toISOString(),
            analysis_type: 'accent_classification'
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `accent_analysis_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    resetForm() {
        // Clear form
        this.form.reset();
        
        // Clear validation states
        document.querySelectorAll('.form-control').forEach(input => {
            input.classList.remove('is-valid', 'is-invalid');
        });
        
        // Hide results
        this.resultsContent.style.display = 'none';
        this.noResults.style.display = 'block';
        
        // Clear media preview
        if (this.clearMediaPreview) {
            this.clearMediaPreview();
        }
        
        // Clear messages
        this.clearMessages();
        
        // Reset to first tab
        document.getElementById('file-tab').click();
        
        // Clear stored results
        this.currentResults = null;
    }

    showProgress(message = 'Processing...') {
        this.progressText.textContent = message;
        this.progressContainer.style.display = 'block';
        
        // Animate progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            this.progressBar.style.width = `${progress}%`;
        }, 200);
        
        this.progressInterval = interval;
    }

    hideProgress() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        // Complete the progress bar
        this.progressBar.style.width = '100%';
        
        setTimeout(() => {
            this.progressContainer.style.display = 'none';
            this.progressBar.style.width = '0%';
        }, 500);
    }

    disableForm() {
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.innerHTML = '<span class="loading-spinner me-2"></span>Analyzing...';
        
        document.querySelectorAll('input, button').forEach(element => {
            if (element !== this.analyzeBtn) {
                element.disabled = true;
            }
        });
    }

    enableForm() {
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.innerHTML = '<i class="fas fa-brain me-2"></i>Analyze Accent';
        
        document.querySelectorAll('input, button').forEach(element => {
            element.disabled = false;
        });
    }

    showError(message) {
        this.clearMessages();
        
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alert.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        this.form.appendChild(alert);
    }

    showFileInfo(message) {
        this.clearMessages();
        
        const alert = document.createElement('div');
        alert.className = 'alert alert-info fade show mt-3';
        alert.innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            ${message}
        `;
        
        this.form.appendChild(alert);
    }

    clearMessages() {
        this.form.querySelectorAll('.alert').forEach(alert => alert.remove());
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AccentClassifier();
    
    // Health check
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'healthy') {
                console.warn('Application health check failed:', data);
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
        });
});

// Add some utility functions for enhanced UX
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+Enter or Cmd+Enter to analyze
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById('analyzeBtn').click();
        }
        
        // Escape to reset
        if (e.key === 'Escape') {
            const resetBtn = document.getElementById('resetBtn');
            if (resetBtn && !resetBtn.disabled) {
                resetBtn.click();
            }
        }
    });
    
    // Add tooltips to buttons
    const tooltips = {
        'analyzeBtn': 'Keyboard shortcut: Ctrl+Enter',
        'resetBtn': 'Keyboard shortcut: Escape',
        'downloadBtn': 'Download analysis results as JSON'
    };
    
    Object.entries(tooltips).forEach(([id, title]) => {
        const element = document.getElementById(id);
        if (element) {
            element.setAttribute('title', title);
            element.setAttribute('data-bs-toggle', 'tooltip');
        }
    });
    
    // Initialize Bootstrap tooltips
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}); 