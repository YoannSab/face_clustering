/**
 * Modern Face Clustering Application
 * Enhanced JavaScript with better UX, error handling, and performance
 */

class FaceClusteringApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000/api';
        this.currentTab = 'clustering';
        this.isProcessing = false;
        this.progressInterval = null;
        this.cache = new Map();
        this.clusterData = {}; // Store cluster data including paths
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupTabNavigation();
        this.setupFormValidation();
        this.loadSettings();
    }

    setupEventListeners() {
        // Main form submissions
        document.getElementById('clustering-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startClustering();
        });

        document.getElementById('search-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.performSearch();
        });

        // Algorithm parameter updates
        document.getElementById('algorithm-select').addEventListener('change', (e) => {
            this.updateAlgorithmParams(e.target.value);
        });

        // Range slider updates
        document.getElementById('eps-input').addEventListener('input', (e) => {
            document.getElementById('eps-value').textContent = e.target.value;
        });

        // Directory input changes
        document.getElementById('directory-input').addEventListener('input', 
            this.debounce((e) => this.updateImageCount(e.target.value), 500)
        );

        // Action buttons
        document.getElementById('save-metadata-btn')?.addEventListener('click', () => {
            this.saveMetadata();
        });

        document.getElementById('export-btn')?.addEventListener('click', () => {
            this.exportResults();
        });

        // Modal controls
        document.getElementById('modal-close')?.addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('image-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'image-modal') {
                this.closeModal();
            }
        });

        // View controls for search
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.changeSearchView(e.target.dataset.view);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    setupTabNavigation() {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const tabName = btn.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    setupFormValidation() {
        // Real-time validation for directory paths
        const directoryInputs = ['directory-input', 'search-directory'];
        directoryInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('blur', () => {
                    this.validateDirectoryPath(input);
                });
            }
        });
    }

    loadSettings() {
        // Load user preferences from localStorage
        const savedSettings = localStorage.getItem('faceClusteringSettings');
        if (savedSettings) {
            try {
                const settings = JSON.parse(savedSettings);
                this.applySettings(settings);
            } catch (e) {
                console.warn('Failed to load saved settings:', e);
            }
        }
    }

    saveSettings() {
        const settings = {
            algorithm: document.getElementById('algorithm-select').value,
            eps: document.getElementById('eps-input').value,
            minSamples: document.getElementById('min-samples-input').value,
            clusters: document.getElementById('clusters-input').value,
            lastDirectory: document.getElementById('directory-input').value
        };

        localStorage.setItem('faceClusteringSettings', JSON.stringify(settings));
    }

    applySettings(settings) {
        if (settings.algorithm) {
            document.getElementById('algorithm-select').value = settings.algorithm;
            this.updateAlgorithmParams(settings.algorithm);
        }
        if (settings.eps) {
            document.getElementById('eps-input').value = settings.eps;
            document.getElementById('eps-value').textContent = settings.eps;
        }
        if (settings.minSamples) {
            document.getElementById('min-samples-input').value = settings.minSamples;
        }
        if (settings.clusters) {
            document.getElementById('clusters-input').value = settings.clusters;
        }
        if (settings.lastDirectory) {
            document.getElementById('directory-input').value = settings.lastDirectory;
            this.updateImageCount(settings.lastDirectory);
        }
    }

    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        this.currentTab = tabName;
    }

    updateAlgorithmParams(algorithm) {
        const dbscanParams = document.getElementById('dbscan-params');
        const kmeansParams = document.getElementById('kmeans-params');

        dbscanParams.style.display = algorithm === 'dbscan' ? 'block' : 'none';
        kmeansParams.style.display = ['kmeans', 'hierarchical'].includes(algorithm) ? 'block' : 'none';
    }

    async updateImageCount(directory) {
        if (!directory.trim()) {
            this.setImageCount('');
            return;
        }

        // Check cache first
        if (this.cache.has(directory)) {
            this.setImageCount(this.cache.get(directory));
            return;
        }

        this.setImageCount('Comptage en cours', true);

        try {
            const response = await this.apiCall('/images/count', {
                method: 'POST',
                body: JSON.stringify({ directory })
            });

            const count = response.count;
            this.cache.set(directory, count);
            this.setImageCount(`${count} images trouvées`);
        } catch (error) {
            this.setImageCount('Erreur lors du comptage');
            console.error('Error counting images:', error);
        }
    }

    setImageCount(text, loading = false) {
        const element = document.getElementById('image-count');
        if (element) {
            element.textContent = text;
            element.classList.toggle('loading', loading);
        }
    }

    async startClustering() {
        if (this.isProcessing) return;

        const directory = document.getElementById('directory-input').value.trim();
        if (!directory) {
            this.showToast('Veuillez spécifier un dossier', 'error');
            return;
        }

        this.isProcessing = true;
        this.saveSettings();
        this.showProgress();

        const params = this.getClusteringParams();

        try {
            this.updateProgress(0, 'Initialisation...');
            
            const response = await this.apiCall('/faces/cluster', {
                method: 'POST',
                body: JSON.stringify({
                    directory,
                    ...params
                })
            });

            this.hideProgress();
            this.displayResults(response);
            this.showToast('Analyse terminée avec succès !', 'success');

        } catch (error) {
            this.hideProgress();
            this.showToast(`Erreur: ${error.message}`, 'error');
            console.error('Clustering error:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    getClusteringParams() {
        const algorithm = document.getElementById('algorithm-select').value;
        const params = { algorithm };

        if (algorithm === 'dbscan') {
            params.eps = parseFloat(document.getElementById('eps-input').value);
            params.min_samples = parseInt(document.getElementById('min-samples-input').value);
        } else if (['kmeans', 'hierarchical'].includes(algorithm)) {
            params.n_clusters = parseInt(document.getElementById('clusters-input').value);
        }

        return params;
    }

    showProgress() {
        document.getElementById('progress-panel').style.display = 'block';
        document.getElementById('results-section').style.display = 'none';
        
        // Simulate progress for better UX
        this.simulateProgress();
    }

    hideProgress() {
        document.getElementById('progress-panel').style.display = 'none';
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    simulateProgress() {
        let progress = 0;
        const stages = [
            { progress: 20, text: 'Lecture des images...' },
            { progress: 50, text: 'Détection des visages...' },
            { progress: 75, text: 'Calcul des embeddings...' },
            { progress: 90, text: 'Clustering en cours...' },
            { progress: 100, text: 'Finalisation...' }
        ];

        let stageIndex = 0;
        this.progressInterval = setInterval(() => {
            if (stageIndex < stages.length) {
                const stage = stages[stageIndex];
                this.updateProgress(stage.progress, stage.text);
                stageIndex++;
            }
        }, 2000);
    }

    updateProgress(percentage, text) {
        document.getElementById('progress-fill').style.width = `${percentage}%`;
        document.getElementById('progress-text').textContent = text;
    }    displayResults(data) {
        const { clusters, statistics } = data;
        
        // Store cluster data for later use
        this.clusterData = clusters;
        console.log('Stored cluster data:', this.clusterData);
        
        // Update statistics
        document.getElementById('total-faces').textContent = statistics.total_faces;
        document.getElementById('clustered-faces').textContent = statistics.clustered_faces;
        document.getElementById('num-clusters').textContent = statistics.num_clusters;
        document.getElementById('clustering-rate').textContent = `${Math.round(statistics.clustering_rate * 100)}%`;

        // Update results title
        document.getElementById('results-title').textContent = 
            `${statistics.num_clusters} personnes identifiées`;

        // Display clusters
        this.displayClusters(clusters);

        // Show results section
        document.getElementById('results-section').style.display = 'block';
        
        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    displayClusters(clusters) {
        const grid = document.getElementById('clusters-grid');
        grid.innerHTML = '';

        Object.entries(clusters).forEach(([clusterId, cluster]) => {
            const clusterCard = this.createClusterCard(clusterId, cluster);
            grid.appendChild(clusterCard);
        });
    }

    createClusterCard(clusterId, cluster) {
        const card = document.createElement('div');
        card.className = 'cluster-card';

        const header = document.createElement('div');
        header.className = 'cluster-header';
        header.innerHTML = `
            <div class="cluster-title">Personne ${parseInt(clusterId) + 1}</div>
            <div class="cluster-count">${cluster.count} images</div>
        `;

        const facesGrid = document.createElement('div');
        facesGrid.className = 'cluster-faces';
        
        cluster.faces.forEach((faceData, index) => {
            const img = document.createElement('img');
            img.className = 'face-image';
            img.src = `data:image/jpeg;base64,${faceData}`;
            img.alt = `Visage ${index + 1}`;
            img.addEventListener('click', () => {
                this.showImageModal(img.src, `Personne ${parseInt(clusterId) + 1} - Visage ${index + 1}`);
            });
            facesGrid.appendChild(img);
        });

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.className = 'cluster-name-input';
        nameInput.placeholder = 'Nom de la personne';
        nameInput.dataset.clusterId = clusterId;

        card.appendChild(header);
        card.appendChild(facesGrid);
        card.appendChild(nameInput);

        return card;
    }

    showImageModal(src, title) {
        const modal = document.getElementById('image-modal');
        const modalImage = document.getElementById('modal-image');
        const modalInfo = document.getElementById('modal-info');

        modalImage.src = src;
        modalInfo.textContent = title;
        modal.style.display = 'flex';
    }

    closeModal() {
        document.getElementById('image-modal').style.display = 'none';
    }    async saveMetadata() {
        const nameInputs = document.querySelectorAll('.cluster-name-input');
        const clusters = {};

        // Collect cluster data
        nameInputs.forEach(input => {
            const clusterId = input.dataset.clusterId;
            const name = input.value.trim();
            
            if (name) {
                // Get paths from the original clustering results
                const paths = this.getClusterPaths(clusterId);
                console.log(`Cluster ${clusterId}: ${name}, paths:`, paths);
                
                clusters[clusterId] = {
                    name: name,
                    paths: paths
                };
            }
        });

        console.log('Metadata to save:', clusters);

        if (Object.keys(clusters).length === 0) {
            this.showToast('Aucun nom saisi', 'warning');
            return;
        }

        try {
            this.showLoading('Sauvegarde des métadonnées...');
            
            const response = await this.apiCall('/metadata/add', {
                method: 'POST',
                body: JSON.stringify(clusters)
            });

            this.hideLoading();
            this.showToast(`Métadonnées sauvegardées pour ${response.success_count} clusters`, 'success');

        } catch (error) {
            this.hideLoading();
            this.showToast(`Erreur lors de la sauvegarde: ${error.message}`, 'error');
            console.error('Metadata save error:', error);
        }
    }getClusterPaths(clusterId) {
        // Retrieve paths from stored cluster data
        if (this.clusterData && this.clusterData[clusterId]) {
            return this.clusterData[clusterId].paths || [];
        }
        return [];
    }

    async performSearch() {
        const directory = document.getElementById('search-directory').value.trim();
        const personNames = document.getElementById('person-names').value.trim();

        if (!directory || !personNames) {
            this.showToast('Veuillez remplir tous les champs', 'error');
            return;
        }

        try {
            this.showLoading('Recherche en cours...');

            const response = await this.apiCall('/search/faces', {
                method: 'POST',
                body: JSON.stringify({
                    directory,
                    face_names: personNames
                })
            });

            this.hideLoading();
            this.displaySearchResults(response.matches);
            this.showToast(`${response.count} images trouvées`, 'success');

        } catch (error) {
            this.hideLoading();
            this.showToast(`Erreur de recherche: ${error.message}`, 'error');
            console.error('Search error:', error);
        }
    }

    displaySearchResults(results) {
        const resultsSection = document.getElementById('search-results');
        const gallery = document.getElementById('search-gallery');
        
        resultsSection.style.display = 'block';
        gallery.innerHTML = '';

        if (results.length === 0) {
            gallery.innerHTML = '<p>Aucun résultat trouvé</p>';
            return;
        }

        results.forEach(imagePath => {
            const item = this.createSearchItem(imagePath);
            gallery.appendChild(item);
        });

        document.getElementById('search-results-title').textContent = 
            `${results.length} images trouvées`;
    }

    createSearchItem(imagePath) {
        const item = document.createElement('div');
        item.className = 'search-item';

        const filename = imagePath.split('\\').pop();
        const directory = imagePath.replace(filename, '');

        item.innerHTML = `
            <img src="file:///${imagePath}" alt="${filename}" class="search-image" 
                 onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRkYwMDAwIi8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSJ3aGl0ZSI+RXJyZXVyPC90ZXh0Pgo8L3N2Zz4='">
            <div class="search-info">
                <div class="search-filename">${filename}</div>
                <div class="search-path">${directory}</div>
            </div>
        `;

        item.addEventListener('click', () => {
            this.openImageInExplorer(imagePath);
        });

        return item;
    }

    changeSearchView(view) {
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });

        const gallery = document.getElementById('search-gallery');
        gallery.classList.toggle('list-view', view === 'list');
    }

    openImageInExplorer(imagePath) {
        // This would need to be implemented with electron or similar for desktop apps
        // For web, we can show the image in a modal
        this.showImageModal(`file:///${imagePath}`, imagePath);
    }

    async exportResults() {
        try {
            const data = this.collectResultsData();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `face-clustering-results-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('Résultats exportés avec succès', 'success');
        } catch (error) {
            this.showToast('Erreur lors de l\'export', 'error');
            console.error('Export error:', error);
        }
    }

    collectResultsData() {
        const nameInputs = document.querySelectorAll('.cluster-name-input');
        const results = {
            timestamp: new Date().toISOString(),
            statistics: {
                total_faces: document.getElementById('total-faces').textContent,
                clustered_faces: document.getElementById('clustered-faces').textContent,
                num_clusters: document.getElementById('num-clusters').textContent,
                clustering_rate: document.getElementById('clustering-rate').textContent
            },
            clusters: {}
        };

        nameInputs.forEach(input => {
            const clusterId = input.dataset.clusterId;
            const name = input.value.trim();
            
            results.clusters[clusterId] = {
                name: name || `Personne ${parseInt(clusterId) + 1}`,
                paths: this.getClusterPaths(clusterId)
            };
        });

        return results;
    }

    validateDirectoryPath(input) {
        const path = input.value.trim();
        if (!path) return;

        // Basic validation for Windows paths
        const isValidPath = /^[A-Za-z]:(\\[^<>:"|?*\n\r]*)*\\?$/.test(path) || 
                           /^\\\\[^<>:"|?*\n\r]+\\[^<>:"|?*\n\r]*/.test(path);

        input.classList.toggle('invalid', !isValidPath);
        
        if (!isValidPath) {
            this.showToast('Chemin de dossier invalide', 'warning');
        }
    }

    handleKeyboardShortcuts(e) {
        // Ctrl+Enter to start clustering
        if (e.ctrlKey && e.key === 'Enter' && this.currentTab === 'clustering') {
            if (!this.isProcessing) {
                this.startClustering();
            }
        }
        
        // Escape to close modal
        if (e.key === 'Escape') {
            this.closeModal();
        }
        
        // Tab navigation (Ctrl+1, Ctrl+2)
        if (e.ctrlKey && ['1', '2'].includes(e.key)) {
            const tabMap = { '1': 'clustering', '2': 'search' };
            this.switchTab(tabMap[e.key]);
        }
    }

    showLoading(message = 'Chargement...') {
        const overlay = document.getElementById('loading-overlay');
        const spinner = overlay.querySelector('.loading-spinner p');
        spinner.textContent = message;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }

    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, duration);
    }

    async apiCall(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const response = await fetch(url, { ...defaultOptions, ...options });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FaceClusteringApp();
});

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(console.error);
}
