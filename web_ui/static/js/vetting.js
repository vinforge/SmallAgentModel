/**
 * Vetting Manager for Phase 7.3: UI Integration & The "Go/No-Go" Decision
 * 
 * This module handles the frontend interaction with SAM's automated vetting
 * system, including triggering vetting processes, displaying results, and
 * managing the final approval/rejection decisions.
 */

class VettingManager {
    constructor() {
        this.statusCheckInterval = null;
        this.refreshInterval = 30000; // 30 seconds
        this.isVettingInProgress = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateStatus();
        this.loadVettedContent();
        this.startStatusPolling();
    }
    
    setupEventListeners() {
        // Vet All button
        const vetAllBtn = document.getElementById('vet-all-btn');
        if (vetAllBtn) {
            vetAllBtn.addEventListener('click', () => this.triggerVetting());
        }
        
        // Refresh buttons
        const refreshStatusBtn = document.getElementById('refresh-status-btn');
        if (refreshStatusBtn) {
            refreshStatusBtn.addEventListener('click', () => this.updateStatus());
        }
        
        const refreshVettedBtn = document.getElementById('refresh-vetted-btn');
        if (refreshVettedBtn) {
            refreshVettedBtn.addEventListener('click', () => this.loadVettedContent());
        }
        
        // Decision buttons (delegated event handling)
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('approve-btn')) {
                const filename = e.target.dataset.filename;
                this.showApprovalConfirmation(filename);
            } else if (e.target.classList.contains('reject-btn')) {
                const filename = e.target.dataset.filename;
                this.showRejectionDialog(filename);
            } else if (e.target.classList.contains('details-btn')) {
                const filename = e.target.dataset.filename;
                this.showDetailedAnalysis(filename);
            }
        });
        
        // Modal confirmation handlers
        const confirmApproveBtn = document.getElementById('confirm-approve-btn');
        if (confirmApproveBtn) {
            confirmApproveBtn.addEventListener('click', () => this.executeApproval());
        }
        
        const confirmRejectBtn = document.getElementById('confirm-reject-btn');
        if (confirmRejectBtn) {
            confirmRejectBtn.addEventListener('click', () => this.executeRejection());
        }
    }
    
    async triggerVetting() {
        if (this.isVettingInProgress) {
            this.showWarning('Vetting process is already in progress. Please wait...');
            return;
        }
        
        const button = document.getElementById('vet-all-btn');
        const originalText = button.textContent;
        
        try {
            // Show loading state
            this.isVettingInProgress = true;
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Content...';
            
            this.showInfo('Starting automated vetting process...');
            
            const response = await fetch('/vetting/api/vet-all', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                const stats = result.stats || {};
                this.showSuccess(
                    `Vetting completed successfully! ` +
                    `${stats.vetted_files || 0} files analyzed. ` +
                    `${stats.quarantine_files || 0} files remaining in quarantine.`
                );
                
                // Refresh displays
                this.updateStatus();
                this.loadVettedContent();
                
            } else {
                this.showError(`Vetting failed: ${result.message}`);
            }
            
        } catch (error) {
            this.showError(`Error: ${error.message}`);
        } finally {
            // Restore button
            this.isVettingInProgress = false;
            button.disabled = false;
            button.textContent = originalText;
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/vetting/api/status');
            const status = await response.json();
            
            if (status.status === 'success') {
                this.updateStatusDisplay(status);
            } else {
                console.error('Error getting vetting status:', status.error);
            }
            
        } catch (error) {
            console.error('Error updating vetting status:', error);
        }
    }
    
    updateStatusDisplay(status) {
        // Update file counts
        this.updateElementText('quarantine-count', status.quarantine_files);
        this.updateElementText('vetted-count', status.vetted_files);
        this.updateElementText('approved-count', status.approved_files);
        this.updateElementText('rejected-count', status.rejected_files);
        
        // Update system status indicator
        const statusIndicator = document.getElementById('system-status');
        if (statusIndicator) {
            if (status.system_operational) {
                statusIndicator.className = 'badge bg-success';
                statusIndicator.textContent = 'Operational';
            } else {
                statusIndicator.className = 'badge bg-danger';
                statusIndicator.textContent = 'Error';
            }
        }
        
        // Enable/disable vet button
        const vetButton = document.getElementById('vet-all-btn');
        if (vetButton) {
            vetButton.disabled = !status.ready_for_vetting || this.isVettingInProgress;
            
            if (!status.ready_for_vetting) {
                vetButton.title = 'No files in quarantine to vet';
            } else {
                vetButton.title = 'Analyze all quarantined content';
            }
        }
        
        // Show/hide vetted results section
        const vettedSection = document.getElementById('vetted-results-section');
        if (vettedSection) {
            vettedSection.style.display = status.has_vetted_content ? 'block' : 'none';
        }
    }
    
    async loadVettedContent() {
        try {
            const response = await fetch('/vetting/api/vetted-content');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.displayVettedContent(result.files);
            } else {
                this.showError(`Error loading vetted content: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Error loading vetted content: ${error.message}`);
        }
    }
    
    displayVettedContent(files) {
        const container = document.getElementById('vetted-content-list');
        if (!container) return;
        
        if (files.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📭</div>
                    <h4>No vetted content available</h4>
                    <p>Run the vetting process to analyze quarantined content</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = files.map(file => this.createVettedItemHTML(file)).join('');
    }
    
    createVettedItemHTML(file) {
        const recommendationClass = this.getRecommendationClass(file.recommendation);

        return `
            <div class="vetted-item ${recommendationClass}">
                <div class="vetted-item-header">
                    <div class="vetted-item-info">
                        <div class="recommendation-badge ${recommendationClass}">
                            ${this.getRecommendationIcon(file.recommendation)} ${file.recommendation}
                        </div>
                        <div class="vetted-url">${this.truncateUrl(file.url, 80)}</div>
                        <div class="content-preview">${file.content_preview}</div>
                    </div>

                    <div class="action-buttons">
                        <button class="action-btn approve approve-btn"
                                data-filename="${file.filename}"
                                ${file.recommendation === 'FAIL' ? 'title="Warning: Recommended FAIL"' : ''}>
                            ✅ Use & Add
                        </button>
                        <button class="action-btn reject reject-btn"
                                data-filename="${file.filename}">
                            ❌ Discard
                        </button>
                        <button class="action-btn details details-btn"
                                data-filename="${file.filename}">
                            🔍 Details
                        </button>
                    </div>
                </div>

                <div class="score-grid">
                    <div class="score-item">
                        <div class="score-label">Overall Score</div>
                        <div class="score-value">${file.overall_score}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Confidence</div>
                        <div class="score-value">${(file.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Risk Factors</div>
                        <div class="score-value">${file.risk_factors}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Source Rep</div>
                        <div class="score-value">${file.source_reputation_score}</div>
                    </div>
                </div>

                <div class="dimension-scores">
                    ${this.createScoreBar('Credibility', file.scores.credibility)}
                    ${this.createScoreBar('Purity', file.scores.purity)}
                    ${this.createScoreBar('Persuasion', file.scores.persuasion, true)}
                    ${this.createScoreBar('Speculation', file.scores.speculation, true)}
                </div>

                <div class="file-metadata">
                    <div class="metadata-item">
                        <span>🕒</span> ${this.formatTimestamp(file.timestamp)}
                    </div>
                    <div class="metadata-item">
                        <span>📄</span> ${this.formatFileSize(file.file_size)}
                    </div>
                    <div class="metadata-item">
                        <span>⚡</span> ${(file.processing_time * 1000).toFixed(0)}ms
                    </div>
                </div>
            </div>
        `;
    }
    
    createScoreBar(label, score, inverse = false) {
        const percentage = (score * 100).toFixed(1);
        const colorClass = this.getScoreColorClass(score, inverse);

        return `
            <div class="score-bar">
                <div class="score-bar-header">
                    <span class="score-bar-label">${label}</span>
                    <span class="score-bar-value">${percentage}%</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar ${colorClass}"
                         style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    }
    
    getScoreColorClass(score, inverse = false) {
        if (inverse) {
            // For persuasion/speculation, lower is better
            if (score <= 0.3) return 'success';
            if (score <= 0.6) return 'warning';
            return 'danger';
        } else {
            // For credibility/purity, higher is better
            if (score >= 0.7) return 'success';
            if (score >= 0.4) return 'warning';
            return 'danger';
        }
    }

    getRecommendationClass(recommendation) {
        switch (recommendation) {
            case 'PASS': return 'pass';
            case 'REVIEW': return 'review';
            case 'FAIL': return 'fail';
            default: return 'unknown';
        }
    }

    getRecommendationIcon(recommendation) {
        switch (recommendation) {
            case 'PASS': return '✅';
            case 'REVIEW': return '⚠️';
            case 'FAIL': return '❌';
            default: return '❓';
        }
    }
    
    showApprovalConfirmation(filename) {
        document.getElementById('approval-filename').textContent = filename;
        document.getElementById('pending-approval-filename').value = filename;
        document.getElementById('approval-modal').style.display = 'flex';
    }

    showRejectionDialog(filename) {
        document.getElementById('rejection-filename').textContent = filename;
        document.getElementById('pending-rejection-filename').value = filename;
        document.getElementById('rejection-reason').value = '';
        document.getElementById('rejection-modal').style.display = 'flex';
    }
    
    async executeApproval() {
        const filename = document.getElementById('pending-approval-filename').value;
        
        try {
            const response = await fetch('/vetting/api/approve-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filename })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showSuccess(`Content approved and added to knowledge base: ${filename}`);
                this.loadVettedContent();
                this.updateStatus();
            } else {
                this.showError(`Approval failed: ${result.message}`);
            }
            
        } catch (error) {
            this.showError(`Error approving content: ${error.message}`);
        }
    }
    
    async executeRejection() {
        const filename = document.getElementById('pending-rejection-filename').value;
        const reason = document.getElementById('rejection-reason').value || 'User rejected';
        
        try {
            const response = await fetch('/vetting/api/reject-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filename, reason })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showSuccess(`Content rejected: ${filename}`);
                this.loadVettedContent();
                this.updateStatus();
            } else {
                this.showError(`Rejection failed: ${result.message}`);
            }
            
        } catch (error) {
            this.showError(`Error rejecting content: ${error.message}`);
        }
    }
    
    async showDetailedAnalysis(filename) {
        try {
            const response = await fetch(`/vetting/api/vetted-content/${filename}`);
            const result = await response.json();
            
            if (result.status === 'success') {
                this.displayDetailedAnalysis(result.data);
            } else {
                this.showError(`Error loading details: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Error loading details: ${error.message}`);
        }
    }
    
    displayDetailedAnalysis(data) {
        const vetting = data.vetting_results || {};

        // Populate modal content
        document.getElementById('details-url').textContent = data.url || 'Unknown';

        const recommendationElement = document.getElementById('details-recommendation');
        const recommendation = vetting.recommendation || 'Unknown';
        recommendationElement.textContent = recommendation;
        recommendationElement.className = `recommendation-badge ${this.getRecommendationClass(recommendation)}`;

        document.getElementById('details-score').textContent = (vetting.overall_score || 0).toFixed(3);
        document.getElementById('details-confidence').textContent = ((vetting.confidence || 0) * 100).toFixed(1) + '%';
        document.getElementById('details-reason').textContent = vetting.reason || 'No reason provided';

        // Risk factors
        const riskFactors = vetting.risk_assessment?.risk_factors || [];
        const riskContainer = document.getElementById('details-risk-factors');

        if (riskFactors.length > 0) {
            riskContainer.innerHTML = riskFactors.map(risk => `
                <div class="warning-box">
                    <strong>${risk.dimension}:</strong> ${risk.description}
                    <br><small>Score: ${risk.score.toFixed(3)} | Threshold: ${risk.threshold}</small>
                </div>
            `).join('');
        } else {
            riskContainer.innerHTML = '<div style="color: #999;">No risk factors identified</div>';
        }

        // Sanitization results
        const sanitization = vetting.sanitization || {};
        document.getElementById('details-removed-elements').textContent =
            (sanitization.removed_elements || []).join(', ') || 'None';
        document.getElementById('details-suspicious-patterns').textContent =
            (sanitization.suspicious_patterns || []).length || '0';
        document.getElementById('details-purity-score').textContent =
            (sanitization.purity_score || 0).toFixed(3);

        document.getElementById('details-modal').style.display = 'flex';
    }
    
    getRiskSeverityClass(severity) {
        switch (severity) {
            case 'critical': return 'danger';
            case 'high': return 'warning';
            case 'medium': return 'info';
            case 'low': return 'light';
            default: return 'secondary';
        }
    }
    
    startStatusPolling() {
        this.statusCheckInterval = setInterval(() => {
            this.updateStatus();
        }, this.refreshInterval);
    }
    
    stopStatusPolling() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }
    
    // Utility methods
    updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }
    
    truncateUrl(url, maxLength = 50) {
        if (url.length <= maxLength) return url;
        return url.substring(0, maxLength - 3) + '...';
    }
    
    formatTimestamp(timestamp) {
        try {
            return new Date(timestamp).toLocaleString();
        } catch {
            return timestamp;
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    // Notification methods
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showError(message) {
        this.showNotification(message, 'danger');
    }
    
    showWarning(message) {
        this.showNotification(message, 'warning');
    }
    
    showInfo(message) {
        this.showNotification(message, 'info');
    }
    
    showNotification(message, type) {
        // Create toast notification
        const toastContainer = document.getElementById('toast-container') || this.createToastContainer();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icon = {
            'success': '✅',
            'danger': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }[type] || 'ℹ️';

        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${icon}</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);

        // Animate in
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1055;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;

        // Add toast styles
        const style = document.createElement('style');
        style.textContent = `
            .toast {
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                min-width: 300px;
                opacity: 0;
                transform: translateX(100%);
                transition: all 0.3s ease;
            }

            .toast.show {
                opacity: 1;
                transform: translateX(0);
            }

            .toast-content {
                display: flex;
                align-items: center;
                padding: 15px;
                gap: 10px;
            }

            .toast-icon {
                font-size: 18px;
            }

            .toast-message {
                flex: 1;
                color: #333;
                font-weight: 500;
            }

            .toast-close {
                background: none;
                border: none;
                font-size: 20px;
                cursor: pointer;
                color: #999;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .toast-close:hover {
                color: #666;
            }

            .toast-success {
                border-left: 4px solid #28a745;
            }

            .toast-danger {
                border-left: 4px solid #dc3545;
            }

            .toast-warning {
                border-left: 4px solid #ffc107;
            }

            .toast-info {
                border-left: 4px solid #17a2b8;
            }
        `;

        document.head.appendChild(style);
        document.body.appendChild(container);
        return container;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if we're on a page with vetting elements
    if (document.getElementById('vet-all-btn') || document.getElementById('vetted-content-list')) {
        window.vettingManager = new VettingManager();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VettingManager;
}
