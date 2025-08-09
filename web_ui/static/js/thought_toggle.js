/**
 * Thought Toggle JavaScript - Sprint 16
 * Handles interactive thought visibility controls in the web UI.
 */

class ThoughtToggleManager {
    constructor() {
        this.thoughtsEnabled = true;
        this.expandedThoughts = new Set();
        this.thoughtData = new Map();
        
        this.initializeEventListeners();
        this.loadConfiguration();
        
        console.log('🧠 Thought Toggle Manager initialized');
    }
    
    initializeEventListeners() {
        // Keyboard shortcut: Alt + T
        document.addEventListener('keydown', (event) => {
            if (event.altKey && event.key.toLowerCase() === 't') {
                event.preventDefault();
                this.toggleMostRecentThought();
            }
        });
        
        // Listen for new messages
        this.observeNewMessages();
    }
    
    loadConfiguration() {
        // Load thought visibility preference from localStorage
        const savedPreference = localStorage.getItem('sam_thoughts_enabled');
        if (savedPreference !== null) {
            this.thoughtsEnabled = JSON.parse(savedPreference);
        }
        
        console.log(`Thoughts enabled: ${this.thoughtsEnabled}`);
    }
    
    saveConfiguration() {
        localStorage.setItem('sam_thoughts_enabled', JSON.stringify(this.thoughtsEnabled));
    }
    
    observeNewMessages() {
        // Use MutationObserver to detect new chat messages
        const chatContainer = document.querySelector('.chat-container') || document.body;
        
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        this.processNewMessage(node);
                    }
                });
            });
        });
        
        observer.observe(chatContainer, {
            childList: true,
            subtree: true
        });
    }
    
    processNewMessage(element) {
        // Look for thought data in comments
        const thoughtComments = this.findThoughtComments(element);
        
        thoughtComments.forEach((comment, index) => {
            try {
                const thoughtData = JSON.parse(comment.data.replace('THOUGHT_DATA: ', ''));
                this.createThoughtToggles(element, thoughtData, index);
            } catch (error) {
                console.error('Error parsing thought data:', error);
            }
        });
    }
    
    findThoughtComments(element) {
        const comments = [];
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_COMMENT,
            null,
            false
        );
        
        let node;
        while (node = walker.nextNode()) {
            if (node.data.includes('THOUGHT_DATA:')) {
                comments.push(node);
            }
        }
        
        return comments;
    }
    
    createThoughtToggles(messageElement, thoughtDataArray, messageIndex) {
        if (!this.thoughtsEnabled) {
            return; // Don't create toggles if thoughts are disabled
        }
        
        // Find the "SAM's Thoughts Available" text and replace with interactive toggles
        const thoughtIndicators = messageElement.querySelectorAll('*');
        let thoughtIndicator = null;

        for (let element of thoughtIndicators) {
            if (element.textContent && element.textContent.includes("SAM's Thoughts Available")) {
                thoughtIndicator = element;
                break;
            }
        }

        if (thoughtIndicator) {
            
            // Create container for thought toggles
            const toggleContainer = document.createElement('div');
            toggleContainer.className = 'thought-toggles-container';
            toggleContainer.style.cssText = `
                margin: 15px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            `;
            
            // Create toggles for each thought block
            thoughtDataArray.forEach((thoughtBlock, blockIndex) => {
                const toggleId = `thought_${messageIndex}_${blockIndex}`;
                const contentId = `thought_content_${messageIndex}_${blockIndex}`;
                
                // Store thought data
                this.thoughtData.set(toggleId, thoughtBlock);
                
                // Create toggle button
                const toggleButton = this.createToggleButton(toggleId, contentId, thoughtBlock);
                const contentDiv = this.createContentDiv(contentId, thoughtBlock);
                
                toggleContainer.appendChild(toggleButton);
                toggleContainer.appendChild(contentDiv);
            });
            
            // Replace the indicator with the toggle container
            thoughtIndicator.parentNode.replaceChild(toggleContainer, thoughtIndicator);
        }
    }
    
    createToggleButton(toggleId, contentId, thoughtBlock) {
        const button = document.createElement('button');
        button.id = toggleId;
        button.className = 'thought-toggle-btn';
        button.style.cssText = `
            width: 100%;
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px 16px;
            cursor: pointer;
            font-size: 14px;
            color: #495057;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
            margin-bottom: 8px;
        `;
        
        // Button content
        const leftContent = document.createElement('div');
        leftContent.style.cssText = 'display: flex; align-items: center; gap: 8px;';
        
        const arrow = document.createElement('span');
        arrow.id = `${toggleId}_arrow`;
        arrow.textContent = '▶';
        arrow.style.cssText = 'font-size: 12px; transition: transform 0.2s ease;';
        
        const label = document.createElement('span');
        label.textContent = '🧠 SAM\'s Thoughts';
        
        const metadata = document.createElement('span');
        metadata.textContent = `(${thoughtBlock.token_count} tokens)`;
        metadata.style.cssText = 'font-size: 12px; color: #6c757d;';
        
        leftContent.appendChild(arrow);
        leftContent.appendChild(label);
        
        button.appendChild(leftContent);
        button.appendChild(metadata);
        
        // Click handler
        button.addEventListener('click', () => {
            this.toggleThought(contentId, toggleId);
        });
        
        // Hover effects
        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = '#f8f9fa';
            button.style.borderColor = '#007bff';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.backgroundColor = '#ffffff';
            button.style.borderColor = '#dee2e6';
        });
        
        return button;
    }
    
    createContentDiv(contentId, thoughtBlock) {
        const div = document.createElement('div');
        div.id = contentId;
        div.className = 'thought-content';
        div.style.cssText = `
            display: none;
            padding: 16px;
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: #495057;
            white-space: pre-wrap;
            overflow-x: auto;
            margin-bottom: 8px;
        `;
        
        // Metadata header
        const metadata = document.createElement('div');
        metadata.style.cssText = `
            font-size: 11px;
            color: #6c757d;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e9ecef;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        metadata.textContent = `Tokens: ${thoughtBlock.token_count} | Time: ${thoughtBlock.timestamp.substring(0, 19)}`;
        
        // Content
        const content = document.createElement('div');
        content.textContent = thoughtBlock.content;
        
        div.appendChild(metadata);
        div.appendChild(content);
        
        return div;
    }
    
    toggleThought(contentId, toggleId) {
        const content = document.getElementById(contentId);
        const button = document.getElementById(toggleId);
        const arrow = document.getElementById(`${toggleId}_arrow`);
        
        if (!content || !button || !arrow) {
            console.error('Thought toggle elements not found');
            return;
        }
        
        const isExpanded = content.style.display !== 'none';
        
        if (isExpanded) {
            // Collapse
            content.style.display = 'none';
            arrow.textContent = '▶';
            arrow.style.transform = 'rotate(0deg)';
            button.style.backgroundColor = '#ffffff';
            this.expandedThoughts.delete(toggleId);
        } else {
            // Expand
            content.style.display = 'block';
            arrow.textContent = '▼';
            arrow.style.transform = 'rotate(90deg)';
            button.style.backgroundColor = '#f8f9fa';
            this.expandedThoughts.add(toggleId);
        }
        
        // Smooth animation
        if (!isExpanded) {
            content.style.maxHeight = '0px';
            content.style.overflow = 'hidden';
            content.style.transition = 'max-height 0.3s ease';
            
            // Trigger animation
            setTimeout(() => {
                content.style.maxHeight = content.scrollHeight + 'px';
            }, 10);
            
            // Remove max-height after animation
            setTimeout(() => {
                content.style.maxHeight = 'none';
                content.style.overflow = 'auto';
            }, 300);
        }
    }
    
    toggleMostRecentThought() {
        const toggleButtons = document.querySelectorAll('.thought-toggle-btn');
        if (toggleButtons.length > 0) {
            const mostRecent = toggleButtons[toggleButtons.length - 1];
            mostRecent.click();
            
            // Visual feedback
            mostRecent.style.boxShadow = '0 0 0 3px rgba(0, 123, 255, 0.25)';
            setTimeout(() => {
                mostRecent.style.boxShadow = 'none';
            }, 500);
        }
    }
    
    setThoughtsEnabled(enabled) {
        this.thoughtsEnabled = enabled;
        this.saveConfiguration();
        
        // Show/hide existing thought toggles
        const toggleContainers = document.querySelectorAll('.thought-toggles-container');
        toggleContainers.forEach(container => {
            container.style.display = enabled ? 'block' : 'none';
        });
        
        console.log(`Thoughts ${enabled ? 'enabled' : 'disabled'}`);
    }
    
    expandAllThoughts() {
        const toggleButtons = document.querySelectorAll('.thought-toggle-btn');
        toggleButtons.forEach(button => {
            const contentId = button.id.replace('thought_', 'thought_content_');
            const content = document.getElementById(contentId);
            
            if (content && content.style.display === 'none') {
                button.click();
            }
        });
    }
    
    collapseAllThoughts() {
        const toggleButtons = document.querySelectorAll('.thought-toggle-btn');
        toggleButtons.forEach(button => {
            const contentId = button.id.replace('thought_', 'thought_content_');
            const content = document.getElementById(contentId);
            
            if (content && content.style.display !== 'none') {
                button.click();
            }
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.thoughtToggleManager = new ThoughtToggleManager();
});

// Export for external use
window.ThoughtToggleManager = ThoughtToggleManager;
