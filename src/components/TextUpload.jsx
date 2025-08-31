import React, { useState, useRef } from 'react';
import './TextUpload.css';

const TextUpload = ({ onTextSubmit, onFileUpload, isLoading }) => {
  const [text, setText] = useState('');
  const [source, setSource] = useState('user_input');
  const [title, setTitle] = useState('Untitled');
  const [uploadStatus, setUploadStatus] = useState('');
  const fileInputRef = useRef(null);

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setUploadStatus('Please enter some text to upload.');
      return;
    }

    setUploadStatus('Uploading text...');
    try {
      await onTextSubmit(text, source, title);
      setUploadStatus('Text uploaded successfully!');
      setText('');
      setTitle('Untitled');
    } catch (error) {
      setUploadStatus(`Upload failed: ${error.message}`);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setUploadStatus('File size must be less than 10MB.');
      return;
    }

    setUploadStatus('Processing file...');
    try {
      const text = await readFileAsText(file);
      await onFileUpload(text, file.name, file.name);
      setUploadStatus('File uploaded successfully!');
      setTitle(file.name);
    } catch (error) {
      setUploadStatus(`File upload failed: ${error.message}`);
    }
  };

  const readFileAsText = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = (e) => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInputRef.current.files = files;
      handleFileUpload({ target: { files } });
    }
  };

  return (
    <div className="text-upload">
      <h2>📚 Upload Text or Documents</h2>
      
      {/* Text Input */}
      <div className="upload-section">
        <h3>📝 Enter Text</h3>
        <form onSubmit={handleTextSubmit}>
          <div className="form-group">
            <label htmlFor="source">Source:</label>
            <input
              type="text"
              id="source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="e.g., user_input, document, article"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="title">Title:</label>
            <input
              type="text"
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., AI Overview, Research Paper"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="text">Text Content:</label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your text here... (supports up to 50,000 characters)"
              rows="8"
              maxLength="50000"
            />
            <div className="char-count">
              {text.length}/50,000 characters
            </div>
          </div>
          
          <button 
            type="submit" 
            disabled={isLoading || !text.trim()}
            className="submit-btn"
          >
            {isLoading ? '⏳ Uploading...' : '🚀 Upload Text'}
          </button>
        </form>
      </div>

      {/* File Upload */}
      <div className="upload-section">
        <h3>📁 Upload File</h3>
        <div
          className="file-drop-zone"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.csv,.json"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <div className="drop-zone-content">
            <div className="upload-icon">📁</div>
            <p>Drag and drop a text file here, or</p>
            <button 
              type="button" 
              onClick={() => fileInputRef.current.click()}
              className="browse-btn"
            >
              Browse Files
            </button>
            <p className="file-info">
              Supported formats: .txt, .md, .csv, .json (max 10MB)
            </p>
          </div>
        </div>
      </div>

      {/* Status Messages */}
      {uploadStatus && (
        <div className={`status-message ${uploadStatus.includes('successfully') ? 'success' : uploadStatus.includes('failed') ? 'error' : 'info'}`}>
          {uploadStatus}
        </div>
      )}

      {/* Usage Tips */}
      <div className="usage-tips">
        <h4>💡 Tips for Better Results:</h4>
        <ul>
          <li>Use descriptive titles for easier document management</li>
          <li>Break long documents into logical sections</li>
          <li>Include relevant keywords in your text</li>
          <li>Text will be automatically chunked into optimal segments</li>
        </ul>
      </div>
    </div>
  );
};

export default TextUpload;
