import React, { useState, useCallback } from 'react';
import TextUpload from './components/TextUpload';
import QueryBox from './components/QueryBox';
import AnswerPanel from './components/AnswerPanel';
import './App.css';

const API_BASE_URL = 'http://localhost:8000'; // Update this for production

function App() {
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [answerData, setAnswerData] = useState(null);
  const [error, setError] = useState(null);
  const [requestTime, setRequestTime] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  // Handle text upload
  const handleTextUpload = useCallback(async (text, source, title) => {
    setIsUploading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, source, title }),
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        setUploadStatus(`✅ Successfully uploaded ${result.data.chunks_ingested} chunks in ${result.data.processing_time.toFixed(2)}s`);
        console.log('Upload result:', result);
      } else {
        throw new Error(result.message || 'Upload failed');
      }
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Handle file upload
  const handleFileUpload = useCallback(async (text, filename, title) => {
    await handleTextUpload(text, filename, title);
  }, [handleTextUpload]);

  // Handle query submission
  const handleQuerySubmit = useCallback(async (query, topK, includeScores) => {
    setIsQuerying(true);
    setError(null);
    setAnswerData(null);
    
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query, 
          top_k: topK, 
          include_scores: includeScores 
        }),
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const result = await response.json();
      const endTime = Date.now();
      const totalTime = (endTime - startTime) / 1000;
      
      setRequestTime(totalTime);
      
      if (result.status === 'success') {
        setAnswerData(result);
        console.log('Query result:', result);
      } else if (result.status === 'no_results') {
        setAnswerData(result);
      } else {
        throw new Error(result.message || 'Query failed');
      }
    } catch (err) {
      setError(`Query failed: ${err.message}`);
      console.error('Query error:', err);
    } finally {
      setIsQuerying(false);
    }
  }, []);

  // Estimate token count (rough approximation)
  const estimateTokens = useCallback((text) => {
    if (!text) return 0;
    // Rough estimate: 1 token ≈ 4 characters for English text
    return Math.ceil(text.length / 4);
  }, []);

  const tokenEstimate = answerData?.metadata?.answer_length 
    ? estimateTokens(answerData.metadata.answer_length)
    : null;

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 Mini RAG System</h1>
        <p>Upload documents, ask questions, get AI-powered answers with citations</p>
      </header>

      <main className="App-main">
        <div className="container">
          {/* Text Upload Section */}
          <section className="upload-section">
            <TextUpload
              onTextSubmit={handleTextUpload}
              onFileUpload={handleFileUpload}
              isLoading={isUploading}
            />
            {uploadStatus && (
              <div className="upload-status">
                {uploadStatus}
              </div>
            )}
          </section>

          {/* Query Section */}
          <section className="query-section">
            <QueryBox
              onQuerySubmit={handleQuerySubmit}
              isLoading={isQuerying}
              requestTime={requestTime}
              tokenEstimate={tokenEstimate}
            />
          </section>

          {/* Answer Section */}
          <section className="answer-section">
            <AnswerPanel
              answerData={answerData}
              isLoading={isQuerying}
              error={error}
            />
          </section>
        </div>
      </main>

      <footer className="App-footer">
        <div className="footer-content">
          <p>Built with React + FastAPI + Pinecone + Google Gemini</p>
          <div className="tech-stack">
            <span className="tech-item">🔍 Vector Search</span>
            <span className="tech-item">🤖 AI LLM</span>
            <span className="tech-item">📚 Citations</span>
            <span className="tech-item">⚡ Async Processing</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;