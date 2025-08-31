import React, { useState } from 'react';
import './QueryBox.css';

const QueryBox = ({ onQuerySubmit, isLoading, requestTime, tokenEstimate }) => {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [includeScores, setIncludeScores] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    try {
      await onQuerySubmit(query, topK, includeScores);
      setQuery(''); // Clear query after successful submission
    } catch (error) {
      console.error('Query submission failed:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit(e);
    }
  };

  return (
    <div className="query-box">
      <h2>🔍 Ask Questions About Your Documents</h2>
      
      <form onSubmit={handleSubmit} className="query-form">
        <div className="query-input-group">
          <label htmlFor="query">Your Question:</label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask anything about your uploaded documents... (Ctrl+Enter to submit)"
            rows="3"
            disabled={isLoading}
            className="query-textarea"
          />
          <div className="query-controls">
            <div className="control-group">
              <label htmlFor="topK">Top Results:</label>
              <select
                id="topK"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                disabled={isLoading}
              >
                <option value={5}>5 results</option>
                <option value={10}>10 results</option>
                <option value={15}>15 results</option>
                <option value={20}>20 results</option>
              </select>
            </div>
            
            <div className="control-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeScores}
                  onChange={(e) => setIncludeScores(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="checkmark"></span>
                Include Relevance Scores
              </label>
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="query-submit-btn"
        >
          {isLoading ? (
            <>
              <span className="spinner"></span>
              Processing Query...
            </>
          ) : (
            <>
              🚀 Submit Query
              <span className="shortcut-hint">Ctrl+Enter</span>
            </>
          )}
        </button>
      </form>

      {/* Performance Metrics */}
      {(requestTime || tokenEstimate) && (
        <div className="performance-metrics">
          <h4>📊 Performance Metrics</h4>
          <div className="metrics-grid">
            {requestTime && (
              <div className="metric">
                <span className="metric-label">Response Time:</span>
                <span className="metric-value">{requestTime.toFixed(2)}s</span>
              </div>
            )}
            {tokenEstimate && (
              <div className="metric">
                <span className="metric-label">Token Estimate:</span>
                <span className="metric-value">{tokenEstimate}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Query Tips */}
      <div className="query-tips">
        <h4>💡 Query Tips:</h4>
        <ul>
          <li>Be specific with your questions for better results</li>
          <li>Use keywords that are likely in your documents</li>
          <li>Ask follow-up questions to dive deeper into topics</li>
          <li>Try different phrasings if you don't get the expected results</li>
        </ul>
      </div>

      {/* Example Queries */}
      <div className="example-queries">
        <h4>📝 Example Queries:</h4>
        <div className="example-grid">
          {[
            "What are the main concepts discussed?",
            "Can you summarize the key points?",
            "What are the advantages and disadvantages?",
            "How does this relate to other topics?",
            "What are the practical applications?",
            "Can you explain the methodology used?"
          ].map((example, index) => (
            <button
              key={index}
              type="button"
              onClick={() => setQuery(example)}
              className="example-query-btn"
              disabled={isLoading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QueryBox;
