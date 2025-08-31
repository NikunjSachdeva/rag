import React, { useState } from 'react';
import './AnswerPanel.css';

const AnswerPanel = ({ answerData, isLoading, error }) => {
  const [expandedCitations, setExpandedCitations] = useState(new Set());
  const [selectedSource, setSelectedSource] = useState(null);

  if (isLoading) {
    return (
      <div className="answer-panel loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>🤖 Generating answer with citations...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="answer-panel error">
        <div className="error-message">
          <h3>❌ Error Occurred</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!answerData || !answerData.answer) {
    return (
      <div className="answer-panel empty">
        <div className="empty-state">
          <h3>🔍 No Results Yet</h3>
          <p>Upload some documents and ask a question to get started!</p>
        </div>
      </div>
    );
  }

  const toggleCitation = (citationId) => {
    const newExpanded = new Set(expandedCitations);
    if (newExpanded.has(citationId)) {
      newExpanded.delete(citationId);
    } else {
      newExpanded.add(citationId);
    }
    setExpandedCitations(newExpanded);
  };

  const formatAnswerWithCitations = (answer, citations) => {
    if (!citations || citations.length === 0) return answer;

    let formattedAnswer = answer;
    
    // Sort citations by ID to ensure proper replacement
    const sortedCitations = [...citations].sort((a, b) => a.citation_id - b.citation_id);
    
    // Replace citation markers with clickable spans
    sortedCitations.forEach((citation) => {
      const marker = `[${citation.citation_id}]`;
      const replacement = `<span class="citation-marker" data-citation-id="${citation.citation_id}">${marker}</span>`;
      formattedAnswer = formattedAnswer.replace(marker, replacement);
    });

    return formattedAnswer;
  };

  const renderAnswer = () => {
    const { answer, citations } = answerData;
    const formattedAnswer = formatAnswerWithCitations(answer, citations);

    return (
      <div 
        className="answer-content"
        dangerouslySetInnerHTML={{ __html: formattedAnswer }}
        onClick={(e) => {
          if (e.target.classList.contains('citation-marker')) {
            const citationId = parseInt(e.target.dataset.citationId);
            const citation = citations.find(c => c.citation_id === citationId);
            if (citation) {
              setSelectedSource(citation);
            }
          }
        }}
      />
    );
  };

  const renderCitations = () => {
    const { citations } = answerData;
    if (!citations || citations.length === 0) return null;

    return (
      <div className="citations-section">
        <h3>📚 Citations & Sources</h3>
        <div className="citations-grid">
          {citations.map((citation) => (
            <div key={citation.citation_id} className="citation-item">
              <div className="citation-header">
                <span className="citation-number">[{citation.citation_id}]</span>
                <span className="citation-score">Score: {citation.relevance_score.toFixed(3)}</span>
                <button
                  className="expand-btn"
                  onClick={() => toggleCitation(citation.citation_id)}
                >
                  {expandedCitations.has(citation.citation_id) ? '−' : '+'}
                </button>
              </div>
              
              <div className="citation-meta">
                <span className="source">Source: {citation.source}</span>
                <span className="chunk">Chunk: {citation.chunk_id}</span>
              </div>
              
              {expandedCitations.has(citation.citation_id) && (
                <div className="citation-content">
                  <p className="text-snippet">{citation.text_snippet}</p>
                  <div className="citation-details">
                    <span>Words: {citation.word_count}</span>
                    <span>Position: {citation.position}</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderSources = () => {
    const { sources } = answerData;
    if (!sources || sources.length === 0) return null;

    return (
      <div className="sources-section">
        <h3>📖 Source Documents</h3>
        <div className="sources-list">
          {sources.map((source, index) => (
            <div key={index} className="source-item">
              <div className="source-header">
                <h4>{source.title}</h4>
                <span className="source-score">Score: {source.relevance_score.toFixed(3)}</span>
              </div>
              <div className="source-meta">
                <span>Source: {source.source}</span>
                <span>Chunk: {source.chunk_id}</span>
                <span>Words: {source.word_count}</span>
              </div>
              <div className="source-text">
                {source.text.length > 300 
                  ? `${source.text.substring(0, 300)}...` 
                  : source.text
                }
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderMetadata = () => {
    const { metadata } = answerData;
    if (!metadata) return null;

    return (
      <div className="metadata-section">
        <h3>📊 Query Information</h3>
        <div className="metadata-grid">
          <div className="metadata-item">
            <span className="label">Processing Time:</span>
            <span className="value">{metadata.total_processing_time?.toFixed(2)}s</span>
          </div>
          <div className="metadata-item">
            <span className="label">Documents Used:</span>
            <span className="value">{metadata.documents_used}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Citations Found:</span>
            <span className="value">{metadata.citations_found}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Average Score:</span>
            <span className="value">{metadata.average_relevance_score?.toFixed(3)}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Query Length:</span>
            <span className="value">{metadata.query_length} chars</span>
          </div>
          <div className="metadata-item">
            <span className="label">Answer Length:</span>
            <span className="value">{metadata.answer_length} chars</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="answer-panel">
      <div className="answer-header">
        <h2>🤖 AI Answer with Citations</h2>
        {answerData.query && (
          <div className="query-display">
            <strong>Question:</strong> {answerData.query}
          </div>
        )}
      </div>

      <div className="answer-body">
        {renderAnswer()}
        {renderCitations()}
        {renderSources()}
        {renderMetadata()}
      </div>

      {/* Source Detail Modal */}
      {selectedSource && (
        <div className="source-modal" onClick={() => setSelectedSource(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Source Details</h3>
              <button className="close-btn" onClick={() => setSelectedSource(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="source-detail">
                <h4>{selectedSource.title}</h4>
                <p><strong>Source:</strong> {selectedSource.source}</p>
                <p><strong>Chunk ID:</strong> {selectedSource.chunk_id}</p>
                <p><strong>Relevance Score:</strong> {selectedSource.relevance_score.toFixed(3)}</p>
                <p><strong>Word Count:</strong> {selectedSource.word_count}</p>
                <div className="full-text">
                  <h5>Full Text:</h5>
                  <p>{selectedSource.text_snippet}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnswerPanel;
