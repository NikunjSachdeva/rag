import React, { useState } from 'react';

const AnswerPanel = ({ answerData, isLoading, error }) => {
  const [expandedCitations, setExpandedCitations] = useState(new Set());
  const [selectedSource, setSelectedSource] = useState(null);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-8">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mb-4"></div>
          <p className="text-lg font-semibold text-gray-700">🤖 Generating answer with citations...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-2xl p-8 border border-red-200">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
            <span className="text-2xl">❌</span>
          </div>
          <h3 className="text-xl font-bold text-red-800 mb-2">Error Occurred</h3>
          <p className="text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!answerData || !answerData.answer) {
    return (
      <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-8 text-center">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-blue-100 rounded-full mb-6">
          <span className="text-3xl">🔍</span>
        </div>
        <h3 className="text-2xl font-bold text-gray-800 mb-3">No Results Yet</h3>
        <p className="text-gray-600 text-lg">
          Upload some documents and ask a question to get started!
        </p>
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
    const sortedCitations = [...citations].sort((a, b) => a.citation_id - b.citation_id);
    
    sortedCitations.forEach((citation) => {
      const marker = `[${citation.citation_id}]`;
      const replacement = `<span class="citation-marker inline-flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-800 font-semibold rounded-full text-xs cursor-pointer hover:bg-blue-200 transition-colors duration-200 mx-1" data-citation-id="${citation.citation_id}">${citation.citation_id}</span>`;
      formattedAnswer = formattedAnswer.replace(marker, replacement);
    });

    return formattedAnswer;
  };

  const renderAnswer = () => {
    const { answer, citations } = answerData;
    const formattedAnswer = formatAnswerWithCitations(answer, citations);

    return (
      <div 
        className="bg-white rounded-xl p-6 shadow-sm border border-gray-100"
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
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center mr-3">
            <span className="text-purple-600">📚</span>
          </div>
          <h3 className="text-xl font-bold text-gray-800">Citations & Sources</h3>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {citations.map((citation) => (
            <div key={citation.citation_id} className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:shadow-md transition-shadow duration-200">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <span className="inline-flex items-center justify-center w-8 h-8 bg-blue-100 text-blue-800 font-semibold rounded-full text-sm">
                    {citation.citation_id}
                  </span>
                  <span className="text-sm font-medium text-green-600 bg-green-50 px-2 py-1 rounded-full">
                    Score: {citation.relevance_score?.toFixed(3)}
                  </span>
                </div>
                <button
                  className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-colors duration-200"
                  onClick={() => toggleCitation(citation.citation_id)}
                >
                  {expandedCitations.has(citation.citation_id) ? '−' : '+'}
                </button>
              </div>
              
              <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                <span className="bg-gray-100 px-2 py-1 rounded">📁 {citation.source}</span>
                <span className="bg-gray-100 px-2 py-1 rounded"># {citation.chunk_id}</span>
              </div>
              
              {expandedCitations.has(citation.citation_id) && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <p className="text-gray-700 text-sm leading-relaxed mb-3">
                    {citation.text_snippet}
                  </p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>📝 {citation.word_count} words</span>
                    <span>📍 Pos: {citation.position}</span>
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
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center mr-3">
            <span className="text-orange-600">📖</span>
          </div>
          <h3 className="text-xl font-bold text-gray-800">Source Documents</h3>
        </div>
        <div className="grid gap-4">
          {sources.map((source, index) => (
            <div key={index} className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:shadow-md transition-shadow duration-200">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-gray-800 text-lg">{source.title}</h4>
                <span className="text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full">
                  Score: {source.relevance_score?.toFixed(3)}
                </span>
              </div>
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">📁 {source.source}</span>
                <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded"># {source.chunk_id}</span>
                <span className="bg-green-100 text-green-700 text-xs px-2 py-1 rounded">📝 {source.word_count} words</span>
              </div>
              <div className="text-gray-600 text-sm leading-relaxed bg-gray-50 p-4 rounded-lg">
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
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center mr-3">
            <span className="text-indigo-600">📊</span>
          </div>
          <h3 className="text-xl font-bold text-gray-800">Query Information</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {[
            { label: 'Processing Time', value: `${metadata.total_processing_time?.toFixed(2)}s`, icon: '⏱️' },
            { label: 'Documents Used', value: metadata.documents_used, icon: '📄' },
            { label: 'Citations Found', value: metadata.citations_found, icon: '🔗' },
            { label: 'Average Score', value: metadata.average_relevance_score?.toFixed(3), icon: '⭐' },
            { label: 'Query Length', value: `${metadata.query_length} chars`, icon: '📏' },
            { label: 'Answer Length', value: `${metadata.answer_length} chars`, icon: '📝' }
          ].map((item, index) => (
            <div key={index} className="bg-white rounded-xl p-4 text-center shadow-sm border border-gray-100">
              <div className="text-2xl mb-2">{item.icon}</div>
              <div className="text-sm text-gray-600 font-medium mb-1">{item.label}</div>
              <div className="text-lg font-bold text-gray-800">{item.value}</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-6">
      <div className="mb-8">
        <div className="flex items-center mb-4">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mr-3">
            <span className="text-white text-xl">🤖</span>
          </div>
          <h2 className="text-2xl font-bold text-gray-800">AI Answer with Citations</h2>
        </div>
        {answerData.query && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <strong className="text-blue-800">Question:</strong>
            <p className="text-blue-900 mt-1">{answerData.query}</p>
          </div>
        )}
      </div>

      <div className="space-y-6">
        {renderAnswer()}
        {renderCitations()}
        {renderSources()}
        {renderMetadata()}
      </div>

      {selectedSource && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedSource(null)}>
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <h3 className="text-xl font-bold text-gray-800">Source Details</h3>
              <button className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center" onClick={() => setSelectedSource(null)}>
                ×
              </button>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-gray-800">{selectedSource.title}</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <span className="text-sm text-gray-600">Source</span>
                    <p className="font-medium">{selectedSource.source}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <span className="text-sm text-gray-600">Chunk ID</span>
                    <p className="font-medium">{selectedSource.chunk_id}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <span className="text-sm text-gray-600">Relevance Score</span>
                    <p className="font-medium text-green-600">{selectedSource.relevance_score?.toFixed(3)}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <span className="text-sm text-gray-600">Word Count</span>
                    <p className="font-medium">{selectedSource.word_count}</p>
                  </div>
                </div>
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 mb-2">Full Text:</h5>
                  <p className="text-gray-600 bg-gray-50 p-4 rounded-lg leading-relaxed">
                    {selectedSource.text_snippet}
                  </p>
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