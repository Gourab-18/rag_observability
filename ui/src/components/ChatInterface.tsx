import { useState } from 'react'
import { useRAGQuery, QueryResponse } from '../hooks/useRAGQuery'
import SourceList from './SourceList'
import FeedbackForm from './FeedbackForm'

export default function ChatInterface() {
  const [query, setQuery] = useState('')
  const [history, setHistory] = useState<QueryResponse[]>([])
  const { query: sendQuery, loading, error, response } = useRAGQuery()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    try {
      const result = await sendQuery({
        query: query.trim(),
        top_k: 5,
        prompt_version: 'v1',
        use_cache: true,
      })
      
      if (result) {
        // Store query with response for display
        const responseWithQuery = { ...result, query: query.trim() }
        setHistory([...history, responseWithQuery])
        setQuery('')
      }
    } catch (err) {
      console.error('Query error:', err)
    }
  }

  const currentResponse = response || (history.length > 0 ? history[history.length - 1] : null)

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b flex-shrink-0">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">RAG Pipeline Chat</h1>
          <p className="text-sm text-gray-600">Ask questions about your documents</p>
        </div>
      </header>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-4 py-6 min-h-0">
        <div className="max-w-4xl mx-auto space-y-6">
          {history.map((item, idx) => {
            // Store query with response for display
            const queryText = (item as any).query || 'Previous query'
            return (
            <div key={idx} className="space-y-4">
              {/* User Query */}
              <div className="flex justify-end">
                <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-2xl">
                  <p className="text-sm font-medium">You</p>
                  <p className="mt-1">{queryText}</p>
                </div>
              </div>

              {/* Assistant Response */}
              <div className="flex justify-start">
                <div className="bg-white rounded-lg shadow-sm border px-6 py-4 max-w-2xl">
                  <p className="text-sm font-medium text-gray-700 mb-2">Assistant</p>
                  <div className="prose prose-sm max-w-none">
                    <p className="text-gray-900 whitespace-pre-wrap">{item.answer}</p>
                  </div>
                  
                  {/* Metadata */}
                  <div className="mt-4 pt-4 border-t text-xs text-gray-500 space-y-1">
                    <p>Trace ID: <code className="bg-gray-100 px-1 rounded">{item.trace_id}</code></p>
                    <p>Latency: {item.latency.toFixed(2)}s | Cost: ${item.cost_estimate.toFixed(4)} | Model: {item.model_used}</p>
                  </div>

                  {/* Sources */}
                  {item.sources && item.sources.length > 0 && (
                    <div className="mt-4">
                      <SourceList sources={item.sources} />
                    </div>
                  )}

                  {/* Feedback */}
                  <div className="mt-4">
                    <FeedbackForm traceId={item.trace_id} query={(item as any).query} answer={item.answer} />
                  </div>
                </div>
              </div>
            </div>
            )
          })}

          {/* Current Response (if loading or just received) */}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white rounded-lg shadow-sm border px-6 py-4">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <p className="text-gray-600">Processing your query...</p>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="flex justify-start">
              <div className="bg-red-50 border border-red-200 rounded-lg px-6 py-4">
                <p className="text-red-800 font-medium">Error</p>
                <p className="text-red-600 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white border-t flex-shrink-0">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

