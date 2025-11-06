import { useState } from 'react'
import axios from 'axios'
import { TestTube, TrendingUp, DollarSign, Clock, CheckCircle, XCircle } from 'lucide-react'

// Use relative path for Vite proxy
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

interface ABTestRequest {
  query: string
  variants?: string[]
  top_k?: number
}

interface ABTestResponse {
  query_id: string
  query: string
  prompt_version_a: string
  prompt_version_b: string
  winner: string | null
  metrics_a: {
    answer_length: number
    num_sources: number
    token_usage: Record<string, number>
    cost_estimate: number
    latency: number
    model_used: string
  }
  metrics_b: {
    answer_length: number
    num_sources: number
    token_usage: Record<string, number>
    cost_estimate: number
    latency: number
    model_used: string
  }
  response_a: {
    answer: string
    sources: number
    trace_id: string
  }
  response_b: {
    answer: string
    sources: number
    trace_id: string
  }
  message: string
}

export default function ABTestPanel() {
  const [query, setQuery] = useState('')
  const [variants, setVariants] = useState(['v1', 'v2'])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ABTestResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleTest = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post<ABTestResponse>(
        `${API_BASE_URL}/api/ab-test`,
        {
          query: query.trim(),
          variants: variants,
          top_k: 5
        }
      )

      setResult(response.data)
    } catch (err: any) {
      let errorMessage = err.response?.data?.detail || err.message || 'A/B test failed'
      
      // Provide helpful error messages
      if (errorMessage.includes('No relevant chunks found')) {
        errorMessage = `No documents found for this query. Please try:
        • A different query related to your indexed documents
        • Ensure documents are uploaded and indexed
        • Check that Qdrant has data (322 chunks currently indexed)`
      }
      
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <TestTube className="h-5 w-5 mr-2 text-purple-600" />
        A/B Testing: Prompt Version Comparison
      </h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Test Query
          </label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter a query to test..."
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            disabled={loading}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prompt Versions to Compare
          </label>
          <div className="flex space-x-2">
            <input
              type="text"
              value={variants[0] || ''}
              onChange={(e) => setVariants([e.target.value, variants[1] || 'v2'])}
              placeholder="v1"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={loading}
            />
            <span className="self-center text-gray-500">vs</span>
            <input
              type="text"
              value={variants[1] || ''}
              onChange={(e) => setVariants([variants[0] || 'v1', e.target.value])}
              placeholder="v2"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={loading}
            />
          </div>
        </div>

        <button
          onClick={handleTest}
          disabled={loading || !query.trim()}
          className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Running A/B Test...</span>
            </>
          ) : (
            <>
              <TestTube className="h-4 w-4" />
              <span>Run A/B Test</span>
            </>
          )}
        </button>

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-6 space-y-4">
            <div className="flex items-center justify-between p-3 bg-purple-50 border border-purple-200 rounded-lg">
              <span className="font-medium text-purple-900">Winner:</span>
              <span className="text-lg font-bold text-purple-700">
                {result.winner 
                  ? `Version ${result.winner.toUpperCase()} (${result[`prompt_version_${result.winner}`]})`
                  : 'Tie'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Version A */}
              <div className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-gray-900">
                    Version {result.prompt_version_a}
                  </h3>
                  {result.winner === 'a' && (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  )}
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <DollarSign className="h-4 w-4 text-gray-500" />
                    <span>Cost: ${result.metrics_a.cost_estimate.toFixed(4)}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-gray-500" />
                    <span>Latency: {result.metrics_a.latency.toFixed(2)}s</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-gray-500" />
                    <span>Length: {result.metrics_a.answer_length} chars</span>
                  </div>
                  <div className="text-gray-600">
                    Sources: {result.metrics_a.num_sources}
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t">
                  <p className="text-xs text-gray-600 line-clamp-3">
                    {result.response_a.answer.substring(0, 150)}...
                  </p>
                </div>
              </div>

              {/* Version B */}
              <div className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-gray-900">
                    Version {result.prompt_version_b}
                  </h3>
                  {result.winner === 'b' && (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  )}
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <DollarSign className="h-4 w-4 text-gray-500" />
                    <span>Cost: ${result.metrics_b.cost_estimate.toFixed(4)}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-gray-500" />
                    <span>Latency: {result.metrics_b.latency.toFixed(2)}s</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-gray-500" />
                    <span>Length: {result.metrics_b.answer_length} chars</span>
                  </div>
                  <div className="text-gray-600">
                    Sources: {result.metrics_b.num_sources}
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t">
                  <p className="text-xs text-gray-600 line-clamp-3">
                    {result.response_b.answer.substring(0, 150)}...
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

