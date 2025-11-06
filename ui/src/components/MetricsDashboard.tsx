import { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, DollarSign, Clock, TrendingUp } from 'lucide-react'

// Use relative path for Vite proxy, or absolute URL if specified
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

interface Metrics {
  cache: {
    embeddings: number
    retrieval: number
    llm: number
    total: number
    hit_rate: number
    target_hit_rate: number
  }
  performance: {
    target_latency_p95: number
    target_latency_p99: number
    target_cache_hit_rate: number
  }
  quality: {
    target_hallucination_rate: number
    target_token_reduction: number
  }
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  const fetchMetrics = async () => {
    try {
      const response = await axios.get<Metrics>(`${API_BASE_URL}/api/metrics`, {
        headers: {
          'Content-Type': 'application/json',
        },
      })
      setMetrics(response.data)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch metrics')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <p className="text-red-800">Error: {error}</p>
      </div>
    )
  }

  if (!metrics) return null

  const cacheHitRatePercent = (metrics.cache.hit_rate * 100).toFixed(1)
  const targetHitRatePercent = (metrics.cache.target_hit_rate * 100).toFixed(0)

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Metrics Dashboard</h2>

      {/* Cache Metrics */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Activity className="h-5 w-5 mr-2 text-blue-600" />
          Cache Statistics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Embeddings</p>
            <p className="text-2xl font-bold text-gray-900">{metrics.cache.embeddings}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Retrieval</p>
            <p className="text-2xl font-bold text-gray-900">{metrics.cache.retrieval}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">LLM Responses</p>
            <p className="text-2xl font-bold text-gray-900">{metrics.cache.llm}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Total Cached</p>
            <p className="text-2xl font-bold text-gray-900">{metrics.cache.total}</p>
          </div>
        </div>
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Cache Hit Rate</span>
            <span className="text-sm font-medium text-gray-900">
              {cacheHitRatePercent}% / {targetHitRatePercent}% target
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                metrics.cache.hit_rate >= metrics.cache.target_hit_rate
                  ? 'bg-green-600'
                  : 'bg-yellow-600'
              }`}
              style={{ width: `${Math.min(metrics.cache.hit_rate * 100, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Performance Targets */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Clock className="h-5 w-5 mr-2 text-blue-600" />
          Performance Targets
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">P95 Latency Target</p>
            <p className="text-xl font-bold text-gray-900">
              {metrics.performance.target_latency_p95}s
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">P99 Latency Target</p>
            <p className="text-xl font-bold text-gray-900">
              {metrics.performance.target_latency_p99}s
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Cache Hit Rate Target</p>
            <p className="text-xl font-bold text-gray-900">
              {(metrics.performance.target_cache_hit_rate * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      {/* Quality Targets */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-blue-600" />
          Quality Targets
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Max Hallucination Rate</p>
            <p className="text-xl font-bold text-gray-900">
              {(metrics.quality.target_hallucination_rate * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">Token Reduction Target</p>
            <p className="text-xl font-bold text-gray-900">
              {(metrics.quality.target_token_reduction * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-800">
          💡 For detailed metrics, visit{' '}
          <a
            href="http://localhost:3000"
            target="_blank"
            rel="noopener noreferrer"
            className="underline font-medium"
          >
            Grafana Dashboard
          </a>{' '}
          or{' '}
          <a
            href="http://localhost:9090"
            target="_blank"
            rel="noopener noreferrer"
            className="underline font-medium"
          >
            Prometheus
          </a>
        </p>
      </div>
    </div>
  )
}

