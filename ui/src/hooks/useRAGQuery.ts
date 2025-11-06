import { useState } from 'react'
import axios from 'axios'

// Use relative path for Vite proxy, or absolute URL if specified
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

export interface Source {
  chunk_id: string
  doc_id: string
  text: string
  score: number
  metadata: Record<string, any>
}

export interface QueryResponse {
  answer: string
  sources: Source[]
  trace_id: string
  latency: number
  token_usage: Record<string, number>
  cost_estimate: number
  model_used: string
  prompt_version: string
}

export interface QueryRequest {
  query: string
  top_k?: number
  prompt_version?: string
  use_cache?: boolean
}

export function useRAGQuery() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<QueryResponse | null>(null)

  const query = async (request: QueryRequest) => {
    setLoading(true)
    setError(null)
    
    try {
      const result = await axios.post<QueryResponse>(
        `${API_BASE_URL}/api/query`,
        request,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      )
      
      setResponse(result.data)
      return result.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred'
      setError(errorMessage)
      throw err
    } finally {
      setLoading(false)
    }
  }

  return { query, loading, error, response }
}

