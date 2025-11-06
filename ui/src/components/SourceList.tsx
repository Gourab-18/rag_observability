import { useState } from 'react'
import { Source } from '../hooks/useRAGQuery'
import { ExternalLink, FileText } from 'lucide-react'

interface SourceListProps {
  sources: Source[]
}

export default function SourceList({ sources }: SourceListProps) {
  const [expandedSource, setExpandedSource] = useState<string | null>(null)

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const getConfidenceLabel = (score: number) => {
    if (score >= 0.8) return 'High'
    if (score >= 0.6) return 'Medium'
    return 'Low'
  }

  return (
    <div className="mt-4 space-y-2">
      <h4 className="text-sm font-semibold text-gray-700 mb-2">Sources ({sources.length})</h4>
      <div className="space-y-2">
        {sources.map((source, idx) => (
          <div
            key={source.chunk_id || idx}
            className="border border-gray-200 rounded-lg p-3 hover:border-gray-300 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <FileText className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium text-gray-900">
                    Source {idx + 1}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${getConfidenceColor(source.score)}`}>
                    {getConfidenceLabel(source.score)} ({(source.score * 100).toFixed(0)}%)
                  </span>
                </div>
                
                <div className="text-xs text-gray-600 mb-2">
                  <span>Doc: {source.doc_id}</span>
                  {source.metadata?.source && (
                    <span className="ml-2">• File: {source.metadata.source}</span>
                  )}
                </div>

                {expandedSource === source.chunk_id ? (
                  <div className="mt-2">
                    <p className="text-sm text-gray-700 whitespace-pre-wrap bg-gray-50 p-2 rounded">
                      {source.text}
                    </p>
                    <button
                      onClick={() => setExpandedSource(null)}
                      className="mt-2 text-xs text-blue-600 hover:text-blue-800"
                    >
                      Show less
                    </button>
                  </div>
                ) : (
                  <div className="mt-2">
                    <p className="text-sm text-gray-700 line-clamp-2">
                      {source.text.substring(0, 200)}
                      {source.text.length > 200 && '...'}
                    </p>
                    <button
                      onClick={() => setExpandedSource(source.chunk_id)}
                      className="mt-1 text-xs text-blue-600 hover:text-blue-800"
                    >
                      Show more
                    </button>
                  </div>
                )}
              </div>

              {source.metadata?.source && (
                <a
                  href={source.metadata.source}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-2 text-gray-400 hover:text-gray-600"
                  title="Open source document"
                >
                  <ExternalLink className="h-4 w-4" />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

