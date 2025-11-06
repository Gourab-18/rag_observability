import { useState } from 'react'
import axios from 'axios'
import { Trash2, AlertTriangle, CheckCircle } from 'lucide-react'

// Use relative path for Vite proxy
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

interface ClearResult {
  success: boolean
  message: string
  deleted: {
    qdrant_points: number
    processed_files: number
  }
}

export default function ClearContextButton({ onClearSuccess }: { onClearSuccess?: () => void }) {
  const [loading, setLoading] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [result, setResult] = useState<ClearResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleClear = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.delete<ClearResult>(
        `${API_BASE_URL}/api/admin/clear-all`
      )

      setResult(response.data)
      setShowConfirm(false)
      
      if (onClearSuccess) {
        onClearSuccess()
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to clear context'
      setError(errorMessage)
      setShowConfirm(false)
    } finally {
      setLoading(false)
    }
  }

  if (result) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <CheckCircle className="h-5 w-5 text-green-600" />
          <span className="font-medium text-green-900">Context Cleared Successfully!</span>
        </div>
        <div className="text-sm text-green-800 space-y-1">
          <p>Deleted {result.deleted.qdrant_points} chunks from Qdrant</p>
          <p>Removed {result.deleted.processed_files} processed files</p>
          <p className="mt-2">{result.message}</p>
        </div>
        <button
          onClick={() => {
            setResult(null)
            setShowConfirm(false)
          }}
          className="mt-3 text-sm text-green-700 hover:text-green-900 underline"
        >
          Close
        </button>
      </div>
    )
  }

  if (showConfirm) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-3">
          <AlertTriangle className="h-5 w-5 text-red-600" />
          <span className="font-medium text-red-900">Confirm Clear All Context</span>
        </div>
        <p className="text-sm text-red-800 mb-4">
          This will permanently delete:
          <br />• All indexed documents from Qdrant
          <br />• All processed chunk files
          <br />• The manifest file
          <br />
          <br />
          <strong>This action cannot be undone!</strong>
        </p>
        <div className="flex space-x-2">
          <button
            onClick={handleClear}
            disabled={loading}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Clearing...</span>
              </>
            ) : (
              <>
                <Trash2 className="h-4 w-4" />
                <span>Yes, Clear Everything</span>
              </>
            )}
          </button>
          <button
            onClick={() => {
              setShowConfirm(false)
              setError(null)
            }}
            disabled={loading}
            className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <button
        onClick={() => setShowConfirm(true)}
        className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center space-x-2"
      >
        <Trash2 className="h-4 w-4" />
        <span>Clear All Context</span>
      </button>
      
      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}
    </div>
  )
}

