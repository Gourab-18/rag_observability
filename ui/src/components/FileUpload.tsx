import { useState } from 'react'
import axios from 'axios'
import { Upload, File, X, CheckCircle } from 'lucide-react'

// Use relative path for Vite proxy
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

interface UploadResult {
  success: boolean
  message: string
  doc_id: string
  chunks: number
  file_path: string
}

export default function FileUpload({ onUploadSuccess }: { onUploadSuccess?: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      const ext = selectedFile.name.split('.').pop()?.toLowerCase()
      
      if (!['pdf', 'txt', 'md'].includes(ext || '')) {
        setError('Only PDF, TXT, and MD files are allowed')
        return
      }
      
      setFile(selectedFile)
      setError(null)
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      // Estimate time based on file size (rough: 1KB ≈ 1 chunk, 0.5s per chunk)
      const estimatedChunks = Math.max(10, Math.floor(file.size / 1000))
      const estimatedSeconds = Math.ceil(estimatedChunks * 0.5)

      const response = await axios.post<UploadResult>(
        `${API_BASE_URL}/api/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: (estimatedSeconds + 30) * 1000, // Add 30s buffer
        }
      )

      setResult(response.data)
      setFile(null)
      
      // Reset file input
      const fileInput = document.getElementById('file-input') as HTMLInputElement
      if (fileInput) fileInput.value = ''

      if (onUploadSuccess) {
        onUploadSuccess()
      }
    } catch (err: any) {
      if (err.code === 'ECONNABORTED') {
        setError('Upload timed out. The file may be too large. Please try a smaller file or wait longer.')
      } else {
        const errorMessage = err.response?.data?.detail || err.message || 'Upload failed'
        setError(errorMessage)
      }
    } finally {
      setUploading(false)
    }
  }

  const handleRemove = () => {
    setFile(null)
    setError(null)
    setResult(null)
    const fileInput = document.getElementById('file-input') as HTMLInputElement
    if (fileInput) fileInput.value = ''
  }

  return (
    <div className="bg-white rounded-lg border p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Document</h3>
      
      {!result ? (
        <>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <input
              id="file-input"
              type="file"
              accept=".pdf,.txt,.md"
              onChange={handleFileChange}
              className="hidden"
            />
            
            {!file ? (
              <label
                htmlFor="file-input"
                className="cursor-pointer flex flex-col items-center space-y-2"
              >
                <Upload className="h-8 w-8 text-gray-400" />
                <span className="text-sm text-gray-600">
                  Click to upload or drag and drop
                </span>
                <span className="text-xs text-gray-500">
                  PDF, TXT, or MD files only
                </span>
              </label>
            ) : (
              <div className="flex items-center justify-between bg-gray-50 rounded p-3">
                <div className="flex items-center space-x-2">
                  <File className="h-5 w-5 text-gray-500" />
                  <span className="text-sm font-medium text-gray-900">{file.name}</span>
                  <span className="text-xs text-gray-500">
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <button
                  onClick={handleRemove}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="mt-4 w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {uploading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Processing... (this may take 30-60 seconds)</span>
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" />
                <span>Upload and Index</span>
              </>
            )}
          </button>
          
          {uploading && file && (
            <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-xs text-blue-800">
                <strong>Processing steps:</strong><br/>
                1. Uploading file... ✓<br/>
                2. Chunking document... (fast)<br/>
                3. Generating embeddings... (slow, ~0.5s per chunk)<br/>
                4. Indexing in Qdrant... (fast)<br/>
                <br/>
                <strong>Estimated time:</strong> ~{Math.ceil((file.size / 1000) * 0.5)} seconds
              </p>
            </div>
          )}
        </>
      ) : (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="font-medium text-green-900">Upload Successful!</span>
          </div>
          <div className="text-sm text-green-800 space-y-1">
            <p>Document ID: <code className="bg-green-100 px-1 rounded">{result.doc_id}</code></p>
            <p>Chunks created: {result.chunks}</p>
            <p className="mt-2">{result.message}</p>
          </div>
          <button
            onClick={handleRemove}
            className="mt-3 text-sm text-green-700 hover:text-green-900 underline"
          >
            Upload another file
          </button>
        </div>
      )}
    </div>
  )
}

