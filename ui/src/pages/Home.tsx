import { useState } from 'react'
import ChatInterface from '../components/ChatInterface'
import FileUpload from '../components/FileUpload'
import { Link } from 'react-router-dom'
import { Settings, Upload, MessageSquare } from 'lucide-react'

export default function Home() {
  const [showUpload, setShowUpload] = useState(false)

  return (
    <div className="h-screen flex flex-col">
      <div className="flex items-center justify-between bg-white border-b px-4 py-2">
        <h1 className="text-xl font-bold text-gray-900">RAG Pipeline</h1>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowUpload(!showUpload)}
            className="flex items-center space-x-1 text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded hover:bg-gray-100"
          >
            <Upload className="h-4 w-4" />
            <span>{showUpload ? 'Hide Upload' : 'Upload File'}</span>
          </button>
          <Link
            to="/admin"
            className="flex items-center space-x-1 text-sm text-gray-600 hover:text-gray-900"
          >
            <Settings className="h-4 w-4" />
            <span>Admin</span>
          </Link>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden flex">
        {/* Upload Panel */}
        {showUpload && (
          <div className="w-96 border-r bg-gray-50 overflow-y-auto p-4">
            <FileUpload
              onUploadSuccess={() => {
                setShowUpload(false)
                // Optionally refresh chat or show success message
              }}
            />
            <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> After uploading, wait a few seconds for indexing to complete before querying.
              </p>
            </div>
          </div>
        )}
        
        {/* Chat Interface */}
        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          <ChatInterface />
        </div>
      </div>
    </div>
  )
}

