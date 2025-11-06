import { Link } from 'react-router-dom'
import MetricsDashboard from '../components/MetricsDashboard'
import ClearContextButton from '../components/ClearContextButton'
import { ArrowLeft } from 'lucide-react'

export default function Admin() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link
                to="/"
                className="flex items-center space-x-1 text-gray-600 hover:text-gray-900"
              >
                <ArrowLeft className="h-4 w-4" />
                <span>Back to Chat</span>
              </Link>
              <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
        <MetricsDashboard />
        
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Database Management</h2>
          <p className="text-sm text-gray-600 mb-4">
            Clear all indexed context from the database to start fresh. This will remove all documents, chunks, and the manifest.
          </p>
          <ClearContextButton />
        </div>
      </div>
    </div>
  )
}

