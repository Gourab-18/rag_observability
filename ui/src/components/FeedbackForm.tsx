import { useState } from 'react'
import axios from 'axios'
import { ThumbsUp, ThumbsDown, Send } from 'lucide-react'

interface FeedbackFormProps {
  traceId: string
  query?: string
  answer?: string
}

// Use relative path for Vite proxy, or absolute URL if specified
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

export default function FeedbackForm({ traceId, query, answer }: FeedbackFormProps) {
  const [rating, setRating] = useState<number | null>(null)
  const [thumbsUp, setThumbsUp] = useState<boolean | null>(null)
  const [textFeedback, setTextFeedback] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!rating && thumbsUp === null && !textFeedback.trim()) {
      return
    }

    setSubmitting(true)
    
    try {
      await axios.post(
        `${API_BASE_URL}/api/feedback`,
        {
          trace_id: traceId,
          rating: rating || undefined,
          thumbs_up: thumbsUp === true ? true : undefined,
          thumbs_down: thumbsUp === false ? true : undefined,
          text_feedback: textFeedback.trim() || undefined,
          query,
          answer,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      )
      
      setSubmitted(true)
      setTimeout(() => {
        setSubmitted(false)
        setRating(null)
        setThumbsUp(null)
        setTextFeedback('')
      }, 3000)
    } catch (error) {
      console.error('Error submitting feedback:', error)
      alert('Failed to submit feedback. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  if (submitted) {
    return (
      <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
        <p className="text-sm text-green-800">Thank you for your feedback!</p>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="mt-4 pt-4 border-t">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-600">Helpful?</span>
          <button
            type="button"
            onClick={() => setThumbsUp(thumbsUp === true ? null : true)}
            className={`p-1 rounded ${
              thumbsUp === true
                ? 'bg-green-100 text-green-600'
                : 'text-gray-400 hover:text-green-600'
            }`}
          >
            <ThumbsUp className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setThumbsUp(thumbsUp === false ? null : false)}
            className={`p-1 rounded ${
              thumbsUp === false
                ? 'bg-red-100 text-red-600'
                : 'text-gray-400 hover:text-red-600'
            }`}
          >
            <ThumbsDown className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-600">Rating:</span>
          {[1, 2, 3, 4, 5].map((num) => (
            <button
              key={num}
              type="button"
              onClick={() => setRating(rating === num ? null : num)}
              className={`w-6 h-6 rounded text-xs ${
                rating === num
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {num}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-3 flex space-x-2">
        <input
          type="text"
          value={textFeedback}
          onChange={(e) => setTextFeedback(e.target.value)}
          placeholder="Additional feedback (optional)..."
          className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={submitting || (!rating && thumbsUp === null && !textFeedback.trim())}
          className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-1"
        >
          <Send className="h-3 w-3" />
          <span>Submit</span>
        </button>
      </div>
    </form>
  )
}

