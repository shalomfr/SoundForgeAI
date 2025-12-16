import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, SkipBack } from 'lucide-react'

export default function BeforeAfter({ originalUrl, processedUrl }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [showProcessed, setShowProcessed] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  
  const originalRef = useRef(null)
  const processedRef = useRef(null)

  useEffect(() => {
    if (originalRef.current) {
      originalRef.current.addEventListener('loadedmetadata', () => {
        setDuration(originalRef.current.duration)
      })
      originalRef.current.addEventListener('timeupdate', () => {
        setCurrentTime(originalRef.current.currentTime)
      })
      originalRef.current.addEventListener('ended', () => {
        setIsPlaying(false)
      })
    }
  }, [originalUrl])

  const togglePlay = () => {
    if (isPlaying) {
      originalRef.current?.pause()
      processedRef.current?.pause()
    } else {
      if (showProcessed) {
        processedRef.current?.play()
      } else {
        originalRef.current?.play()
      }
    }
    setIsPlaying(!isPlaying)
  }

  const toggleVersion = () => {
    const wasPlaying = isPlaying
    const time = showProcessed ? processedRef.current?.currentTime : originalRef.current?.currentTime
    
    // Pause both
    originalRef.current?.pause()
    processedRef.current?.pause()
    
    // Switch
    setShowProcessed(!showProcessed)
    
    // Sync time and resume if was playing
    setTimeout(() => {
      if (!showProcessed) {
        if (processedRef.current) {
          processedRef.current.currentTime = time || 0
          if (wasPlaying) processedRef.current.play()
        }
      } else {
        if (originalRef.current) {
          originalRef.current.currentTime = time || 0
          if (wasPlaying) originalRef.current.play()
        }
      }
    }, 50)
  }

  const restart = () => {
    if (originalRef.current) originalRef.current.currentTime = 0
    if (processedRef.current) processedRef.current.currentTime = 0
    setCurrentTime(0)
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const seekTo = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = x / rect.width
    const newTime = percentage * duration
    
    if (originalRef.current) originalRef.current.currentTime = newTime
    if (processedRef.current) processedRef.current.currentTime = newTime
  }

  return (
    <motion.div
      className="glass rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="font-display font-semibold text-lg mb-4">
        Compare Before / After
      </h3>

      {/* Hidden audio elements */}
      <audio ref={originalRef} src={originalUrl} preload="auto" />
      <audio ref={processedRef} src={processedUrl} preload="auto" />

      {/* Toggle switch */}
      <div className="flex items-center justify-center gap-4 mb-6">
        <span className={`font-medium transition-colors ${!showProcessed ? 'text-white' : 'text-white/40'}`}>
          Original
        </span>
        <motion.button
          onClick={toggleVersion}
          className="w-20 h-10 rounded-full p-1 relative"
          style={{
            background: showProcessed 
              ? 'linear-gradient(135deg, #00d9ff 0%, #a855f7 100%)'
              : '#242430'
          }}
        >
          <motion.div
            className="w-8 h-8 rounded-full bg-white shadow-lg"
            animate={{ x: showProcessed ? 40 : 0 }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
          />
        </motion.button>
        <span className={`font-medium transition-colors ${showProcessed ? 'text-accent-cyan' : 'text-white/40'}`}>
          Enhanced
        </span>
      </div>

      {/* Progress bar */}
      <div 
        className="h-2 bg-elevated rounded-full overflow-hidden cursor-pointer mb-4"
        onClick={seekTo}
      >
        <motion.div
          className="h-full rounded-full"
          style={{
            background: showProcessed 
              ? 'linear-gradient(90deg, #00d9ff, #a855f7)'
              : '#6b7280',
            width: `${(currentTime / duration) * 100}%`
          }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={restart}
          className="w-10 h-10 rounded-full bg-elevated flex items-center justify-center hover:bg-surface transition-colors"
        >
          <SkipBack className="w-5 h-5" />
        </button>
        
        <motion.button
          onClick={togglePlay}
          className="w-14 h-14 rounded-full flex items-center justify-center"
          style={{
            background: showProcessed 
              ? 'linear-gradient(135deg, #00d9ff 0%, #a855f7 100%)'
              : '#6b7280'
          }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isPlaying ? (
            <Pause className="w-6 h-6 text-void" />
          ) : (
            <Play className="w-6 h-6 text-void ml-1" />
          )}
        </motion.button>

        <div className="w-10 h-10 flex items-center justify-center">
          <span className="text-xs text-white/40 font-mono">
            {formatTime(currentTime)}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

