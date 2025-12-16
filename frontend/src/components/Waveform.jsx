import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import WaveSurfer from 'wavesurfer.js'
import { Play, Pause, Volume2 } from 'lucide-react'

export default function Waveform({ 
  url, 
  label, 
  color = '#00d9ff',
  isEmpty = false,
  isProcessing = false,
  isAnalyzing = false
}) {
  const containerRef = useRef(null)
  const wavesurferRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)

  useEffect(() => {
    if (!containerRef.current || !url) return

    // Destroy previous instance
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy()
    }

    // Create new instance
    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      waveColor: color + '60',
      progressColor: color,
      cursorColor: color,
      cursorWidth: 2,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      height: 100,
      normalize: true,
      backend: 'WebAudio',
    })

    wavesurfer.load(url)

    wavesurfer.on('ready', () => {
      setDuration(wavesurfer.getDuration())
    })

    wavesurfer.on('audioprocess', () => {
      setCurrentTime(wavesurfer.getCurrentTime())
    })

    wavesurfer.on('play', () => setIsPlaying(true))
    wavesurfer.on('pause', () => setIsPlaying(false))
    wavesurfer.on('finish', () => setIsPlaying(false))

    wavesurferRef.current = wavesurfer

    return () => {
      wavesurfer.destroy()
    }
  }, [url, color])

  const togglePlay = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause()
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <motion.div
      className="glass rounded-2xl p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="font-semibold text-sm" style={{ color }}>
          {label}
        </span>
        {url && (
          <span className="text-xs text-white/40 font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        )}
      </div>

      {/* Waveform container */}
      <div className="waveform-container bg-void/50 rounded-xl p-3 min-h-[100px] relative">
        {isEmpty ? (
          <div className="flex items-center justify-center h-[100px] text-white/30">
            {isProcessing ? (
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  {[...Array(5)].map((_, i) => (
                    <motion.div
                      key={i}
                      className="w-1 bg-accent-cyan rounded-full"
                      animate={{
                        height: [20, 40, 20],
                      }}
                      transition={{
                        duration: 0.6,
                        repeat: Infinity,
                        delay: i * 0.1,
                      }}
                    />
                  ))}
                </div>
                <span className="ml-3 text-sm">Processing...</span>
              </div>
            ) : (
              <span className="text-sm">Enhanced audio will appear here</span>
            )}
          </div>
        ) : isAnalyzing ? (
          <div className="flex items-center justify-center h-[100px]">
            <div className="animate-shimmer w-full h-16 rounded-lg" />
          </div>
        ) : (
          <div ref={containerRef} />
        )}
      </div>

      {/* Controls */}
      {url && !isEmpty && (
        <div className="flex items-center gap-3 mt-4">
          <motion.button
            onClick={togglePlay}
            className="w-10 h-10 rounded-full flex items-center justify-center transition-colors"
            style={{ backgroundColor: color + '20' }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" style={{ color }} />
            ) : (
              <Play className="w-5 h-5 ml-0.5" style={{ color }} />
            )}
          </motion.button>
          
          <div className="flex-1 h-1 bg-elevated rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ 
                backgroundColor: color,
                width: `${(currentTime / duration) * 100}%`
              }}
            />
          </div>

          <Volume2 className="w-4 h-4 text-white/40" />
        </div>
      )}
    </motion.div>
  )
}

