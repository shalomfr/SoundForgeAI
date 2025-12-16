import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Upload, Music, FileAudio } from 'lucide-react'

export default function AudioUploader({ onFileSelect }) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDragIn = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragOut = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer?.files
    if (files && files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('audio/')) {
        onFileSelect(file)
      }
    }
  }, [onFileSelect])

  const handleFileInput = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="w-full max-w-2xl"
    >
      {/* Main upload area */}
      <label
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`drop-zone flex flex-col items-center justify-center p-12 cursor-pointer ${
          isDragging ? 'drag-over' : ''
        }`}
      >
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileInput}
          className="hidden"
        />
        
        {/* Icon */}
        <motion.div
          className="relative mb-6"
          animate={{
            scale: isDragging ? 1.1 : 1,
          }}
        >
          <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-accent-cyan/20 to-accent-purple/20 flex items-center justify-center">
            <Upload className="w-10 h-10 text-accent-cyan" />
          </div>
          {isDragging && (
            <motion.div
              className="absolute inset-0 rounded-2xl border-2 border-accent-cyan"
              initial={{ scale: 1 }}
              animate={{ scale: 1.2, opacity: 0 }}
              transition={{ repeat: Infinity, duration: 1 }}
            />
          )}
        </motion.div>

        {/* Text */}
        <h2 className="font-display font-bold text-2xl mb-2">
          {isDragging ? 'Drop it here!' : 'Drop your audio file'}
        </h2>
        <p className="text-white/50 mb-6">
          or click to browse • WAV, MP3, FLAC, M4A
        </p>

        {/* Browse button */}
        <div className="btn-primary">
          Browse Files
        </div>
      </label>

      {/* Features list */}
      <div className="mt-8 grid grid-cols-3 gap-4">
        {[
          { icon: Music, label: 'Any format', desc: 'WAV, MP3, FLAC, OGG' },
          { icon: FileAudio, label: 'Up to 1GB', desc: 'Large files supported' },
          { icon: Upload, label: 'Instant', desc: 'Fast cloud processing' },
        ].map((item, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + i * 0.1 }}
            className="glass rounded-xl p-4 text-center"
          >
            <item.icon className="w-6 h-6 text-accent-cyan mx-auto mb-2" />
            <div className="font-semibold text-sm">{item.label}</div>
            <div className="text-xs text-white/40">{item.desc}</div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}

