import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Mic, Square, Loader2 } from 'lucide-react'

export default function Recorder({ onRecordingComplete }) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPreparing, setIsPreparing] = useState(false)
  const [duration, setDuration] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)
  
  const mediaRecorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const chunksRef = useRef([])
  const intervalRef = useRef(null)
  const animationRef = useRef(null)

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (audioContextRef.current) audioContextRef.current.close()
    }
  }, [])

  const startRecording = async () => {
    setIsPreparing(true)
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      })
      
      // Setup audio analysis
      audioContextRef.current = new AudioContext()
      analyserRef.current = audioContextRef.current.createAnalyser()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      source.connect(analyserRef.current)
      analyserRef.current.fftSize = 256
      
      // Start level monitoring
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
      const updateLevel = () => {
        analyserRef.current.getByteFrequencyData(dataArray)
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length
        setAudioLevel(average / 255)
        animationRef.current = requestAnimationFrame(updateLevel)
      }
      updateLevel()
      
      // Setup media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        chunksRef.current = []
        onRecordingComplete(blob)
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop())
        if (animationRef.current) cancelAnimationFrame(animationRef.current)
      }
      
      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start(100)
      
      setIsRecording(true)
      setDuration(0)
      
      // Duration counter
      intervalRef.current = setInterval(() => {
        setDuration(d => d + 1)
      }, 1000)
      
    } catch (error) {
      console.error('Recording error:', error)
      alert('Could not access microphone. Please allow microphone access.')
    } finally {
      setIsPreparing(false)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh]">
      {/* Main recording button */}
      <div className="relative mb-8">
        {/* Pulsing rings when recording */}
        {isRecording && (
          <>
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-accent-cyan"
              animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-accent-cyan"
              animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
              transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
            />
          </>
        )}
        
        {/* Level indicator ring */}
        <motion.div
          className="absolute inset-[-8px] rounded-full"
          style={{
            background: `conic-gradient(from 0deg, #00d9ff ${audioLevel * 360}deg, transparent ${audioLevel * 360}deg)`,
            opacity: isRecording ? 1 : 0
          }}
        />
        
        <motion.button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isPreparing}
          className="relative w-32 h-32 rounded-full flex items-center justify-center"
          style={{
            background: isRecording 
              ? 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
              : 'linear-gradient(135deg, #00d9ff 0%, #a855f7 100%)'
          }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isPreparing ? (
            <Loader2 className="w-12 h-12 text-void animate-spin" />
          ) : isRecording ? (
            <Square className="w-12 h-12 text-void" fill="currentColor" />
          ) : (
            <Mic className="w-12 h-12 text-void" />
          )}
        </motion.button>
      </div>

      {/* Duration display */}
      <motion.div
        className="text-4xl font-mono font-bold mb-4"
        animate={{ opacity: isRecording ? 1 : 0.5 }}
      >
        {formatDuration(duration)}
      </motion.div>

      {/* Status text */}
      <p className="text-white/60 mb-8">
        {isPreparing 
          ? 'Preparing microphone...'
          : isRecording 
            ? 'Recording... Click to stop'
            : 'Click to start recording'}
      </p>

      {/* Audio level meter */}
      {isRecording && (
        <motion.div
          className="w-64 h-2 bg-elevated rounded-full overflow-hidden"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <motion.div
            className="h-full rounded-full"
            style={{
              background: audioLevel > 0.8 
                ? '#ef4444' 
                : audioLevel > 0.5 
                  ? '#f97316' 
                  : '#22c55e',
              width: `${audioLevel * 100}%`
            }}
          />
        </motion.div>
      )}

      {/* Tips */}
      <div className="mt-12 glass rounded-xl p-6 max-w-md">
        <h3 className="font-semibold mb-3">Recording Tips</h3>
        <ul className="text-sm text-white/60 space-y-2">
          <li>• Use a quiet environment</li>
          <li>• Keep microphone 6-12 inches from mouth</li>
          <li>• Avoid background music or TV</li>
          <li>• Our AI will enhance the rest!</li>
        </ul>
      </div>
    </div>
  )
}

