import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import AudioUploader from './components/AudioUploader'
import Waveform from './components/Waveform'
import ProcessingControls from './components/ProcessingControls'
import BeforeAfter from './components/BeforeAfter'
import ProblemsDisplay from './components/ProblemsDisplay'
import Recorder from './components/Recorder'
import { 
  Sparkles, 
  Settings2, 
  Download, 
  RefreshCw,
  Mic,
  Upload,
  Zap
} from 'lucide-react'

const API_URL = import.meta.env.PROD ? '' : 'http://localhost:8000'

function App() {
  const [mode, setMode] = useState('upload') // upload, record
  const [audioFile, setAudioFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [processedUrl, setProcessedUrl] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [processResult, setProcessResult] = useState(null)
  const [selectedPreset, setSelectedPreset] = useState('auto')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [settings, setSettings] = useState({
    auto_mode: true,
    noise_reduction: 0.5,
    de_reverb: 0.3,
    de_esser: 0.4,
    compression_ratio: 3.0,
    target_lufs: -16.0,
  })

  // Handle file selection
  const handleFileSelect = useCallback(async (file) => {
    setAudioFile(file)
    setAudioUrl(URL.createObjectURL(file))
    setProcessedUrl(null)
    setProcessResult(null)
    setAnalysis(null)

    // Auto-analyze
    setIsAnalyzing(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })
      
      if (response.ok) {
        const data = await response.json()
        setAnalysis(data)
      }
    } catch (error) {
      console.error('Analysis error:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }, [])

  // Handle recording complete
  const handleRecordingComplete = useCallback((blob) => {
    const file = new File([blob], 'recording.wav', { type: 'audio/wav' })
    handleFileSelect(file)
    setMode('upload')
  }, [handleFileSelect])

  // Process audio
  const handleProcess = async () => {
    if (!audioFile) return

    setIsProcessing(true)
    try {
      const formData = new FormData()
      formData.append('file', audioFile)

      const params = new URLSearchParams({
        auto_mode: settings.auto_mode,
        preset: selectedPreset,
        noise_reduction: settings.noise_reduction,
        de_reverb: settings.de_reverb,
        de_esser: settings.de_esser,
        compression_ratio: settings.compression_ratio,
        target_lufs: settings.target_lufs,
        output_format: 'wav'
      })

      const response = await fetch(`${API_URL}/process?${params}`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        setProcessedUrl(url)
        
        setProcessResult({
          processingTime: response.headers.get('X-Processing-Time-Ms'),
          problemsDetected: response.headers.get('X-Problems-Detected'),
          processorsApplied: response.headers.get('X-Processors-Applied'),
        })
      }
    } catch (error) {
      console.error('Processing error:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  // Download processed file
  const handleDownload = () => {
    if (!processedUrl) return
    
    const a = document.createElement('a')
    a.href = processedUrl
    a.download = `${audioFile.name.replace(/\.[^/.]+$/, '')}_enhanced.wav`
    a.click()
  }

  // Reset
  const handleReset = () => {
    setAudioFile(null)
    setAudioUrl(null)
    setProcessedUrl(null)
    setAnalysis(null)
    setProcessResult(null)
  }

  const presets = [
    { id: 'auto', name: 'Auto Magic', icon: Sparkles, description: 'AI does everything' },
    { id: 'podcast', name: 'Podcast', icon: Mic, description: 'Perfect for episodes' },
    { id: 'interview', name: 'Interview', icon: null, description: 'Clear dialogue' },
    { id: 'audiobook', name: 'Audiobook', icon: null, description: 'Warm narration' },
    { id: 'voiceover', name: 'Voiceover', icon: null, description: 'Broadcast quality' },
  ]

  return (
    <div className="min-h-screen bg-void bg-grid">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent-cyan/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent-purple/10 rounded-full blur-[120px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <motion.div 
            className="flex items-center gap-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-cyan to-accent-purple flex items-center justify-center">
              <Zap className="w-5 h-5 text-void" />
            </div>
            <div>
              <h1 className="font-display font-bold text-xl text-gradient">SoundForge AI</h1>
              <p className="text-xs text-white/40">AI Audio Enhancement</p>
            </div>
          </motion.div>

          <motion.div 
            className="flex items-center gap-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <button
              onClick={() => setMode('upload')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                mode === 'upload' 
                  ? 'bg-accent-cyan/20 text-accent-cyan' 
                  : 'text-white/60 hover:text-white'
              }`}
            >
              <Upload className="w-4 h-4 inline mr-2" />
              Upload
            </button>
            <button
              onClick={() => setMode('record')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                mode === 'record' 
                  ? 'bg-accent-cyan/20 text-accent-cyan' 
                  : 'text-white/60 hover:text-white'
              }`}
            >
              <Mic className="w-4 h-4 inline mr-2" />
              Record
            </button>
          </motion.div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          {mode === 'record' && !audioFile ? (
            <motion.div
              key="recorder"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <Recorder onRecordingComplete={handleRecordingComplete} />
            </motion.div>
          ) : !audioFile ? (
            <motion.div
              key="uploader"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center justify-center min-h-[60vh]"
            >
              <AudioUploader onFileSelect={handleFileSelect} />
            </motion.div>
          ) : (
            <motion.div
              key="editor"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* File info & actions */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-elevated flex items-center justify-center">
                    <Mic className="w-6 h-6 text-accent-cyan" />
                  </div>
                  <div>
                    <h2 className="font-display font-semibold text-lg">{audioFile.name}</h2>
                    <p className="text-sm text-white/40">
                      {analysis?.audio_info?.duration_seconds 
                        ? `${analysis.audio_info.duration_seconds.toFixed(1)}s • ${analysis.audio_info.sample_rate}Hz`
                        : 'Analyzing...'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleReset}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    New File
                  </button>
                  {processedUrl && (
                    <button
                      onClick={handleDownload}
                      className="btn-primary flex items-center gap-2"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  )}
                </div>
              </div>

              {/* Waveforms */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Waveform 
                  url={audioUrl} 
                  label="Original" 
                  color="#6b7280"
                  isAnalyzing={isAnalyzing}
                />
                <Waveform 
                  url={processedUrl} 
                  label="Enhanced" 
                  color="#00d9ff"
                  isEmpty={!processedUrl}
                  isProcessing={isProcessing}
                />
              </div>

              {/* Problems Display */}
              {analysis && (
                <ProblemsDisplay problems={analysis.problems} />
              )}

              {/* Presets */}
              <div className="glass rounded-2xl p-6">
                <h3 className="font-display font-semibold text-lg mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-accent-cyan" />
                  Processing Mode
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {presets.map((preset) => (
                    <motion.button
                      key={preset.id}
                      onClick={() => {
                        setSelectedPreset(preset.id)
                        setSettings(prev => ({ ...prev, auto_mode: preset.id === 'auto' }))
                      }}
                      className={`preset-card text-left ${selectedPreset === preset.id ? 'active' : ''}`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        {preset.icon && <preset.icon className="w-4 h-4 text-accent-cyan" />}
                        <span className="font-semibold">{preset.name}</span>
                      </div>
                      <p className="text-xs text-white/40">{preset.description}</p>
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Advanced Settings */}
              <div className="glass rounded-2xl overflow-hidden">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                  <span className="font-display font-semibold flex items-center gap-2">
                    <Settings2 className="w-5 h-5 text-accent-purple" />
                    Manual Controls
                  </span>
                  <motion.span
                    animate={{ rotate: showAdvanced ? 180 : 0 }}
                    className="text-white/40"
                  >
                    ▼
                  </motion.span>
                </button>
                
                <AnimatePresence>
                  {showAdvanced && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <ProcessingControls 
                        settings={settings}
                        onChange={setSettings}
                        disabled={settings.auto_mode && selectedPreset === 'auto'}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Process Button */}
              <motion.button
                onClick={handleProcess}
                disabled={isProcessing}
                className="w-full py-5 rounded-2xl font-display font-bold text-xl relative overflow-hidden group"
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                style={{
                  background: 'linear-gradient(135deg, #00d9ff 0%, #a855f7 100%)',
                }}
              >
                <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                <span className="relative text-void flex items-center justify-center gap-3">
                  {isProcessing ? (
                    <>
                      <RefreshCw className="w-6 h-6 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-6 h-6" />
                      Enhance Audio
                    </>
                  )}
                </span>
              </motion.button>

              {/* Results */}
              {processResult && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass rounded-2xl p-6"
                >
                  <h3 className="font-display font-semibold text-lg mb-4 text-accent-green flex items-center gap-2">
                    ✓ Enhancement Complete
                  </h3>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-accent-cyan">
                        {processResult.processingTime}ms
                      </div>
                      <div className="text-sm text-white/40">Processing Time</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-accent-purple">
                        {processResult.problemsDetected}
                      </div>
                      <div className="text-sm text-white/40">Issues Fixed</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-accent-green">
                        {processResult.processorsApplied}
                      </div>
                      <div className="text-sm text-white/40">Processors Used</div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Before/After Compare */}
              {processedUrl && (
                <BeforeAfter 
                  originalUrl={audioUrl} 
                  processedUrl={processedUrl} 
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 mt-auto">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between text-sm text-white/40">
          <span>SoundForge AI v1.0</span>
          <span>12 Professional Audio Engines</span>
        </div>
      </footer>
    </div>
  )
}

export default App

