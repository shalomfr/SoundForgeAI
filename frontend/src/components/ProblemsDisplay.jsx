import { motion } from 'framer-motion'
import { 
  AlertTriangle, 
  Volume2, 
  Waves, 
  Zap, 
  Wind,
  CircleDot
} from 'lucide-react'

const problemInfo = {
  noise_level: {
    label: 'Background Noise',
    icon: Volume2,
    description: 'Ambient noise detected'
  },
  clipping: {
    label: 'Clipping',
    icon: Zap,
    description: 'Audio distortion from peaks'
  },
  sibilance: {
    label: 'Sibilance',
    icon: Wind,
    description: 'Harsh S sounds'
  },
  reverb: {
    label: 'Reverb',
    icon: Waves,
    description: 'Room echo detected'
  },
  muddiness: {
    label: 'Muddiness',
    icon: CircleDot,
    description: 'Low-mid buildup'
  },
  harshness: {
    label: 'Harshness',
    icon: AlertTriangle,
    description: 'Fatiguing frequencies'
  },
  breath_sounds: {
    label: 'Breath Sounds',
    icon: Wind,
    description: 'Audible breathing'
  },
  dynamic_range: {
    label: 'Dynamic Range',
    icon: Waves,
    description: 'Volume inconsistency'
  }
}

const getSeverityClass = (severity) => {
  switch (severity) {
    case 'none': return 'problem-none'
    case 'low': return 'problem-low'
    case 'medium': return 'problem-medium'
    case 'high': return 'problem-high'
    default: return 'problem-none'
  }
}

const getSeverityColor = (severity) => {
  switch (severity) {
    case 'none': return '#22c55e'
    case 'low': return '#84cc16'
    case 'medium': return '#f97316'
    case 'high': return '#ef4444'
    default: return '#22c55e'
  }
}

export default function ProblemsDisplay({ problems }) {
  if (!problems) return null

  const significantProblems = Object.entries(problems)
    .filter(([_, data]) => data.severity !== 'none')
    .sort((a, b) => {
      const order = { high: 0, medium: 1, low: 2 }
      return order[a[1].severity] - order[b[1].severity]
    })

  const allGood = significantProblems.length === 0

  return (
    <motion.div
      className="glass rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="font-display font-semibold text-lg mb-4 flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-accent-orange" />
        Audio Analysis
      </h3>

      {allGood ? (
        <div className="text-center py-6">
          <div className="w-16 h-16 rounded-full bg-accent-green/20 flex items-center justify-center mx-auto mb-3">
            <span className="text-3xl">✓</span>
          </div>
          <p className="text-accent-green font-semibold">Audio quality looks great!</p>
          <p className="text-white/40 text-sm mt-1">No significant issues detected</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(problems).map(([key, data], index) => {
            const info = problemInfo[key]
            if (!info) return null
            
            const Icon = info.icon
            const color = getSeverityColor(data.severity)
            
            return (
              <motion.div
                key={key}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="p-3 rounded-xl bg-void/50 relative overflow-hidden"
              >
                {/* Severity indicator */}
                <div 
                  className="absolute top-0 left-0 w-full h-1"
                  style={{ backgroundColor: color }}
                />
                
                <div className="flex items-start gap-2 mt-1">
                  <div 
                    className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                    style={{ backgroundColor: color + '20' }}
                  >
                    <Icon className="w-4 h-4" style={{ color }} />
                  </div>
                  <div className="min-w-0">
                    <div className="font-medium text-sm truncate">{info.label}</div>
                    <div className="text-xs text-white/40 capitalize">{data.severity}</div>
                  </div>
                </div>

                {/* Value bar */}
                <div className="mt-2 h-1.5 bg-elevated rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${data.value * 100}%` }}
                    transition={{ delay: index * 0.05 + 0.2, duration: 0.5 }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      )}
    </motion.div>
  )
}

