import { motion } from 'framer-motion'

const Slider = ({ label, value, onChange, min = 0, max = 1, step = 0.1, disabled, unit = '' }) => (
  <div className={`space-y-2 ${disabled ? 'opacity-40' : ''}`}>
    <div className="flex justify-between text-sm">
      <span className="text-white/70">{label}</span>
      <span className="font-mono text-accent-cyan">{value.toFixed(1)}{unit}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      disabled={disabled}
      className="w-full"
    />
  </div>
)

export default function ProcessingControls({ settings, onChange, disabled }) {
  const updateSetting = (key, value) => {
    onChange({ ...settings, [key]: value })
  }

  return (
    <motion.div
      className="p-6 border-t border-white/5"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {disabled && (
        <div className="mb-4 p-3 rounded-lg bg-accent-cyan/10 text-accent-cyan text-sm">
          Switch to a manual preset to customize settings
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Noise Reduction */}
        <Slider
          label="Noise Reduction"
          value={settings.noise_reduction}
          onChange={(v) => updateSetting('noise_reduction', v)}
          disabled={disabled}
        />

        {/* De-Reverb */}
        <Slider
          label="De-Reverb"
          value={settings.de_reverb}
          onChange={(v) => updateSetting('de_reverb', v)}
          disabled={disabled}
        />

        {/* De-Esser */}
        <Slider
          label="De-Esser"
          value={settings.de_esser}
          onChange={(v) => updateSetting('de_esser', v)}
          disabled={disabled}
        />

        {/* Compression Ratio */}
        <Slider
          label="Compression Ratio"
          value={settings.compression_ratio}
          onChange={(v) => updateSetting('compression_ratio', v)}
          min={1}
          max={8}
          step={0.5}
          disabled={disabled}
          unit=":1"
        />

        {/* Target LUFS */}
        <div className={`space-y-2 ${disabled ? 'opacity-40' : ''}`}>
          <div className="flex justify-between text-sm">
            <span className="text-white/70">Target LUFS</span>
            <span className="font-mono text-accent-cyan">{settings.target_lufs} LUFS</span>
          </div>
          <input
            type="range"
            min={-24}
            max={-6}
            step={1}
            value={settings.target_lufs}
            onChange={(e) => updateSetting('target_lufs', parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-white/30">
            <span>Quieter</span>
            <span>Louder</span>
          </div>
        </div>

        {/* Info panel */}
        <div className="p-4 rounded-xl bg-void/50 space-y-2">
          <h4 className="font-semibold text-sm text-white/70">LUFS Guidelines</h4>
          <ul className="text-xs text-white/40 space-y-1">
            <li>• -16 LUFS: Podcast standard</li>
            <li>• -14 LUFS: Spotify, YouTube</li>
            <li>• -18 LUFS: Audiobooks</li>
            <li>• -23 LUFS: Broadcast TV</li>
          </ul>
        </div>
      </div>
    </motion.div>
  )
}

