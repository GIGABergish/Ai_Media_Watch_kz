import { Shield, Scale, Cpu, Database, Globe, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { motion } from 'framer-motion';

const formulaWeights = [
  { label: 'Текст и речь', weight: 35, color: '#8b5cf6', description: 'Анализ транскрипции аудио, ключевых фраз и лингвистических паттернов' },
  { label: 'Визуальные признаки', weight: 25, color: '#3b82f6', description: 'Компьютерное зрение: игровые интерфейсы, деньги, QR-коды, графики' },
  { label: 'Метаданные и ссылки', weight: 15, color: '#06b6d4', description: 'Хэштеги, описания, внешние ссылки, Telegram/WhatsApp контакты' },
  { label: 'Поведенческие паттерны', weight: 15, color: '#f97316', description: 'Давление срочности, реферальные призывы, тактики вовлечения' },
  { label: 'Похожесть на базу', weight: 10, color: '#ec4899', description: 'Сравнение с известными мошенническими паттернами в базе знаний' },
];

const ethicsPoints = [
  {
    icon: Scale,
    title: 'Нет юридических решений',
    text: 'Система не выносит автоматических юридических заключений о виновности. Все результаты требуют верификации специалистом.',
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
  },
  {
    icon: AlertTriangle,
    title: 'Риск-ориентированная оценка',
    text: 'Высокий Risk Score означает наличие признаков для дополнительной проверки, а не доказанное нарушение.',
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/20',
  },
  {
    icon: Shield,
    title: 'Защита персональных данных',
    text: 'Система не хранит персональные данные пользователей, не использует реальные аккаунты и не создаёт юридически значимых записей.',
    color: 'text-green-400',
    bg: 'bg-green-500/10',
    border: 'border-green-500/20',
  },
  {
    icon: Info,
    title: 'Ложные срабатывания',
    text: 'Образовательный контент о финансах может получить промежуточные оценки риска. Контекст всегда важнее автоматического вывода.',
    color: 'text-violet-400',
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/20',
  },
];

const systemInfo = [
  { label: 'Версия системы', value: 'Sentinel Media AI v1.0', icon: Cpu },
  { label: 'Модель детекции', value: 'Multimodal Risk v2.3', icon: Database },
  { label: 'Языки анализа', value: 'RU, KZ, EN', icon: Globe },
  { label: 'Статус системы', value: 'Активна', icon: CheckCircle },
];

export default function SettingsPage() {
  return (
    <div className="max-w-4xl space-y-6">
      {/* Main Ethics Block */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="card p-6 border-violet-500/20 bg-violet-500/[0.03]"
      >
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-xl bg-violet-500/20 border border-violet-500/30 flex items-center justify-center">
            <Shield className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">Этические принципы AI Media Watch</h2>
            <p className="text-xs text-slate-500">Explainable AI · Responsible Detection</p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.08] mb-5">
          <p className="text-sm text-slate-300 leading-relaxed">
            <span className="text-violet-300 font-semibold">AI Media Watch</span> не выносит автоматических юридических решений. Система помогает приоритизировать контент для ручной проверки специалистами, объясняя найденные признаки риска на основе мультимодального анализа.
          </p>
          <p className="text-xs text-slate-500 mt-2 leading-relaxed">
            Формулировка результата: «Контент содержит признаки, требующие дополнительной проверки» — а не автоматическое обвинение.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {ethicsPoints.map((point) => {
            const Icon = point.icon;
            return (
              <div key={point.title} className={`p-4 rounded-xl border ${point.bg} ${point.border}`}>
                <div className="flex items-center gap-2 mb-2">
                  <Icon className={`w-4 h-4 ${point.color}`} />
                  <span className={`text-xs font-semibold ${point.color}`}>{point.title}</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">{point.text}</p>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* Risk Formula */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card p-6"
      >
        <h3 className="text-sm font-bold text-white mb-1">Формула Risk Score</h3>
        <p className="text-xs text-slate-500 mb-5">Прозрачная многокомпонентная оценка риска</p>

        <div className="space-y-4">
          {formulaWeights.map((item, i) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.15 + i * 0.07 }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-sm font-medium text-white">{item.label}</span>
                </div>
                <span className="text-sm font-bold font-mono" style={{ color: item.color }}>{item.weight}%</span>
              </div>
              <div className="h-2 bg-white/[0.06] rounded-full overflow-hidden mb-1.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${item.weight * 2.86}%` }}
                  transition={{ duration: 0.8, delay: 0.2 + i * 0.07, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: item.color, opacity: 0.8 }}
                />
              </div>
              <p className="text-[10px] text-slate-500">{item.description}</p>
            </motion.div>
          ))}
        </div>

        <div className="mt-5 p-4 rounded-xl bg-yellow-500/[0.06] border border-yellow-500/15">
          <p className="text-[11px] text-yellow-300/80 leading-relaxed">
            <span className="font-semibold">Важно:</span> Система использует риск-ориентированную оценку. Результат не является юридическим заключением и требует проверки специалистом. Итоговый балл отражает вероятность наличия признаков, а не факт нарушения.
          </p>
        </div>
      </motion.div>

      {/* System Info */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card p-6"
      >
        <h3 className="text-sm font-bold text-white mb-4">Информация о системе</h3>
        <div className="grid grid-cols-2 gap-3">
          {systemInfo.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.label} className="flex items-center gap-3 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
                <Icon className="w-4 h-4 text-slate-400 flex-shrink-0" />
                <div>
                  <div className="text-[10px] text-slate-500">{item.label}</div>
                  <div className="text-xs font-medium text-white">{item.value}</div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 p-4 rounded-xl bg-white/[0.02] border border-white/[0.06]">
          <div className="text-xs font-semibold text-slate-400 mb-2">Поддерживаемые платформы</div>
          <div className="flex flex-wrap gap-2">
            {['Instagram', 'TikTok', 'YouTube', 'Telegram', 'VK', 'Facebook'].map((p) => (
              <span key={p} className="text-[10px] px-2 py-1 rounded bg-white/[0.05] border border-white/[0.08] text-slate-300">
                {p}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-4 p-4 rounded-xl bg-white/[0.02] border border-white/[0.06]">
          <div className="text-xs font-semibold text-slate-400 mb-2">Будущие интеграции (архитектура готова)</div>
          <div className="grid grid-cols-2 gap-2">
            {[
              'YouTube Data API v3',
              'Meta Content Library API',
              'VK API',
              'Whisper ASR (аудио)',
              'TesseractJS (OCR)',
              'CLIP (визуальный анализ)',
            ].map((api) => (
              <div key={api} className="flex items-center gap-2 text-[10px] text-slate-500">
                <div className="w-1 h-1 rounded-full bg-slate-600" />
                {api}
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
