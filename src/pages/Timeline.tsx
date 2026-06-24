import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Zap, Mic, Type, Eye, Database, Activity, Filter } from 'lucide-react';
import { DemoCase, SignalSource } from '../types';
import { getRiskBg, getRiskColor, getRiskLabel, getSourceColor } from '../lib/utils';

interface TimelinePageProps {
  selectedCase: DemoCase;
  onBack: () => void;
}

const sourceIcons: Record<SignalSource, React.ElementType> = {
  OCR: Type,
  Audio: Mic,
  Visual: Eye,
  Metadata: Database,
  Behavior: Activity,
};

const ALL_SOURCES: SignalSource[] = ['OCR', 'Audio', 'Visual', 'Metadata', 'Behavior'];

export default function TimelinePage({ selectedCase, onBack }: TimelinePageProps) {
  const [activeFilter, setActiveFilter] = useState<SignalSource | 'all'>('all');

  const filtered = activeFilter === 'all'
    ? selectedCase.timeline
    : selectedCase.timeline.filter((e) => e.source === activeFilter);

  const totalDurationSeconds = parseInt(selectedCase.duration.split(':')[0]) * 60 + parseInt(selectedCase.duration.split(':')[1]);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={onBack} className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.08] flex items-center justify-center hover:bg-white/[0.08] transition-colors">
          <ArrowLeft className="w-4 h-4 text-slate-400" />
        </button>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
            <Zap className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">Evidence Timeline</h2>
            <p className="text-xs text-slate-500">Хронология обнаруженных сигналов</p>
          </div>
        </div>
        <div className="ml-auto text-xs text-slate-500">
          {selectedCase.title} · {selectedCase.duration} · {selectedCase.platform}
        </div>
      </div>

      {/* Video Progress Bar */}
      <div className="card p-4">
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Временная шкала видео</div>
        <div className="relative h-10 bg-white/[0.04] rounded-lg overflow-hidden border border-white/[0.06]">
          {selectedCase.timeline.map((event) => {
            const pct = (event.timeSeconds / totalDurationSeconds) * 100;
            const color = event.severity === 'critical' ? '#ef4444'
              : event.severity === 'high' ? '#f97316'
              : event.severity === 'medium' ? '#eab308'
              : '#22c55e';
            return (
              <div
                key={event.id}
                className="absolute top-0 h-full w-0.5 group cursor-pointer"
                style={{ left: `${pct}%`, backgroundColor: color }}
                title={`${event.time} — ${event.signal}`}
              >
                <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full border-2 border-[#0d0d1a] opacity-90 group-hover:scale-150 transition-transform"
                  style={{ backgroundColor: color }} />
                <div className="absolute top-4 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] font-mono text-slate-500">
                  {event.time}
                </div>
              </div>
            );
          })}
          <div className="absolute bottom-1 right-2 text-[9px] font-mono text-slate-600">{selectedCase.duration}</div>
        </div>
        <div className="flex items-center gap-4 mt-3">
          {[
            { label: 'Критический', color: '#ef4444' },
            { label: 'Высокий', color: '#f97316' },
            { label: 'Средний', color: '#eab308' },
            { label: 'Низкий', color: '#22c55e' },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="text-[10px] text-slate-500">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Filters + Timeline */}
      <div className="grid grid-cols-4 gap-4">
        {/* Filters */}
        <div className="card p-4 space-y-2">
          <div className="flex items-center gap-2 mb-3">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Источники</span>
          </div>
          <button
            onClick={() => setActiveFilter('all')}
            className={`w-full text-left text-xs px-3 py-2 rounded-lg transition-all ${activeFilter === 'all' ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'}`}
          >
            Все сигналы ({selectedCase.timeline.length})
          </button>
          {ALL_SOURCES.map((source) => {
            const count = selectedCase.timeline.filter((e) => e.source === source).length;
            if (count === 0) return null;
            const Icon = sourceIcons[source];
            return (
              <button
                key={source}
                onClick={() => setActiveFilter(source)}
                className={`w-full flex items-center gap-2 text-xs px-3 py-2 rounded-lg transition-all ${activeFilter === source ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'}`}
              >
                <Icon className="w-3 h-3" />
                {source}
                <span className="ml-auto text-[10px] font-mono bg-white/[0.06] px-1.5 py-0.5 rounded">{count}</span>
              </button>
            );
          })}

          <div className="pt-3 border-t border-white/[0.06]">
            <div className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider mb-2">Статистика</div>
            <div className="space-y-1.5">
              {[
                { label: 'Критических', value: selectedCase.timeline.filter(e => e.severity === 'critical').length, color: 'text-red-400' },
                { label: 'Высоких', value: selectedCase.timeline.filter(e => e.severity === 'high').length, color: 'text-orange-400' },
                { label: 'Средних', value: selectedCase.timeline.filter(e => e.severity === 'medium').length, color: 'text-yellow-400' },
                { label: 'Низких', value: selectedCase.timeline.filter(e => e.severity === 'low').length, color: 'text-green-400' },
              ].map((s) => (
                <div key={s.label} className="flex items-center justify-between text-[10px]">
                  <span className="text-slate-500">{s.label}</span>
                  <span className={`font-mono font-bold ${s.color}`}>{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Events */}
        <div className="col-span-3 space-y-3">
          {filtered.length === 0 ? (
            <div className="card p-8 text-center text-slate-500 text-sm">
              Нет сигналов для выбранного фильтра
            </div>
          ) : (
            filtered.map((event, i) => {
              const Icon = sourceIcons[event.source];
              const sourceStyle = getSourceColor(event.source);
              const severityColor = event.severity === 'critical' ? '#ef4444'
                : event.severity === 'high' ? '#f97316'
                : event.severity === 'medium' ? '#eab308'
                : '#22c55e';

              return (
                <motion.div
                  key={event.id}
                  initial={{ opacity: 0, x: 16 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className="relative"
                >
                  {i < filtered.length - 1 && (
                    <div className="absolute left-[18px] top-10 w-0.5 h-full bg-gradient-to-b from-white/10 to-transparent -z-0" />
                  )}
                  <div className="flex items-start gap-4">
                    {/* Icon */}
                    <div
                      className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 border relative z-10"
                      style={{ backgroundColor: `${severityColor}15`, borderColor: `${severityColor}35` }}
                    >
                      <Icon className="w-4 h-4" style={{ color: severityColor }} />
                    </div>

                    {/* Content */}
                    <div className="flex-1 card p-3.5">
                      <div className="flex items-start gap-3 mb-2">
                        <span className="font-mono text-sm font-bold text-white bg-black/30 px-2 py-0.5 rounded border border-white/10 flex-shrink-0">
                          {event.time}
                        </span>
                        <span className={`badge border text-[10px] ${sourceStyle}`}>
                          <Icon className="w-2.5 h-2.5 mr-1" />
                          {event.source}
                        </span>
                        <span className={`badge border text-[10px] ml-auto ${getRiskBg(event.severity)}`}>
                          {getRiskLabel(event.severity)}
                        </span>
                      </div>
                      <p className="text-sm text-slate-200 leading-relaxed">{event.signal}</p>
                      <div className="flex items-center gap-2 mt-2.5">
                        <span className="text-[10px] text-slate-500">Уверенность:</span>
                        <div className="flex-1 h-1 bg-white/[0.06] rounded-full overflow-hidden max-w-24">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${event.confidence}%`, backgroundColor: severityColor }}
                          />
                        </div>
                        <span className="text-[10px] font-mono font-bold" style={{ color: severityColor }}>
                          {event.confidence}%
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
