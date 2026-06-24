import { useState, useCallback, useRef } from 'react';
import {
  Upload, Play, ChevronRight, Mic, Type, Eye, Hash, Link2, Users,
  CheckCircle, AlertTriangle, Loader2, Shield, ExternalLink, Clock,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { demoCases } from '../data/cases';
import { DemoCase, EvidenceType } from '../types';
import { getRiskBg, getRiskColor, getRiskLabel, getRiskScoreGradient, getSourceColor } from '../lib/utils';
import { analyzeFile, AnalysisMeta } from '../lib/api';
import { USE_BACKEND } from '../config';

interface AnalysisProps {
  selectedCase: DemoCase;
  onSelectCase: (c: DemoCase) => void;
  onViewScamDNA: () => void;
  onViewTimeline: () => void;
  onViewConnections: () => void;
}

const ANALYSIS_STEPS = [
  { label: 'Извлечение аудио', icon: Mic, duration: 700 },
  { label: 'Анализ речи', icon: Mic, duration: 900 },
  { label: 'OCR текста', icon: Type, duration: 800 },
  { label: 'Анализ визуальных маркеров', icon: Eye, duration: 1000 },
  { label: 'Расчёт Risk Score', icon: Shield, duration: 700 },
  { label: 'Формирование объяснения', icon: CheckCircle, duration: 600 },
];

const evidenceIcons: Record<EvidenceType, React.ElementType> = {
  audio: Mic,
  ocr: Type,
  visual: Eye,
  metadata: Hash,
  links: Link2,
  engagement: Users,
};

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono text-slate-400 w-8 text-right">{value}%</span>
    </div>
  );
}

export default function Analysis({ selectedCase, onSelectCase, onViewScamDNA, onViewTimeline, onViewConnections }: AnalysisProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(-1);
  const [showResults, setShowResults] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<AnalysisMeta | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Runs the step animation (used by demo mocks and as the visual layer for real analysis).
  const runStepAnimation = useCallback(async () => {
    for (let i = 0; i < ANALYSIS_STEPS.length; i++) {
      setAnalysisStep(i);
      await new Promise((r) => setTimeout(r, ANALYSIS_STEPS[i].duration));
    }
  }, []);

  const runAnalysis = useCallback(async (caseToAnalyze?: DemoCase) => {
    if (caseToAnalyze) onSelectCase(caseToAnalyze);
    setError(null);
    setMeta(null);
    setIsAnalyzing(true);
    setShowResults(false);
    setAnalysisStep(0);

    await runStepAnimation();

    setIsAnalyzing(false);
    setShowResults(true);
    setAnalysisStep(-1);
  }, [onSelectCase, runStepAnimation]);

  // Sends a real uploaded video file to the backend while the step animation plays.
  const analyzeRealFile = useCallback(async (file: File) => {
    setError(null);
    setMeta(null);
    setIsAnalyzing(true);
    setShowResults(false);
    setAnalysisStep(0);

    const [result] = await Promise.allSettled([
      analyzeFile(file),
      runStepAnimation(),
    ] as const);

    if (result.status === 'fulfilled') {
      onSelectCase(result.value.case);
      setMeta(result.value.meta);
    } else {
      onSelectCase(demoCases[0]);
      const reason = result.reason instanceof Error ? result.reason.message : String(result.reason);
      setError(`Бэкенд недоступен — показан demo-кейс. ${reason}`);
    }

    setIsAnalyzing(false);
    setShowResults(true);
    setAnalysisStep(-1);
  }, [onSelectCase, runStepAnimation]);

  const handleFile = useCallback((file: File | undefined) => {
    if (!file || !file.type.startsWith('video/')) return;
    if (USE_BACKEND) {
      analyzeRealFile(file);
    } else {
      runAnalysis(demoCases[0]);
    }
  }, [analyzeRealFile, runAnalysis]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const riskGradient = getRiskScoreGradient(selectedCase.riskScore);
  const scoreColor = selectedCase.riskLevel === 'critical' ? '#ef4444'
    : selectedCase.riskLevel === 'high' ? '#f97316'
    : selectedCase.riskLevel === 'medium' ? '#eab308'
    : '#22c55e';

  return (
    <div className="grid grid-cols-5 gap-4 h-full">
      {/* Left Panel */}
      <div className="col-span-2 space-y-4">
        {/* Upload Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition-all duration-300
            ${isDragging
              ? 'border-violet-500 bg-violet-500/10'
              : 'border-white/[0.1] bg-white/[0.02] hover:border-violet-500/50 hover:bg-violet-500/5'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = ''; }}
          />
          <div className="w-10 h-10 rounded-xl bg-violet-500/15 border border-violet-500/25 flex items-center justify-center mx-auto mb-3">
            <Upload className="w-5 h-5 text-violet-400" />
          </div>
          <div className="text-sm font-medium text-white mb-1">Загрузить видео</div>
          <div className="text-xs text-slate-500">MP4, MOV, AVI · до 500 МБ</div>
          <div className="text-xs text-slate-600 mt-1">или перетащите файл сюда</div>
        </div>

        {/* Backend unavailable notice */}
        {error && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-red-500/[0.06] border border-red-500/20">
            <AlertTriangle className="w-3.5 h-3.5 text-red-400 mt-0.5 flex-shrink-0" />
            <p className="text-[11px] text-red-300/90 leading-relaxed break-words">{error}</p>
          </div>
        )}

        {/* Demo Cases */}
        <div className="card p-4">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Demo кейсы</div>
          <div className="space-y-2">
            {demoCases.map((c) => (
              <button
                key={c.id}
                onClick={() => runAnalysis(c)}
                className={`w-full flex items-center gap-3 p-2.5 rounded-lg border transition-all duration-150 text-left group
                  ${selectedCase.id === c.id
                    ? 'border-violet-500/40 bg-violet-500/10'
                    : 'border-white/[0.05] bg-white/[0.02] hover:border-white/[0.1] hover:bg-white/[0.04]'
                  }`}
              >
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-bold font-mono ${getRiskColor(c.riskLevel)} bg-black/30 border border-white/10`}>
                  {c.riskScore}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-white truncate leading-tight">{c.title}</div>
                  <div className="text-[10px] text-slate-500">{c.platform} · {c.duration}</div>
                </div>
                <ChevronRight className="w-3 h-3 text-slate-600 group-hover:text-slate-400 transition-colors flex-shrink-0" />
              </button>
            ))}
          </div>
        </div>

        {/* Quick Nav */}
        {showResults && (
          <div className="card p-4 space-y-2">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Модули анализа</div>
            {[
              { label: 'Scam DNA', desc: 'Цифровой отпечаток риска', action: onViewScamDNA, color: 'text-violet-400' },
              { label: 'Evidence Timeline', desc: 'Хронология доказательств', action: onViewTimeline, color: 'text-cyan-400' },
              { label: 'Карта связей', desc: `Кластер из ${selectedCase.connections.clusterSize} публикаций`, action: onViewConnections, color: 'text-blue-400' },
            ].map((item) => (
              <button
                key={item.label}
                onClick={item.action}
                className="w-full flex items-center gap-3 p-2.5 rounded-lg border border-white/[0.05] bg-white/[0.02] hover:border-white/[0.1] hover:bg-white/[0.04] transition-all text-left group"
              >
                <ExternalLink className={`w-3.5 h-3.5 ${item.color} flex-shrink-0`} />
                <div className="flex-1 min-w-0">
                  <div className={`text-xs font-medium ${item.color}`}>{item.label}</div>
                  <div className="text-[10px] text-slate-500">{item.desc}</div>
                </div>
                <ChevronRight className="w-3 h-3 text-slate-600 group-hover:text-slate-400 transition-colors" />
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Right Panel */}
      <div className="col-span-3 space-y-4">
        {/* Analysis Progress */}
        <AnimatePresence>
          {isAnalyzing && (
            <motion.div
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              className="card p-6"
            >
              <div className="flex items-center gap-3 mb-5">
                <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center">
                  <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-white">Анализ видео...</div>
                  <div className="text-xs text-slate-500">Multimodal AI Pipeline</div>
                </div>
              </div>
              <div className="space-y-3">
                {ANALYSIS_STEPS.map((step, i) => {
                  const StepIcon = step.icon;
                  const isDone = i < analysisStep;
                  const isCurrent = i === analysisStep;
                  return (
                    <div key={step.label} className={`flex items-center gap-3 transition-all duration-300 ${i > analysisStep ? 'opacity-30' : 'opacity-100'}`}>
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 transition-all ${isDone ? 'bg-green-500/20 border border-green-500/40' : isCurrent ? 'bg-violet-500/20 border border-violet-500/40' : 'bg-white/[0.05] border border-white/[0.1]'}`}>
                        {isDone
                          ? <CheckCircle className="w-3 h-3 text-green-400" />
                          : isCurrent
                          ? <Loader2 className="w-3 h-3 text-violet-400 animate-spin" />
                          : <StepIcon className="w-3 h-3 text-slate-600" />
                        }
                      </div>
                      <span className={`text-xs font-medium transition-colors ${isDone ? 'text-green-400' : isCurrent ? 'text-white' : 'text-slate-600'}`}>
                        {step.label}
                      </span>
                      {isCurrent && (
                        <div className="flex-1 h-1 bg-white/[0.06] rounded-full overflow-hidden ml-2">
                          <motion.div
                            initial={{ width: '0%' }}
                            animate={{ width: '100%' }}
                            transition={{ duration: step.duration / 1000, ease: 'linear' }}
                            className="h-full bg-gradient-to-r from-violet-500 to-cyan-400 rounded-full"
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {showResults && !isAnalyzing && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Engine Meta */}
              {meta && (
                <div className="card p-3">
                  <div className="flex flex-wrap items-center gap-2 text-[10px]">
                    <span className="badge border border-violet-500/30 bg-violet-500/10 text-violet-300 font-mono">
                      engine: {meta.engineMode}
                    </span>
                    <span className="flex items-center gap-1 text-slate-400 font-mono">
                      <Clock className="w-2.5 h-2.5" />
                      {meta.elapsedMs} мс
                    </span>
                    {meta.lanesRun.map((lane) => (
                      <span key={lane} className="px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/[0.08] text-slate-400 font-mono">
                        {lane}
                      </span>
                    ))}
                    {meta.degraded.map((d) => (
                      <span key={d} className="px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/25 text-orange-300 font-mono">
                        degraded: {d}
                      </span>
                    ))}
                  </div>
                  {meta.notes.length > 0 && (
                    <p className="text-[10px] text-slate-500 leading-relaxed mt-2">{meta.notes.join(' · ')}</p>
                  )}
                </div>
              )}

              {/* Risk Score Hero */}
              <div className={`card p-5 border ${selectedCase.riskLevel === 'critical' ? 'border-red-500/25 glow-red' : selectedCase.riskLevel === 'high' ? 'border-orange-500/20' : 'border-white/[0.06]'}`}>
                <div className="flex items-start gap-5">
                  {/* Score Ring */}
                  <div className="relative w-24 h-24 flex-shrink-0">
                    <svg className="w-24 h-24 -rotate-90" viewBox="0 0 96 96">
                      <circle cx="48" cy="48" r="40" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
                      <circle
                        cx="48" cy="48" r="40"
                        fill="none"
                        stroke={scoreColor}
                        strokeWidth="8"
                        strokeLinecap="round"
                        strokeDasharray={`${2 * Math.PI * 40}`}
                        strokeDashoffset={`${2 * Math.PI * 40 * (1 - selectedCase.riskScore / 100)}`}
                        style={{ filter: `drop-shadow(0 0 6px ${scoreColor}80)` }}
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span className={`text-2xl font-bold font-mono ${getRiskColor(selectedCase.riskLevel)}`}>
                        {selectedCase.riskScore}
                      </span>
                      <span className="text-[9px] text-slate-500 uppercase tracking-wide">Risk Score</span>
                    </div>
                  </div>

                  <div className="flex-1">
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div>
                        <div className="text-base font-bold text-white leading-tight">{selectedCase.title}</div>
                        <div className="text-xs text-slate-500 mt-0.5">{selectedCase.platform} · {selectedCase.duration}</div>
                      </div>
                      <span className={`badge border text-xs flex-shrink-0 ${getRiskBg(selectedCase.riskLevel)}`}>
                        {getRiskLabel(selectedCase.riskLevel)} риск
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mb-3">
                      <AlertTriangle className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
                      <span className="text-xs font-medium text-orange-300">{selectedCase.categoryRu}</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">{selectedCase.description}</p>
                    <div className="mt-3 p-2.5 rounded-lg bg-yellow-500/8 border border-yellow-500/15">
                      <p className="text-[10px] text-yellow-300/80 leading-relaxed">
                        <span className="font-semibold">Главный сигнал:</span> {selectedCase.mainReason}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Score Breakdown */}
                <div className="mt-4 pt-4 border-t border-white/[0.06]">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">Формула Risk Score</div>
                  <div className="grid grid-cols-5 gap-2">
                    {[
                      { label: 'Текст и речь', weight: 35 },
                      { label: 'Визуальные', weight: 25 },
                      { label: 'Метаданные', weight: 15 },
                      { label: 'Поведение', weight: 15 },
                      { label: 'Похожесть', weight: 10 },
                    ].map((item) => (
                      <div key={item.label} className="text-center">
                        <div className="text-[10px] font-mono text-violet-400">{item.weight}%</div>
                        <div className="text-[9px] text-slate-600 leading-tight mt-0.5">{item.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Evidence Cards */}
              <div className="grid grid-cols-2 gap-3">
                {selectedCase.evidenceCards.map((card, i) => {
                  const Icon = evidenceIcons[card.type];
                  const confColor = card.confidence >= 85 ? '#ef4444' : card.confidence >= 70 ? '#f97316' : card.confidence >= 50 ? '#eab308' : '#22c55e';
                  return (
                    <motion.div
                      key={card.type}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.06 }}
                      className="card p-3"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 rounded-md bg-white/[0.05] flex items-center justify-center">
                          <Icon className="w-3.5 h-3.5 text-violet-400" />
                        </div>
                        <span className="text-xs font-semibold text-white">{card.title}</span>
                        {card.timestamp && (
                          <span className="ml-auto flex items-center gap-1 text-[10px] text-slate-500">
                            <Clock className="w-2.5 h-2.5" />
                            {card.timestamp}
                          </span>
                        )}
                      </div>
                      <ConfidenceBar value={card.confidence} color={confColor} />
                      <div className="mt-2 p-2 rounded bg-white/[0.03] border border-white/[0.05]">
                        <div className="text-[10px] font-mono text-cyan-300 leading-relaxed">«{card.fragment}»</div>
                      </div>
                      <p className="text-[10px] text-slate-500 leading-relaxed mt-2">{card.explanation}</p>
                      {card.findings.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {card.findings.map((f) => (
                            <span key={f} className="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/[0.08] text-slate-400 font-mono">{f}</span>
                          ))}
                        </div>
                      )}
                    </motion.div>
                  );
                })}
              </div>

              {/* Ethics Note */}
              <div className="p-3 rounded-xl bg-blue-500/[0.06] border border-blue-500/15">
                <div className="flex items-start gap-2">
                  <Shield className="w-3.5 h-3.5 text-blue-400 mt-0.5 flex-shrink-0" />
                  <p className="text-[10px] text-blue-300/80 leading-relaxed">
                    <span className="font-semibold">AI Media Watch</span> не выносит автоматических юридических решений. Система помогает приоритизировать контент для ручной проверки, объясняя найденные признаки риска. Результат требует верификации специалистом.
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
