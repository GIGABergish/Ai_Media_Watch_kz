import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer, Tooltip,
} from 'recharts';
import { motion } from 'framer-motion';
import { ArrowLeft, Dna, Info } from 'lucide-react';
import { DemoCase } from '../types';
import { getRiskColor, getRiskBg, getRiskLabel } from '../lib/utils';

interface ScamDNAPageProps {
  selectedCase: DemoCase;
  onBack: () => void;
}

const CustomRadarTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div className="bg-[#0d0d1a] border border-white/10 rounded-lg p-3 text-xs max-w-52 shadow-xl">
        <div className="font-semibold text-violet-300 mb-1">{d.nameRu}</div>
        <div className="text-2xl font-bold font-mono text-white mb-1">{d.value}<span className="text-sm text-slate-400">/100</span></div>
        <p className="text-slate-400 leading-relaxed">{d.description}</p>
      </div>
    );
  }
  return null;
};

function getSummary(c: DemoCase): string {
  const top = [...c.scamDNA].sort((a, b) => b.value - a.value).slice(0, 3);
  const topNames = top.map((d) => d.nameRu.toLowerCase());
  if (c.riskScore >= 80) {
    return `Главный риск этого контента связан не с одним сигналом, а с комбинацией из ${topNames[0]}, ${topNames[1]} и ${topNames[2]}. Именно эта триада характерна для организованных мошеннических схем.`;
  } else if (c.riskScore >= 50) {
    return `Контент содержит признаки ${topNames[0]} и ${topNames[1]}, однако несколько индикаторов остаются на низком уровне. Требуется дополнительная ручная проверка.`;
  }
  return `Уровень риска низкий. Основные маркеры мошеннических схем не выявлены. Контент соответствует образовательным или информационным материалам.`;
}

export default function ScamDNAPage({ selectedCase, onBack }: ScamDNAPageProps) {
  const data = selectedCase.scamDNA.map((d) => ({
    ...d,
    subject: d.name,
    fullMark: 100,
  }));

  const summary = getSummary(selectedCase);
  const topDimensions = [...selectedCase.scamDNA].sort((a, b) => b.value - a.value);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={onBack} className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.08] flex items-center justify-center hover:bg-white/[0.08] transition-colors">
          <ArrowLeft className="w-4 h-4 text-slate-400" />
        </button>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-violet-500/20 border border-violet-500/30 flex items-center justify-center">
            <Dna className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">Scam DNA</h2>
            <p className="text-xs text-slate-500">Цифровой отпечаток мошеннической схемы</p>
          </div>
        </div>
        <div className="ml-auto">
          <span className={`badge border text-sm px-3 py-1 ${getRiskBg(selectedCase.riskLevel)}`}>
            Risk Score: {selectedCase.riskScore} / 100 · {getRiskLabel(selectedCase.riskLevel)} риск
          </span>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-5">
        {/* Radar Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="col-span-3 card p-6"
        >
          <div className="text-sm font-semibold text-white mb-1">Радар рисков</div>
          <div className="text-xs text-slate-500 mb-4">{selectedCase.title}</div>
          <ResponsiveContainer width="100%" height={380}>
            <RadarChart cx="50%" cy="50%" outerRadius="72%" data={data}>
              <PolarGrid stroke="rgba(255,255,255,0.07)" />
              <PolarAngleAxis
                dataKey="subject"
                tick={({ x, y, payload, index }) => {
                  const d = data[index];
                  const color = d.value >= 80 ? '#ef4444' : d.value >= 60 ? '#f97316' : d.value >= 40 ? '#eab308' : '#22c55e';
                  return (
                    <g transform={`translate(${x},${y})`}>
                      <text
                        textAnchor="middle"
                        dominantBaseline="central"
                        fill={color}
                        fontSize={10}
                        fontWeight={500}
                        fontFamily="JetBrains Mono, monospace"
                      >
                        {d.nameRu}
                      </text>
                    </g>
                  );
                }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 100]}
                tick={{ fill: '#475569', fontSize: 9 }}
                tickCount={5}
                axisLine={false}
              />
              <Radar
                name="Risk DNA"
                dataKey="value"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.25}
                strokeWidth={2}
                dot={{ fill: '#8b5cf6', r: 4, strokeWidth: 0 }}
              />
              <Tooltip content={<CustomRadarTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Dimensions & Summary */}
        <div className="col-span-2 space-y-4">
          {/* Summary */}
          <motion.div
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-4 border-violet-500/20 bg-violet-500/[0.04]"
          >
            <div className="flex items-center gap-2 mb-2">
              <Info className="w-3.5 h-3.5 text-violet-400" />
              <span className="text-xs font-semibold text-violet-300">AI-интерпретация</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">{summary}</p>
          </motion.div>

          {/* Dimension Bars */}
          <motion.div
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="card p-4 space-y-3"
          >
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Измерения риска</div>
            {topDimensions.map((dim, i) => {
              const color = dim.value >= 80 ? '#ef4444' : dim.value >= 60 ? '#f97316' : dim.value >= 40 ? '#eab308' : '#22c55e';
              return (
                <motion.div
                  key={dim.key}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.35 + i * 0.05 }}
                  className="group"
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs text-slate-300 group-hover:text-white transition-colors">{dim.nameRu}</span>
                    <span className="text-xs font-bold font-mono" style={{ color }}>{dim.value}%</span>
                  </div>
                  <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${dim.value}%` }}
                      transition={{ duration: 0.8, delay: 0.4 + i * 0.05, ease: 'easeOut' }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}60` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-600 mt-1 leading-relaxed hidden group-hover:block">
                    {dim.description}
                  </p>
                </motion.div>
              );
            })}
          </motion.div>

          {/* Hashtags */}
          {selectedCase.hashtags.length > 0 && (
            <motion.div
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="card p-4"
            >
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Обнаруженные хэштеги</div>
              <div className="flex flex-wrap gap-1.5">
                {selectedCase.hashtags.map((tag) => (
                  <span key={tag} className="text-[10px] px-2 py-1 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-300 font-mono">
                    {tag}
                  </span>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
