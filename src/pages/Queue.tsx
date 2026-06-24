import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, ChevronRight, SlidersHorizontal } from 'lucide-react';
import { demoCases } from '../data/cases';
import { DemoCase, RiskLevel, Platform, CaseStatus } from '../types';
import { getRiskBg, getRiskColor, getRiskLabel, getStatusLabel, getStatusStyle, formatDate } from '../lib/utils';

interface QueuePageProps {
  onSelectCase: (c: DemoCase) => void;
}

const platforms: Platform[] = ['Instagram', 'TikTok', 'YouTube', 'Telegram', 'VK'];
const riskLevels: RiskLevel[] = ['critical', 'high', 'medium', 'low'];
const statuses: CaseStatus[] = ['new', 'reviewing', 'confirmed', 'false_positive', 'archived'];

const queueData: DemoCase[] = [
  ...demoCases,
  {
    ...demoCases[0],
    id: 'q-001',
    title: 'Быстрый заработок на спорт прогнозах',
    platform: 'TikTok',
    riskScore: 79,
    riskLevel: 'high',
    status: 'new',
    uploadDate: '2026-06-22',
    mainReason: 'Обещание 90% точности прогнозов',
  },
  {
    ...demoCases[1],
    id: 'q-002',
    title: 'Криптовалютный сигнальный бот',
    platform: 'Telegram',
    riskScore: 83,
    riskLevel: 'high',
    status: 'reviewing',
    uploadDate: '2026-06-22',
    mainReason: 'Реферальная схема + гарантия прибыли',
  },
  {
    ...demoCases[3],
    id: 'q-003',
    title: 'NFT клуб: доход 200% за месяц',
    platform: 'Instagram',
    riskScore: 76,
    riskLevel: 'high',
    status: 'new',
    uploadDate: '2026-06-21',
    mainReason: 'Нереальная доходность NFT + реферал',
  },
];

export default function QueuePage({ onSelectCase }: QueuePageProps) {
  const [search, setSearch] = useState('');
  const [filterRisk, setFilterRisk] = useState<RiskLevel | 'all'>('all');
  const [filterStatus, setFilterStatus] = useState<CaseStatus | 'all'>('all');
  const [filterPlatform, setFilterPlatform] = useState<Platform | 'all'>('all');
  const [showFilters, setShowFilters] = useState(false);

  const filtered = queueData.filter((c) => {
    const matchSearch = !search || c.title.toLowerCase().includes(search.toLowerCase());
    const matchRisk = filterRisk === 'all' || c.riskLevel === filterRisk;
    const matchStatus = filterStatus === 'all' || c.status === filterStatus;
    const matchPlatform = filterPlatform === 'all' || c.platform === filterPlatform;
    return matchSearch && matchRisk && matchStatus && matchPlatform;
  });

  return (
    <div className="space-y-4">
      {/* Header Controls */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
          <input
            type="text"
            placeholder="Поиск по названию..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-[#0d0d1a] border border-white/[0.08] rounded-lg text-sm text-white placeholder-slate-600 focus:outline-none focus:border-violet-500/50 transition-colors"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`btn btn-ghost gap-2 ${showFilters ? 'border-violet-500/40 text-violet-300' : ''}`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          Фильтры
        </button>
        <div className="ml-auto text-xs text-slate-500">
          Показано: <span className="text-white font-mono">{filtered.length}</span> / {queueData.length}
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="card p-4"
        >
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-2">Уровень риска</label>
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={() => setFilterRisk('all')}
                  className={`text-xs px-2 py-1 rounded border transition-all ${filterRisk === 'all' ? 'bg-violet-500/20 border-violet-500/40 text-violet-300' : 'border-white/[0.07] text-slate-400 hover:border-white/20'}`}
                >
                  Все
                </button>
                {riskLevels.map((r) => (
                  <button
                    key={r}
                    onClick={() => setFilterRisk(r)}
                    className={`text-xs px-2 py-1 rounded border transition-all ${filterRisk === r ? `${getRiskBg(r)} border` : 'border-white/[0.07] text-slate-400 hover:border-white/20'}`}
                  >
                    {getRiskLabel(r)}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-2">Статус</label>
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={() => setFilterStatus('all')}
                  className={`text-xs px-2 py-1 rounded border transition-all ${filterStatus === 'all' ? 'bg-violet-500/20 border-violet-500/40 text-violet-300' : 'border-white/[0.07] text-slate-400 hover:border-white/20'}`}
                >
                  Все
                </button>
                {statuses.map((s) => (
                  <button
                    key={s}
                    onClick={() => setFilterStatus(s)}
                    className={`text-xs px-2 py-1 rounded border transition-all ${filterStatus === s ? `${getStatusStyle(s)} border` : 'border-white/[0.07] text-slate-400 hover:border-white/20'}`}
                  >
                    {getStatusLabel(s)}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-2">Платформа</label>
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={() => setFilterPlatform('all')}
                  className={`text-xs px-2 py-1 rounded border transition-all ${filterPlatform === 'all' ? 'bg-violet-500/20 border-violet-500/40 text-violet-300' : 'border-white/[0.07] text-slate-400 hover:border-white/20'}`}
                >
                  Все
                </button>
                {platforms.map((p) => (
                  <button
                    key={p}
                    onClick={() => setFilterPlatform(p as Platform)}
                    className={`text-xs px-2 py-1 rounded border transition-all ${filterPlatform === p ? 'bg-blue-500/20 border-blue-500/40 text-blue-300' : 'border-white/[0.07] text-slate-400 hover:border-white/20'}`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/[0.06]">
                {['Видео', 'Платформа', 'Risk Score', 'Категория', 'Главная причина', 'Статус', 'Дата', ''].map((h) => (
                  <th key={h} className="text-left text-[10px] font-semibold text-slate-500 uppercase tracking-wider px-4 py-3 whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((c, i) => (
                <motion.tr
                  key={c.id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                  onClick={() => onSelectCase(c)}
                  className="border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors cursor-pointer group"
                >
                  <td className="px-4 py-3">
                    <div className="text-xs font-medium text-white group-hover:text-violet-300 transition-colors max-w-48 truncate">
                      {c.title}
                    </div>
                    <div className="text-[10px] text-slate-600">{c.duration}</div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs text-slate-400">{c.platform}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${c.riskScore}%`,
                            backgroundColor: c.riskLevel === 'critical' ? '#ef4444'
                              : c.riskLevel === 'high' ? '#f97316'
                              : c.riskLevel === 'medium' ? '#eab308'
                              : '#22c55e',
                          }}
                        />
                      </div>
                      <span className={`text-xs font-bold font-mono ${getRiskColor(c.riskLevel)}`}>{c.riskScore}</span>
                    </div>
                    <span className={`text-[10px] ${getRiskColor(c.riskLevel)}`}>{getRiskLabel(c.riskLevel)}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-[10px] text-slate-400 max-w-28 block truncate">{c.categoryRu}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-[10px] text-slate-500 max-w-40 block truncate">{c.mainReason}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`badge border text-[10px] ${getStatusStyle(c.status)}`}>
                      {getStatusLabel(c.status)}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-[10px] text-slate-500 whitespace-nowrap">{formatDate(c.uploadDate)}</span>
                  </td>
                  <td className="px-4 py-3">
                    <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-violet-400 transition-colors" />
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div className="py-12 text-center text-slate-500 text-sm">
              Нет записей, соответствующих фильтрам
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
