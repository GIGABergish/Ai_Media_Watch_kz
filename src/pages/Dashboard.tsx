import { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar,
} from 'recharts';
import { TrendingUp, AlertTriangle, Eye, Network, ArrowUpRight, Flame, Clock, ChevronRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { demoCases, dashboardStats } from '../data/cases';
import { DemoCase } from '../types';
import { getRiskBg, getRiskLabel, getRiskColor, formatDate } from '../lib/utils';

interface DashboardProps {
  onNavigateAnalysis: () => void;
  onSelectCase: (c: DemoCase) => void;
}

function AnimatedNumber({ target, duration = 1500 }: { target: number; duration?: number }) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const startTime = Date.now();
    const tick = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.floor(eased * target));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, duration]);
  return <>{value.toLocaleString('ru-RU')}</>;
}

const kpis = [
  {
    label: 'Проанализировано видео',
    value: 1248,
    icon: Eye,
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    delta: '+142 за 7 дней',
    positive: true,
  },
  {
    label: 'Высокий риск',
    value: 87,
    icon: AlertTriangle,
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/20',
    delta: '+12 за 7 дней',
    positive: false,
  },
  {
    label: 'Требует проверки',
    value: 146,
    icon: Clock,
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/20',
    delta: '+23 за 7 дней',
    positive: false,
  },
  {
    label: 'Выявлено кластеров',
    value: 12,
    icon: Network,
    color: 'text-violet-400',
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/20',
    delta: '+3 за 7 дней',
    positive: false,
  },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#13131f] border border-white/10 rounded-lg p-3 text-xs">
        <div className="text-slate-400 mb-1">{label}</div>
        {payload.map((p: any, i: number) => (
          <div key={i} style={{ color: p.color }}>{p.name}: {p.value}</div>
        ))}
      </div>
    );
  }
  return null;
};

export default function Dashboard({ onNavigateAnalysis, onSelectCase }: DashboardProps) {
  const recentCases = [...demoCases].sort((a, b) => b.riskScore - a.riskScore).slice(0, 4);

  return (
    <div className="space-y-5">
      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-4">
        {kpis.map((kpi, i) => {
          const Icon = kpi.icon;
          return (
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              className={`card p-4 border ${kpi.border}`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className={`w-9 h-9 rounded-lg ${kpi.bg} flex items-center justify-center`}>
                  <Icon className={`w-4 h-4 ${kpi.color}`} />
                </div>
                <div className={`flex items-center gap-1 text-[10px] font-medium ${kpi.positive ? 'text-green-400' : 'text-red-400'}`}>
                  <TrendingUp className="w-3 h-3" />
                  {kpi.delta}
                </div>
              </div>
              <div className={`text-2xl font-bold ${kpi.color} font-mono`}>
                <AnimatedNumber target={kpi.value} />
              </div>
              <div className="text-xs text-slate-500 mt-0.5">{kpi.label}</div>
            </motion.div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Area Chart */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="card col-span-2 p-4"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-sm font-semibold text-white">Активность за 7 дней</div>
              <div className="text-xs text-slate-500">Анализ и выявленные риски</div>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-blue-400" /><span className="text-slate-400">Проанализировано</span></div>
              <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-red-400" /><span className="text-slate-400">Высокий риск</span></div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={dashboardStats.weeklyTrend}>
              <defs>
                <linearGradient id="colorAnalyzed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="analyzed" name="Проанализировано" stroke="#3b82f6" strokeWidth={2} fill="url(#colorAnalyzed)" />
              <Area type="monotone" dataKey="highRisk" name="Высокий риск" stroke="#ef4444" strokeWidth={2} fill="url(#colorRisk)" />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Donut Chart */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.42 }}
          className="card p-4"
        >
          <div className="text-sm font-semibold text-white mb-1">Категории нарушений</div>
          <div className="text-xs text-slate-500 mb-3">Распределение по типам</div>
          <ResponsiveContainer width="100%" height={140}>
            <PieChart>
              <Pie
                data={dashboardStats.categoryDistribution}
                cx="50%"
                cy="50%"
                innerRadius={42}
                outerRadius={65}
                paddingAngle={3}
                dataKey="value"
              >
                {dashboardStats.categoryDistribution.map((entry, index) => (
                  <Cell key={index} fill={entry.color} opacity={0.85} />
                ))}
              </Pie>
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload?.[0]) {
                    return (
                      <div className="bg-[#13131f] border border-white/10 rounded-lg p-2 text-xs">
                        <div className="text-white">{payload[0].name}</div>
                        <div className="text-slate-400">{payload[0].value}%</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5 mt-2">
            {dashboardStats.categoryDistribution.map((item) => (
              <div key={item.name} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-slate-400">{item.name}</span>
                </div>
                <span className="font-mono text-white">{item.value}%</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Risk Distribution Bar */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.48 }}
          className="card p-4"
        >
          <div className="text-sm font-semibold text-white mb-1">Распределение риска</div>
          <div className="text-xs text-slate-500 mb-3">По диапазонам Risk Score</div>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={dashboardStats.riskDistribution} barSize={24}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis dataKey="range" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" name="Видео" radius={[4, 4, 0, 0]}>
                {dashboardStats.riskDistribution.map((entry, index) => {
                  const colors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'];
                  return <Cell key={index} fill={colors[index]} opacity={0.8} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Recent Suspicious Videos */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.52 }}
          className="card p-4 col-span-1"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold text-white">Последние видео</div>
            <button onClick={onNavigateAnalysis} className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1">
              Все <ChevronRight className="w-3 h-3" />
            </button>
          </div>
          <div className="space-y-2">
            {recentCases.map((c) => (
              <div
                key={c.id}
                onClick={() => onSelectCase(c)}
                className="flex items-center gap-3 p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.05] hover:border-violet-500/20 hover:bg-white/[0.04] transition-all cursor-pointer group"
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600/30 to-blue-600/30 border border-white/10 flex items-center justify-center flex-shrink-0">
                  <span className={`text-xs font-bold font-mono ${getRiskColor(c.riskLevel)}`}>{c.riskScore}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-white truncate group-hover:text-violet-300 transition-colors">{c.title}</div>
                  <div className="text-[10px] text-slate-500">{c.platform} · {formatDate(c.uploadDate)}</div>
                </div>
                <span className={`badge border text-[10px] ${getRiskBg(c.riskLevel)}`}>
                  {getRiskLabel(c.riskLevel)}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Top Emerging Pattern */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.56 }}
          className="card p-4 border-red-500/20 bg-red-500/[0.03]"
        >
          <div className="flex items-center gap-2 mb-3">
            <Flame className="w-4 h-4 text-red-400" />
            <div className="text-sm font-semibold text-white">Top Emerging Pattern</div>
          </div>
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <div className="text-xs font-semibold text-red-400 mb-1">Telegram Casino Funnel v2</div>
              <div className="text-[11px] text-slate-400 leading-relaxed">
                Новый паттерн: короткие видео 15–30 сек с промокодом, перенаправление через bio-link в Telegram
              </div>
            </div>
            <div className="space-y-2">
              {[
                { label: 'Совпадений за 7 дней', value: '34 видео', color: 'text-red-400' },
                { label: 'Платформы', value: 'Instagram, TikTok', color: 'text-orange-400' },
                { label: 'Риск паттерна', value: '89 / 100', color: 'text-red-400' },
                { label: 'Новых аккаунтов', value: '7 выявлено', color: 'text-yellow-400' },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">{item.label}</span>
                  <span className={`font-mono font-medium ${item.color}`}>{item.value}</span>
                </div>
              ))}
            </div>
            <button
              onClick={onNavigateAnalysis}
              className="w-full btn btn-ghost text-xs justify-center"
            >
              Открыть анализ
              <ArrowUpRight className="w-3 h-3" />
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
