import { useState, useEffect } from 'react';
import { Bell, RefreshCw, Globe, Activity } from 'lucide-react';
import { Page } from '../App';

const pageTitles: Record<Page, { title: string; subtitle: string }> = {
  dashboard: { title: 'Dashboard', subtitle: 'Обзор системы мониторинга' },
  analysis: { title: 'Анализ видео', subtitle: 'Multimodal AI Detection' },
  queue: { title: 'Очередь проверки', subtitle: 'Управление задачами' },
  connections: { title: 'Карта связей', subtitle: 'Граф подозрительных связей' },
  'scam-dna': { title: 'Scam DNA', subtitle: 'Цифровой отпечаток риска' },
  timeline: { title: 'Evidence Timeline', subtitle: 'Хронология доказательств' },
  settings: { title: 'Настройки', subtitle: 'Конфигурация системы' },
};

interface TopBarProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

export default function TopBar({ currentPage, onNavigate }: TopBarProps) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const { title, subtitle } = pageTitles[currentPage];

  return (
    <header className="h-14 bg-[#090912] border-b border-white/[0.06] flex items-center px-5 gap-4 flex-shrink-0">
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2">
          <h1 className="text-sm font-semibold text-white">{title}</h1>
          <span className="text-xs text-slate-500">/</span>
          <span className="text-xs text-slate-500 truncate">{subtitle}</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-green-500/10 border border-green-500/20">
          <Activity className="w-3 h-3 text-green-400" />
          <span className="text-xs font-medium text-green-400">Monitoring Active</span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.07]">
          <Globe className="w-3 h-3 text-slate-400" />
          <span className="text-xs text-slate-400">RU</span>
        </div>

        <div className="px-2.5 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.07]">
          <span className="text-xs font-mono text-slate-400">
            {time.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        </div>

        <button className="relative w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.07] flex items-center justify-center hover:bg-white/[0.08] transition-colors">
          <Bell className="w-3.5 h-3.5 text-slate-400" />
          <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-red-500" />
        </button>

        <button
          onClick={() => onNavigate('analysis')}
          className="btn btn-primary text-xs"
        >
          <RefreshCw className="w-3 h-3" />
          Новый анализ
        </button>
      </div>
    </header>
  );
}
