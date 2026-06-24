import { Shield, LayoutDashboard, Video, ClipboardList, Network, Dna, BarChart3, Settings, Zap } from 'lucide-react';
import { Page } from '../App';
import { cn } from '../lib/utils';

interface SidebarProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

const navItems: { id: Page; label: string; icon: React.ElementType; badge?: string }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'analysis', label: 'Анализ видео', icon: Video },
  { id: 'queue', label: 'Очередь проверки', icon: ClipboardList, badge: '146' },
  { id: 'connections', label: 'Карта связей', icon: Network },
  { id: 'scam-dna', label: 'Scam DNA', icon: Dna },
  { id: 'timeline', label: 'Evidence Timeline', icon: Zap },
  { id: 'settings', label: 'Настройки', icon: Settings },
];

export default function Sidebar({ currentPage, onNavigate }: SidebarProps) {
  return (
    <aside className="w-60 flex-shrink-0 bg-[#090912] border-r border-white/[0.06] flex flex-col h-full">
      <div className="p-5 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-violet-600 flex items-center justify-center glow-purple">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="text-sm font-bold text-white leading-tight">Sentinel</div>
            <div className="text-xs text-violet-400 leading-tight font-mono">Media AI</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
        <div className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider px-3 py-2 mt-1">
          Навигация
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn(
                'sidebar-link w-full text-left',
                isActive && 'active'
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span className="flex-1">{item.label}</span>
              {item.badge && (
                <span className="text-[10px] font-mono bg-violet-500/20 text-violet-400 px-1.5 py-0.5 rounded-full border border-violet-500/20">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-white/[0.06]">
        <div className="rounded-lg bg-violet-500/8 border border-violet-500/15 p-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs font-medium text-green-400">Система активна</span>
          </div>
          <div className="text-[10px] text-slate-500 leading-relaxed">
            AI Media Watch v1.0<br />
            Мониторинг: 24/7
          </div>
        </div>
        <div className="mt-3 text-[10px] text-slate-600 leading-relaxed">
          Результат не является юридическим заключением
        </div>
      </div>
    </aside>
  );
}
