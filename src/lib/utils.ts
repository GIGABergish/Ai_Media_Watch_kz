import { clsx, type ClassValue } from 'clsx';
import { RiskLevel } from '../types';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function getRiskColor(level: RiskLevel): string {
  switch (level) {
    case 'critical': return 'text-red-400';
    case 'high': return 'text-orange-400';
    case 'medium': return 'text-yellow-400';
    case 'low': return 'text-green-400';
  }
}

export function getRiskBg(level: RiskLevel): string {
  switch (level) {
    case 'critical': return 'bg-red-500/10 border-red-500/25 text-red-400';
    case 'high': return 'bg-orange-500/10 border-orange-500/25 text-orange-400';
    case 'medium': return 'bg-yellow-500/10 border-yellow-500/25 text-yellow-400';
    case 'low': return 'bg-green-500/10 border-green-500/25 text-green-400';
  }
}

export function getRiskLabel(level: RiskLevel): string {
  switch (level) {
    case 'critical': return 'Критический';
    case 'high': return 'Высокий';
    case 'medium': return 'Средний';
    case 'low': return 'Низкий';
  }
}

export function getStatusLabel(status: string): string {
  switch (status) {
    case 'new': return 'Новый';
    case 'reviewing': return 'На проверке';
    case 'confirmed': return 'Подтверждено';
    case 'false_positive': return 'Ложное срабатывание';
    case 'archived': return 'Архив';
    default: return status;
  }
}

export function getStatusStyle(status: string): string {
  switch (status) {
    case 'new': return 'bg-blue-500/10 border-blue-500/25 text-blue-400';
    case 'reviewing': return 'bg-yellow-500/10 border-yellow-500/25 text-yellow-400';
    case 'confirmed': return 'bg-red-500/10 border-red-500/25 text-red-400';
    case 'false_positive': return 'bg-green-500/10 border-green-500/25 text-green-400';
    case 'archived': return 'bg-slate-500/10 border-slate-500/25 text-slate-400';
    default: return 'bg-slate-500/10 border-slate-500/25 text-slate-400';
  }
}

export function getSourceColor(source: string): string {
  switch (source) {
    case 'OCR': return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/25';
    case 'Audio': return 'text-violet-400 bg-violet-500/10 border-violet-500/25';
    case 'Visual': return 'text-blue-400 bg-blue-500/10 border-blue-500/25';
    case 'Metadata': return 'text-orange-400 bg-orange-500/10 border-orange-500/25';
    case 'Behavior': return 'text-pink-400 bg-pink-500/10 border-pink-500/25';
    default: return 'text-slate-400 bg-slate-500/10 border-slate-500/25';
  }
}

export function getRiskScoreGradient(score: number): string {
  if (score >= 80) return 'from-red-600 to-red-400';
  if (score >= 60) return 'from-orange-600 to-orange-400';
  if (score >= 40) return 'from-yellow-600 to-yellow-400';
  return 'from-green-600 to-green-400';
}

export function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
}
