import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Network, Video, User, Hash, MessageCircle, AlertTriangle, Info } from 'lucide-react';
import { DemoCase, ConnectionNode, ConnectionEdge } from '../types';
import { getRiskColor } from '../lib/utils';

interface ConnectionsPageProps {
  selectedCase: DemoCase;
  onBack: () => void;
}

const nodeColors: Record<ConnectionNode['type'], string> = {
  video: '#8b5cf6',
  account: '#3b82f6',
  hashtag: '#06b6d4',
  telegram: '#f97316',
};

const nodeIcons: Record<ConnectionNode['type'], React.ElementType> = {
  video: Video,
  account: User,
  hashtag: Hash,
  telegram: MessageCircle,
};

function NodeIcon({ type, x, y, node, isSource }: {
  type: ConnectionNode['type'];
  x: number;
  y: number;
  node: ConnectionNode;
  isSource: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  const color = nodeColors[type];
  const size = isSource ? 52 : type === 'telegram' ? 44 : 36;

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ cursor: 'pointer' }}
    >
      <circle r={size / 2} fill={`${color}15`} stroke={color} strokeWidth={isSource ? 2.5 : 1.5} opacity={0.9} />
      {isSource && (
        <circle r={size / 2 + 8} fill="none" stroke={color} strokeWidth={1} opacity={0.2} strokeDasharray="4 3" />
      )}
      {node.riskScore !== undefined && node.riskScore > 0 && (
        <>
          <text y={size / 2 - 3} textAnchor="middle" fontSize={8} fontFamily="JetBrains Mono" fill={color} opacity={0.8}>
            {node.riskScore}
          </text>
        </>
      )}
      {hovered && (
        <foreignObject x={-60} y={size / 2 + 5} width={120} height={40} style={{ pointerEvents: 'none' }}>
          <div style={{
            background: '#0d0d1a',
            border: `1px solid ${color}40`,
            borderRadius: 6,
            padding: '4px 8px',
            fontSize: 10,
            color: '#e2e8f0',
            textAlign: 'center',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>
            {node.label}
          </div>
        </foreignObject>
      )}
    </g>
  );
}

function buildPositionedNodes(nodes: ConnectionNode[], sourceId: string): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  nodes.forEach((n) => {
    positions.set(n.id, { x: n.x, y: n.y });
  });
  return positions;
}

export default function ConnectionsPage({ selectedCase, onBack }: ConnectionsPageProps) {
  const { nodes, edges, clusterSize, clusterDescription } = selectedCase.connections;
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

  const positions = buildPositionedNodes(nodes, selectedCase.id);

  const edgeColors: Record<string, string> = {
    account: '#3b82f6',
    telegram: '#f97316',
    hashtag: '#06b6d4',
    related: '#8b5cf6',
    pattern: '#ec4899',
  };

  const typeLegend = [
    { type: 'video', label: 'Видео', color: '#8b5cf6' },
    { type: 'account', label: 'Аккаунт', color: '#3b82f6' },
    { type: 'hashtag', label: 'Хэштег', color: '#06b6d4' },
    { type: 'telegram', label: 'Telegram', color: '#f97316' },
  ];

  const edgeLegend = [
    { type: 'account', label: 'Один аккаунт', color: '#3b82f6' },
    { type: 'telegram', label: 'Telegram ссылка', color: '#f97316' },
    { type: 'hashtag', label: 'Общий хэштег', color: '#06b6d4' },
    { type: 'related', label: 'Связанные', color: '#8b5cf6' },
  ];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={onBack} className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.08] flex items-center justify-center hover:bg-white/[0.08] transition-colors">
          <ArrowLeft className="w-4 h-4 text-slate-400" />
        </button>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
            <Network className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">Карта связей</h2>
            <p className="text-xs text-slate-500">Граф аналитических гипотез</p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div className="px-3 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-orange-400" />
            <span className="text-xs font-medium text-orange-300">Вероятный кластер: {clusterSize} публикаций</span>
          </div>
        </div>
      </div>

      {/* Ethics Note */}
      <div className="flex items-start gap-2 p-3 rounded-xl bg-blue-500/[0.05] border border-blue-500/15">
        <Info className="w-3.5 h-3.5 text-blue-400 mt-0.5 flex-shrink-0" />
        <p className="text-[10px] text-blue-300/80 leading-relaxed">
          <span className="font-semibold">Аналитическая гипотеза, не обвинение.</span> Карта показывает статистические связи между публикациями. {clusterDescription}.
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {/* Graph */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="col-span-3 card p-4 overflow-hidden"
          style={{ minHeight: 440 }}
        >
          <svg
            ref={svgRef}
            width="100%"
            height="420"
            viewBox="0 0 600 430"
            style={{ overflow: 'visible' }}
          >
            <defs>
              {Object.entries(edgeColors).map(([type, color]) => (
                <marker
                  key={type}
                  id={`arrow-${type}`}
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={color} opacity={0.6} />
                </marker>
              ))}
            </defs>

            {/* Edges */}
            {edges.map((edge: ConnectionEdge, i) => {
              const sourceNode = nodes.find((n) => n.id === edge.source);
              const targetNode = nodes.find((n) => n.id === edge.target);
              if (!sourceNode || !targetNode) return null;

              const color = edgeColors[edge.type] || '#8b5cf6';
              const edgeId = `${edge.source}-${edge.target}`;

              return (
                <line
                  key={i}
                  x1={sourceNode.x}
                  y1={sourceNode.y}
                  x2={targetNode.x}
                  y2={targetNode.y}
                  stroke={color}
                  strokeWidth={hoveredEdge === edgeId ? 2 : 1}
                  strokeOpacity={hoveredEdge === edgeId ? 0.8 : 0.3}
                  strokeDasharray={edge.type === 'hashtag' ? '4 3' : edge.type === 'related' ? '6 3' : 'none'}
                  markerEnd={`url(#arrow-${edge.type})`}
                  onMouseEnter={() => setHoveredEdge(edgeId)}
                  onMouseLeave={() => setHoveredEdge(null)}
                  style={{ cursor: 'pointer', transition: 'stroke-opacity 0.2s' }}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((node, i) => {
              const isSource = i === 0;
              const color = nodeColors[node.type];
              return (
                <motion.g
                  key={node.id}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 + i * 0.08, type: 'spring', stiffness: 200 }}
                  style={{ transformOrigin: `${node.x}px ${node.y}px` }}
                >
                  <NodeIcon
                    type={node.type}
                    x={node.x}
                    y={node.y}
                    node={node}
                    isSource={isSource}
                  />
                  <text
                    x={node.x}
                    y={node.y + (isSource ? 34 : 26)}
                    textAnchor="middle"
                    fontSize={9}
                    fill="#94a3b8"
                    fontFamily="Inter, sans-serif"
                  >
                    {node.label.length > 18 ? node.label.slice(0, 18) + '…' : node.label}
                  </text>
                </motion.g>
              );
            })}
          </svg>
        </motion.div>

        {/* Legend & Stats */}
        <div className="space-y-4">
          <div className="card p-4">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Узлы графа</div>
            <div className="space-y-2">
              {typeLegend.map((item) => {
                const Icon = nodeIcons[item.type as ConnectionNode['type']];
                const count = nodes.filter((n) => n.type === item.type).length;
                return (
                  <div key={item.type} className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-md flex items-center justify-center border"
                      style={{ backgroundColor: `${item.color}15`, borderColor: `${item.color}30` }}>
                      <Icon className="w-3 h-3" style={{ color: item.color }} />
                    </div>
                    <span className="text-xs text-slate-400 flex-1">{item.label}</span>
                    <span className="text-xs font-mono text-white">{count}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card p-4">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Типы связей</div>
            <div className="space-y-2">
              {edgeLegend.map((item) => (
                <div key={item.type} className="flex items-center gap-2">
                  <div className="w-8 h-0.5 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-[10px] text-slate-400">{item.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Метрики кластера</div>
            <div className="space-y-2">
              {[
                { label: 'Публикаций', value: clusterSize },
                { label: 'Уникальных связей', value: edges.length },
                { label: 'Узлов в графе', value: nodes.length },
                { label: 'Telegram-каналов', value: nodes.filter(n => n.type === 'telegram').length },
              ].map((m) => (
                <div key={m.label} className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">{m.label}</span>
                  <span className="font-mono font-bold text-white">{m.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
