export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
export type SignalSource = 'OCR' | 'Audio' | 'Visual' | 'Metadata' | 'Behavior';
export type CaseStatus = 'new' | 'reviewing' | 'confirmed' | 'false_positive' | 'archived';
export type EvidenceType = 'audio' | 'ocr' | 'visual' | 'metadata' | 'links' | 'engagement';
export type Platform = 'Instagram' | 'TikTok' | 'YouTube' | 'Telegram' | 'VK';

export interface ScamDNADimension {
  key: string;
  name: string;
  nameRu: string;
  value: number;
  description: string;
}

export interface TimelineEvent {
  id: string;
  time: string;
  timeSeconds: number;
  source: SignalSource;
  signal: string;
  confidence: number;
  severity: RiskLevel;
}

export interface EvidenceCard {
  type: EvidenceType;
  title: string;
  confidence: number;
  fragment: string;
  explanation: string;
  timestamp?: string;
  findings: string[];
}

export interface ConnectionNode {
  id: string;
  type: 'video' | 'account' | 'hashtag' | 'telegram';
  label: string;
  riskScore?: number;
  x: number;
  y: number;
}

export interface ConnectionEdge {
  source: string;
  target: string;
  type: 'account' | 'telegram' | 'hashtag' | 'related' | 'pattern';
}

export interface DemoCase {
  id: string;
  title: string;
  platform: Platform;
  duration: string;
  riskScore: number;
  riskLevel: RiskLevel;
  category: string;
  categoryRu: string;
  status: CaseStatus;
  uploadDate: string;
  description: string;
  mainReason: string;
  hashtags: string[];
  scamDNA: ScamDNADimension[];
  timeline: TimelineEvent[];
  evidenceCards: EvidenceCard[];
  connections: {
    nodes: ConnectionNode[];
    edges: ConnectionEdge[];
    clusterSize: number;
    clusterDescription: string;
  };
}
