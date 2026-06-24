import { useState } from 'react';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import ScamDNAPage from './pages/ScamDNAPage';
import TimelinePage from './pages/Timeline';
import ConnectionsPage from './pages/Connections';
import QueuePage from './pages/Queue';
import SettingsPage from './pages/SettingsPage';
import { demoCases } from './data/cases';
import { DemoCase } from './types';

export type Page = 'dashboard' | 'analysis' | 'queue' | 'connections' | 'scam-dna' | 'timeline' | 'settings';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');
  const [selectedCase, setSelectedCase] = useState<DemoCase>(demoCases[0]);

  const handleSelectCase = (c: DemoCase, page: Page = 'analysis') => {
    setSelectedCase(c);
    setCurrentPage(page);
  };

  return (
    <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
      {currentPage === 'dashboard' && (
        <Dashboard
          onNavigateAnalysis={() => setCurrentPage('analysis')}
          onSelectCase={(c) => handleSelectCase(c, 'analysis')}
        />
      )}
      {currentPage === 'analysis' && (
        <Analysis
          selectedCase={selectedCase}
          onSelectCase={(c) => setSelectedCase(c)}
          onViewScamDNA={() => setCurrentPage('scam-dna')}
          onViewTimeline={() => setCurrentPage('timeline')}
          onViewConnections={() => setCurrentPage('connections')}
        />
      )}
      {currentPage === 'scam-dna' && (
        <ScamDNAPage selectedCase={selectedCase} onBack={() => setCurrentPage('analysis')} />
      )}
      {currentPage === 'timeline' && (
        <TimelinePage selectedCase={selectedCase} onBack={() => setCurrentPage('analysis')} />
      )}
      {currentPage === 'connections' && (
        <ConnectionsPage selectedCase={selectedCase} onBack={() => setCurrentPage('analysis')} />
      )}
      {currentPage === 'queue' && (
        <QueuePage
          onSelectCase={(c) => handleSelectCase(c, 'analysis')}
        />
      )}
      {currentPage === 'settings' && <SettingsPage />}
    </Layout>
  );
}

export default App;
