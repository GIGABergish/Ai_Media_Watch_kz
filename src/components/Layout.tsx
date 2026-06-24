import Sidebar from './Sidebar';
import TopBar from './TopBar';
import { Page } from '../App';

interface LayoutProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
  children: React.ReactNode;
}

export default function Layout({ currentPage, onNavigate, children }: LayoutProps) {
  return (
    <div className="flex h-screen bg-[#07070f] overflow-hidden">
      <Sidebar currentPage={currentPage} onNavigate={onNavigate} />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar currentPage={currentPage} onNavigate={onNavigate} />
        <main className="flex-1 overflow-y-auto p-5">
          {children}
        </main>
      </div>
    </div>
  );
}
