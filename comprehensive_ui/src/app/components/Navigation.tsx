import { Shield } from "lucide-react";

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
}

export function Navigation({ currentPage, onNavigate }: NavigationProps) {
  const navItems = [
    { name: "Home", id: "home" },
    { name: "Results", id: "results" },
    { name: "Dataset", id: "dataset" },
    { name: "Model & Performance", id: "model" },
    { name: "Contact", id: "contact" },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => onNavigate("home")}
          >
            <div className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: "var(--viet-primary)" }}
            >
              <Shield className="w-6 h-6 text-white" />
            </div>
            <span
              className="text-xl tracking-tight"
              style={{ color: "var(--viet-primary)" }}
            >
              VietToxic Detector
            </span>
          </div>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center gap-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`transition-colors px-3 py-2 rounded-lg ${
                  currentPage === item.id
                    ? "font-medium"
                    : "hover:bg-gray-100"
                }`}
                style={{
                  color: currentPage === item.id ? "var(--viet-primary)" : "#374151",
                  backgroundColor: currentPage === item.id ? "rgba(0, 51, 102, 0.05)" : "transparent",
                }}
              >
                {item.name}
              </button>
            ))}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button className="p-2 rounded-lg hover:bg-gray-100">
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
