import { Moon, Shield, Sun } from "lucide-react";

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export function Navigation({ currentPage, onNavigate, theme, onToggleTheme }: NavigationProps) {
  const navItems = [
    { name: "Home", id: "home" },
    { name: "Results", id: "results" },
    { name: "Dataset", id: "dataset" },
    { name: "Protocol", id: "protocol" },
    { name: "Model & Performance", id: "model" },
    { name: "Contact", id: "contact" },
  ];

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/95 shadow-sm backdrop-blur">
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

          <div className="flex items-center gap-3">
            {/* Navigation Items */}
            <div className="hidden md:flex items-center gap-8">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`rounded-lg px-3 py-2 transition-colors ${
                    currentPage === item.id ? "font-medium" : "text-foreground/80 hover:bg-accent"
                  }`}
                  style={{
                    color: currentPage === item.id ? "var(--viet-primary)" : undefined,
                    backgroundColor:
                      currentPage === item.id ? "color-mix(in srgb, var(--viet-primary) 10%, transparent)" : "transparent",
                  }}
                >
                  {item.name}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={onToggleTheme}
              aria-label={theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
              title={theme === "dark" ? "Light mode" : "Dark mode"}
              className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-border bg-muted/70 text-foreground/80 transition-colors hover:bg-accent"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
