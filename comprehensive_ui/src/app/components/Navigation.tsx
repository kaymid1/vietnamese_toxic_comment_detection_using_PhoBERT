import { Moon, Shield, Sun } from "lucide-react";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/app/components/ui/hover-card";
import type { Language } from "@/app/i18n/messages";
import { useI18n } from "@/app/i18n/context";

type DatasetVersion = "v1" | "latest";

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  language: Language;
  onSetLanguage: (language: Language) => void;
  datasetVersion: DatasetVersion;
  onSetDatasetVersion: (version: DatasetVersion) => void;
}

export function Navigation({
  currentPage,
  onNavigate,
  theme,
  onToggleTheme,
  language,
  onSetLanguage,
  datasetVersion,
  onSetDatasetVersion,
}: NavigationProps) {
  const { t } = useI18n();
  const navItems = [
    { name: t("nav.home"), id: "home" },
    { name: t("nav.results"), id: "results" },
    { name: t("nav.dataset"), id: "dataset" },
    { name: t("nav.model"), id: "model" },
    { name: t("nav.contact"), id: "contact" },
  ];

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/95 shadow-sm backdrop-blur">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex cursor-pointer items-center gap-2" onClick={() => onNavigate("home")}>
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
              <Shield className="h-6 w-6 text-primary-foreground" />
            </div>
            <span className="text-xl tracking-tight text-primary">VietToxic Detector</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-8">
              {navItems.map((item) => {
                const isActive = currentPage === item.id;
                const buttonClass = `rounded-lg px-3 py-2 transition-colors ${
                  isActive ? "font-medium" : "text-foreground/80 hover:bg-accent"
                }`;
                const buttonStyle = {
                  color: isActive ? "var(--primary)" : undefined,
                  backgroundColor: isActive ? "color-mix(in srgb, var(--primary) 14%, transparent)" : "transparent",
                };

                if (item.id === "dataset") {
                  return (
                    <HoverCard key={item.id} openDelay={120} closeDelay={100}>
                      <HoverCardTrigger asChild>
                        <button
                          type="button"
                          onClick={() => onNavigate(item.id)}
                          className={buttonClass}
                          style={buttonStyle}
                        >
                          {item.name}
                        </button>
                      </HoverCardTrigger>
                      <HoverCardContent align="start" className="w-56 p-2">
                        <p className="px-2 py-1 text-xs text-muted-foreground">{t("nav.datasetVersionLabel")}</p>
                        <button
                          type="button"
                          onClick={() => {
                            onSetDatasetVersion("v1");
                            onNavigate("dataset");
                          }}
                          className={`w-full rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
                            datasetVersion === "v1" ? "bg-accent" : "hover:bg-accent"
                          }`}
                        >
                          {t("nav.datasetVersionV1Deprecated")}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            onSetDatasetVersion("latest");
                            onNavigate("dataset");
                          }}
                          className={`w-full rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
                            datasetVersion === "latest" ? "bg-accent" : "hover:bg-accent"
                          }`}
                        >
                          {t("nav.datasetVersionLatest")}
                        </button>
                      </HoverCardContent>
                    </HoverCard>
                  );
                }

                return (
                  <button
                    key={item.id}
                    onClick={() => onNavigate(item.id)}
                    className={buttonClass}
                    style={buttonStyle}
                  >
                    {item.name}
                  </button>
                );
              })}
            </div>

            <div className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/70 p-1" aria-label={t("nav.language")}>
              <button
                type="button"
                onClick={() => onSetLanguage("vi")}
                className={`rounded-full px-2 py-1 text-xs transition-colors ${
                  language === "vi" ? "bg-primary text-primary-foreground" : "text-foreground/80 hover:bg-accent"
                }`}
              >
                VN
              </button>
              <button
                type="button"
                onClick={() => onSetLanguage("en")}
                className={`rounded-full px-2 py-1 text-xs transition-colors ${
                  language === "en" ? "bg-primary text-primary-foreground" : "text-foreground/80 hover:bg-accent"
                }`}
              >
                EN
              </button>
            </div>

            <button
              type="button"
              onClick={onToggleTheme}
              aria-label={theme === "dark" ? t("nav.themeToLight") : t("nav.themeToDark")}
              title={theme === "dark" ? t("nav.lightMode") : t("nav.darkMode")}
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
