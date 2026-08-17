import { useMemo, useState } from "react";
import {
  BarChart3,
  CalendarClock,
  ChevronRight,
  Cpu,
  Database,
  FlaskConical,
  Home,
  Mail,
  Menu,
  Moon,
  ScrollText,
  Settings,
  Shield,
  Sparkles,
  Sun,
  MessagesSquare,
  X,
} from "lucide-react";
import type { Language } from "@/app/i18n/messages";
import { useI18n } from "@/app/i18n/context";

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  language: Language;
  onSetLanguage: (language: Language) => void;
  adminAuthenticated: boolean;
  adminUsername?: string;
  onAdminLogout: () => void;
}

export function Navigation({
  currentPage,
  onNavigate,
  theme,
  onToggleTheme,
  language,
  onSetLanguage,
  adminAuthenticated,
  adminUsername,
  onAdminLogout,
}: NavigationProps) {
  const { t } = useI18n();
  const [mobileOpen, setMobileOpen] = useState(false);

  const mainNavItems = [
    { name: t("nav.home"), id: "home", icon: Home },
    { name: t("nav.results"), id: "results", icon: BarChart3 },
    { name: t("nav.model"), id: "model", icon: Cpu },
    { name: t("nav.technicalQa"), id: "technical_qa", icon: MessagesSquare },
    { name: t("nav.contact"), id: "contact", icon: Mail },
  ];

  const datasetNavItems = [
    { name: t("nav.datasetOverview"), id: "dataset", icon: Database },
    ...(adminAuthenticated
      ? [{ name: t("nav.datasetSynthetic"), id: "dataset_synthetic", icon: Sparkles }]
      : []),
  ];

  const adminNavItems = [
    {
      name: adminAuthenticated ? t("nav.adminMlflow") : t("nav.adminLogin"),
      id: adminAuthenticated ? "admin_mlflow" : "admin_login",
      icon: FlaskConical,
    },
    ...(adminAuthenticated
      ? [
          {
            name: t("nav.adminSystemSettings"),
            id: "admin_system_settings",
            icon: Settings,
          },
          {
            name: t("nav.adminScheduledTasks"),
            id: "admin_scheduled_tasks",
            icon: CalendarClock,
          },
        ]
      : []),
  ];

  const pageTitle = useMemo(() => {
    const allItems = [...mainNavItems, ...datasetNavItems, ...adminNavItems];
    if (currentPage === "admin_login") return t("nav.adminLogin");
    return allItems.find((item) => item.id === currentPage || (item.id === "admin_mlflow" && currentPage === "mlflow"))?.name ?? t("nav.home");
  }, [adminNavItems, currentPage, datasetNavItems, mainNavItems, t]);

  const navigate = (page: string) => {
    onNavigate(page);
    setMobileOpen(false);
  };

  const renderNavButton = (item: { name: string; id: string; icon: typeof Home }, nested = false) => {
    const Icon = item.icon;
    const isActive = currentPage === item.id || (item.id === "admin_mlflow" && currentPage === "mlflow");
    return (
      <button
        key={item.id}
        type="button"
        onClick={() => navigate(item.id)}
        className={`dashboard-nav-item ${nested ? "dashboard-nav-item-nested" : ""} ${isActive ? "is-active" : ""}`}
        aria-current={isActive ? "page" : undefined}
      >
        <Icon className="h-4 w-4" />
        <span>{item.name}</span>
        {isActive && <ChevronRight className="ml-auto h-4 w-4" />}
      </button>
    );
  };

  const controls = (
    <>
      <div className="dashboard-language" aria-label={t("nav.language")}>
        <button
          type="button"
          onClick={() => onSetLanguage("vi")}
          className={language === "vi" ? "is-active" : ""}
        >
          VN
        </button>
        <button
          type="button"
          onClick={() => onSetLanguage("en")}
          className={language === "en" ? "is-active" : ""}
        >
          EN
        </button>
      </div>

      <button
        type="button"
        onClick={onToggleTheme}
        aria-label={theme === "dark" ? t("nav.themeToLight") : t("nav.themeToDark")}
        title={theme === "dark" ? t("nav.lightMode") : t("nav.darkMode")}
        className="dashboard-icon-button"
      >
        {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </button>

      {adminAuthenticated && (
        <button
          type="button"
          onClick={onAdminLogout}
          className="dashboard-nav-item dashboard-nav-item-nested"
          title={adminUsername || "Admin"}
        >
          <span>{t("nav.adminLogout")}</span>
        </button>
      )}
    </>
  );

  const sidebarContent = (
    <>
      <button type="button" className="dashboard-brand" onClick={() => navigate("home")}>
        <span className="dashboard-brand-mark">
          <Shield className="h-5 w-5" />
        </span>
        <span>
          <span className="dashboard-brand-title">VietComment Analyzer</span>
          <span className="dashboard-brand-subtitle">Detector</span>
        </span>
      </button>

      <div className="dashboard-sidebar-section">
        <p>{t("nav.mainSection")}</p>
        {mainNavItems.map((item) => renderNavButton(item))}
      </div>

      <div className="dashboard-sidebar-section">
        <p>{t("nav.datasetSection")}</p>
        {datasetNavItems.map((item) => renderNavButton(item, true))}
      </div>

      <div className="dashboard-sidebar-section">
        <p>{t("nav.adminSection")}</p>
        {adminNavItems.map((item) => renderNavButton(item))}
      </div>
    </>
  );

  return (
    <>
      <aside className="dashboard-sidebar">{sidebarContent}</aside>

      <header className="dashboard-topbar">
        <div className="dashboard-topbar-title">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="dashboard-icon-button lg:hidden"
            aria-label={t("nav.openMenu")}
          >
            <Menu className="h-4 w-4" />
          </button>
          <ScrollText className="hidden h-5 w-5 text-primary sm:block" />
          <div>
            <p>{t("nav.workspaceLabel")}</p>
            <h1>{pageTitle}</h1>
          </div>
        </div>
        <div className="dashboard-topbar-actions">{controls}</div>
      </header>

      {mobileOpen && (
        <div className="dashboard-mobile-layer lg:hidden">
          <button className="dashboard-mobile-backdrop" type="button" onClick={() => setMobileOpen(false)} aria-label={t("nav.closeMenu")} />
          <div className="dashboard-mobile-panel">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-sm font-semibold text-foreground">{t("nav.menu")}</span>
              <button type="button" onClick={() => setMobileOpen(false)} className="dashboard-icon-button" aria-label={t("nav.closeMenu")}>
                <X className="h-4 w-4" />
              </button>
            </div>
            {sidebarContent}
            <div className="mt-5 flex items-center gap-2">{controls}</div>
          </div>
        </div>
      )}
    </>
  );
}
