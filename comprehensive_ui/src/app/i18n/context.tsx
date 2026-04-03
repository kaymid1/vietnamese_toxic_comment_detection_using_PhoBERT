import { createContext, useContext } from "react";
import { formatMessage, type Language, type MessageKey, resolveMessage } from "@/app/i18n/messages";

export interface I18nContextValue {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: MessageKey, params?: Record<string, string | number>) => string;
}

export const I18nContext = createContext<I18nContextValue | null>(null);

export const createTranslator = (language: Language) => {
  return (key: MessageKey, params?: Record<string, string | number>) => {
    const template = resolveMessage(language, key);
    return formatMessage(template, params);
  };
};

export const useI18n = (): I18nContextValue => {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within I18nContext provider");
  }
  return context;
};
