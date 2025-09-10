import { ref, reactive, readonly } from "vue";

export type Theme = "light" | "dark";

export interface ThemeState {
  current: Theme;
  isDark: boolean;
}

class ThemeService {
  private state = reactive<ThemeState>({
    current: "light",
    isDark: false,
  });
  private listeners = new Set<() => void>();
  private initialized = false;

  constructor() {
    // Defer heavy work until explicitly initialized to avoid duplicate init with main.ts
  }

  private initializeTheme() {
    const savedTheme = localStorage.getItem("theme") as Theme | null;
    const systemPrefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;
    const themeToUse: Theme = savedTheme
      ? savedTheme
      : systemPrefersDark
      ? "dark"
      : "light";
    // Ensure DOM attribute reflects chosen theme immediately
    document.documentElement.setAttribute("data-theme", themeToUse);
    this.updateState(themeToUse);
  }

  private setupMutationObserver() {
    // Watch for changes to data-theme attribute
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.type === "attributes" &&
          mutation.attributeName === "data-theme"
        ) {
          const theme = document.documentElement.getAttribute(
            "data-theme"
          ) as Theme;
          if (theme && theme !== this.state.current) {
            this.updateState(theme);
            this.notifyListeners();
          }
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  /** Public initializer (idempotent) */
  public init() {
    if (this.initialized) return;
    this.initializeTheme();
    this.setupMutationObserver();
    this.initialized = true;
  }

  private updateState(theme: Theme) {
    this.state.current = theme;
    this.state.isDark = theme === "dark";
  }

  private notifyListeners() {
    this.listeners.forEach((listener) => {
      try {
        listener();
      } catch (error) {
        console.error("Error in theme change listener:", error);
      }
    });
  }

  public getCurrentTheme(): Theme {
    return this.state.current;
  }

  public isDarkMode(): boolean {
    return this.state.isDark;
  }

  public getState(): ThemeState {
    return this.state;
  }

  public onThemeChange(callback: () => void): () => void {
    this.listeners.add(callback);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(callback);
    };
  }

  public setTheme(theme: Theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    this.updateState(theme);
    this.notifyListeners();
  }

  public toggleTheme() {
    const newTheme = this.state.isDark ? "light" : "dark";
    this.setTheme(newTheme);
  }
}

// Create singleton instance
export const themeService = new ThemeService();

// Vue composable for reactive theme access
export function useTheme() {
  const theme = ref<Theme>(themeService.getCurrentTheme());
  const isDark = ref<boolean>(themeService.isDarkMode());

  const unsubscribe = themeService.onThemeChange(() => {
    theme.value = themeService.getCurrentTheme();
    isDark.value = themeService.isDarkMode();
  });

  return {
    theme: readonly(theme),
    isDark: readonly(isDark),
    setTheme: themeService.setTheme.bind(themeService),
    toggleTheme: themeService.toggleTheme.bind(themeService),
    unsubscribe,
  };
}
