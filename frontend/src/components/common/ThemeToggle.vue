<template>
  <button
    @click="toggleTheme"
    class="theme-toggle-btn"
    :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
  >
    <!-- Sun icon for light mode -->
    <svg
      v-if="isDark"
      class="theme-icon"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-width="2"
        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
      />
    </svg>

    <!-- Moon icon for dark mode -->
    <svg
      v-else
      class="theme-icon"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-width="2"
        d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
      />
    </svg>
  </button>
</template>

<script setup lang="ts">
import { useTheme } from "../../services/ThemeService";

const { isDark, toggleTheme } = useTheme();
</script>

<style scoped>
.theme-toggle-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  padding: 0;
  background: transparent;
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all var(--transition-normal);
}

.theme-toggle-btn:hover {
  background: var(--color-surface);
  border-color: var(--color-border-hover);
  color: var(--color-text-primary);
}

.theme-toggle-btn:focus {
  outline: none;
  box-shadow: 0 0 0 3px var(--color-primary-light);
}

.theme-icon {
  width: 1.25rem;
  height: 1.25rem;
  transition: all var(--transition-normal);
}

/* Dark theme specific styles */
[data-theme="dark"] .theme-toggle-btn {
  border-color: var(--color-border);
}

[data-theme="dark"] .theme-toggle-btn:hover {
  background: var(--color-surface-hover);
  color: var(--color-primary);
}
</style>
