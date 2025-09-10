<template>
  <button
    :type="type"
    :disabled="disabled"
    class="btn"
    :class="[variant, size, { loading: loading }]"
    @click="handleClick"
  >
    <span v-if="loading" class="loading-spinner"></span>
    <slot v-if="!loading"></slot>
    <span v-if="loading">{{ loadingText }}</span>
  </button>
</template>

<script setup lang="ts">
interface Props {
  variant?: "primary" | "secondary" | "success" | "warning" | "error";
  size?: "sm" | "md" | "lg";
  type?: "button" | "submit" | "reset";
  disabled?: boolean;
  loading?: boolean;
  loadingText?: string;
}

interface Emits {
  (e: "click", event: MouseEvent): void;
}

const props = withDefaults(defineProps<Props>(), {
  variant: "primary",
  size: "md",
  type: "button",
  disabled: false,
  loading: false,
  loadingText: "Loading...",
});

const emit = defineEmits<Emits>();

const handleClick = (event: MouseEvent) => {
  if (!props.disabled && !props.loading) {
    emit("click", event);
  }
};
</script>

<style scoped>
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  border: none;
  border-radius: var(--border-radius);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  white-space: nowrap;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn.loading {
  cursor: wait;
}

/* Variants */
.btn.primary {
  background-color: var(--color-primary);
  color: white;
}

.btn.primary:hover:not(:disabled) {
  background-color: var(--color-primary-hover);
}

.btn.secondary {
  background-color: var(--color-surface);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
}

.btn.secondary:hover:not(:disabled) {
  background-color: var(--color-surface-hover);
  border-color: var(--color-border-hover);
}

.btn.success {
  background-color: var(--color-success);
  color: white;
}

.btn.success:hover:not(:disabled) {
  background-color: #15803d;
}

.btn.warning {
  background-color: var(--color-warning);
  color: white;
}

.btn.warning:hover:not(:disabled) {
  background-color: #c2410c;
}

.btn.error {
  background-color: var(--color-error);
  color: white;
}

.btn.error:hover:not(:disabled) {
  background-color: #b91c1c;
}

/* Sizes */
.btn.sm {
  padding: calc(var(--spacing-xs)) var(--spacing-sm);
  font-size: 0.8rem;
}

.btn.md {
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: 0.875rem;
}

.btn.lg {
  padding: var(--spacing-md) var(--spacing-lg);
  font-size: 1rem;
}

/* Loading spinner */
.loading-spinner {
  width: 1rem;
  height: 1rem;
  border: 2px solid transparent;
  border-top: 2px solid currentColor;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
</style>
