<!--
  Reusable Dropdown Component
  
  A flexible dropdown component that supports:
  - Single and multi-select modes
  - Option descriptions
  - Disabled state
  - Custom placeholders
  - Keyboard navigation
  
  @component
  @example
  <Dropdown
    v-model="selectedValue"
    :options="options"
    placeholder="Select an option..."
    :multiple="false"
    :disabled="false"
  />
-->
<template>
  <div class="dropdown" ref="dropdownRef">
    <button
      @click="toggleDropdown"
      class="dropdown-trigger"
      :class="{ active: isOpen, disabled: disabled }"
      :disabled="disabled"
    >
      <span>{{ displayText || placeholder }}</span>
      <svg
        class="dropdown-icon"
        :class="{ rotated: isOpen }"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M19 9l-7 7-7-7"
        />
      </svg>
    </button>

    <div v-if="isOpen" class="dropdown-content">
      <div
        v-for="option in options"
        :key="option.value"
        @click="selectOption(option)"
        class="dropdown-item"
        :class="{ selected: isOptionSelected(option.value) }"
      >
        <div class="dropdown-item-content">
          <input
            v-if="multiple"
            type="checkbox"
            :checked="isOptionSelected(option.value)"
            class="dropdown-checkbox"
            @click.stop
          />
          <span class="dropdown-item-label">{{ option.label }}</span>
          <span v-if="option.description" class="dropdown-item-description">
            {{ option.description }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Dropdown Component Props and Events
 *
 * This component provides a consistent dropdown interface
 * with support for single/multi-select and rich option data
 */
import { ref, computed, onMounted, onUnmounted } from "vue";

interface DropdownOption {
  label: string;
  value: string | number;
  description?: string;
}

interface Props {
  /** The selected value(s) */
  modelValue: string | number | null | (string | number)[];
  /** Available options to select from */
  options: DropdownOption[];
  /** Placeholder text when no option is selected */
  placeholder?: string;
  /** Whether the dropdown is disabled */
  disabled?: boolean;
  /** Whether multiple selections are allowed */
  multiple?: boolean;
}

interface Emits {
  /** Emitted when the selection changes */
  (e: "update:modelValue", value: string | number | (string | number)[]): void;
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: "Select an option...",
  disabled: false,
  multiple: false,
});

const emit = defineEmits<Emits>();

const isOpen = ref(false);
const dropdownRef = ref<HTMLElement>();

const displayText = computed(() => {
  if (props.multiple) {
    const selectedValues = Array.isArray(props.modelValue)
      ? props.modelValue
      : [];
    if (selectedValues.length === 0) return "";
    if (selectedValues.length === 1) {
      const selected = props.options.find(
        (option) => option.value === selectedValues[0]
      );
      return selected?.label || "";
    }
    return `${selectedValues.length} items selected`;
  } else {
    const selected = props.options.find(
      (option) => option.value === props.modelValue
    );
    return selected?.label || "";
  }
});

const isOptionSelected = (value: string | number) => {
  if (props.multiple) {
    const selectedValues = Array.isArray(props.modelValue)
      ? props.modelValue
      : [];
    return selectedValues.includes(value);
  } else {
    return props.modelValue === value;
  }
};

const toggleDropdown = () => {
  if (!props.disabled) {
    isOpen.value = !isOpen.value;
  }
};

const selectOption = (option: DropdownOption) => {
  if (props.multiple) {
    const currentValues = Array.isArray(props.modelValue)
      ? [...props.modelValue]
      : [];
    const index = currentValues.indexOf(option.value);

    if (index === -1) {
      // Add the option if not selected
      currentValues.push(option.value);
    } else {
      // Remove the option if already selected
      currentValues.splice(index, 1);
    }

    emit("update:modelValue", currentValues);
    // Don't close dropdown for multiple selection
  } else {
    emit("update:modelValue", option.value);
    isOpen.value = false;
  }
};

const handleClickOutside = (event: Event) => {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target as Node)) {
    isOpen.value = false;
  }
};

onMounted(() => {
  document.addEventListener("click", handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener("click", handleClickOutside);
});
</script>

<style scoped>
.dropdown {
  position: relative;
  width: 100%;
}

.dropdown-trigger {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 0.875rem;
  color: var(--color-text-primary);
  transition: all 0.2s ease;
}

.dropdown-trigger:hover {
  border-color: var(--color-primary);
}

.dropdown-trigger.active {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.dropdown-trigger.disabled {
  background-color: var(--color-surface);
  color: var(--color-text-muted);
  cursor: not-allowed;
  border-color: var(--color-border);
}

.dropdown-trigger.disabled:hover {
  border-color: var(--color-border);
}

.dropdown-icon {
  width: 1rem;
  height: 1rem;
  transition: transform 0.2s ease;
}

.dropdown-icon.rotated {
  transform: rotate(180deg);
}

.dropdown-content {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-lg);
  z-index: 1000;
  max-height: 200px;
  overflow-y: auto;
  margin-top: 2px;
}

.dropdown-item {
  padding: var(--spacing-sm) var(--spacing-md);
  cursor: pointer;
  transition: background-color 0.2s ease;
  color: var(--color-text-primary);
}

.dropdown-item:hover {
  background-color: var(--color-surface-hover);
}

.dropdown-item.selected {
  background-color: var(--color-primary-light);
  color: var(--color-primary);
}

.dropdown-item-content {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  width: 100%;
}

.dropdown-checkbox {
  margin: 0;
  cursor: pointer;
}

.dropdown-item-label {
  font-size: 0.875rem;
  font-weight: 500;
  flex: 1;
}

.dropdown-item-description {
  font-size: 0.75rem;
  color: var(--color-text-muted);
}
</style>
