<template>
  <div class="dynamic-config-form">
    <div v-for="(value, key) in formData" :key="key" class="config-section">
      <h4 class="section-title">{{ formatLabel(key) }}</h4>

      <!-- Handle nested objects (like parameters) -->
      <template v-if="isObject(value) && !hasOptions(value)">
        <div class="nested-section">
          <div
            v-for="(nestedValue, nestedKey) in value"
            :key="nestedKey"
            class="form-field"
          >
            <label class="field-label">
              {{ formatLabel(nestedKey) }}
              <span v-if="nestedValue.description" class="field-description">
                - {{ nestedValue.description }}
              </span>
            </label>
            <input
              v-model="value[nestedKey].value"
              :type="getInputType(nestedValue.value)"
              :step="getInputStep(nestedValue.value)"
              class="field-input"
              @input="validateField(nestedKey, nestedValue.value)"
            />
          </div>
        </div>
      </template>

      <!-- Handle fields with options (dropdowns) -->
      <template v-else-if="hasOptions(value)">
        <div class="form-field">
          <label class="field-label">
            Select Option
            <span v-if="value.description" class="field-description">
              - {{ value.description }}
            </span>
          </label>

          <!-- Clickable option cards (dropdown removed) -->
          <div class="options-list cards-select">
            <div class="current-selection" aria-live="polite">
              Current: <strong>{{ getActiveLabel(value) }}</strong>
            </div>
            <div class="cards-grid">
              <div
                v-for="option in value.options"
                :key="option.value"
                class="option-item"
                :class="{ active: option.value === value.value }"
                role="button"
                tabindex="0"
                @click="selectOption(value, option.value)"
                @keydown.enter.prevent="selectOption(value, option.value)"
                @keydown.space.prevent="selectOption(value, option.value)"
                :aria-pressed="option.value === value.value"
              >
                <div class="option-header">
                  <span class="option-label">{{ option.label }}</span>
                  <code class="option-value">{{ option.value }}</code>
                </div>
                <p v-if="option.description" class="option-description">
                  {{ option.description }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";

interface Props {
  modelValue: Record<string, any>;
}

interface Emits {
  (e: "update:modelValue", value: Record<string, any>): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const formData = ref({ ...props.modelValue });

// Get active option label for a field with options
const getActiveLabel = (field: any): string => {
  if (!field || !Array.isArray(field.options)) return "";
  const active = field.options.find((o: any) => o.value === field.value);
  return active ? active.label : String(field.value ?? "");
};

// Select an option for a given field
const selectOption = (field: any, optionValue: any) => {
  if (!field) return;
  field.value = optionValue;
};

// Format label from camelCase/snake_case to readable text
const formatLabel = (key: string | number): string => {
  const keyStr = String(key);
  return keyStr
    .replace(/([A-Z])/g, " $1")
    .replace(/_/g, " ")
    .replace(/^./, (str) => str.toUpperCase())
    .trim();
};

// Check if value is an object
const isObject = (value: any): boolean => {
  return typeof value === "object" && value !== null && !Array.isArray(value);
};

// Check if an object has options array (indicating it's a dropdown field)
const hasOptions = (value: any): boolean => {
  return isObject(value) && Array.isArray(value.options);
};

// Get appropriate input type based on value
const getInputType = (value: any): string => {
  if (typeof value === "number") {
    return Number.isInteger(value) ? "number" : "number";
  }
  return "text";
};

// Get step for number inputs
const getInputStep = (value: any): string | undefined => {
  if (typeof value === "number" && !Number.isInteger(value)) {
    return "0.1";
  }
  return undefined;
};

// Basic field validation
const validateField = (key: string | number, value: any) => {
  // Add custom validation logic here if needed
  console.log(`Validating ${key}:`, value);
};

// Watch for changes and emit updates
watch(
  formData,
  (newValue) => {
    emit("update:modelValue", newValue);
  },
  { deep: true }
);

// Watch for external prop changes
watch(
  () => props.modelValue,
  (newValue) => {
    formData.value = { ...newValue };
  },
  { deep: true }
);
</script>

<style scoped>
.dynamic-config-form {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  padding-bottom: var(--spacing-xl);
}

.config-section {
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-lg);
  padding: var(--spacing-lg);
  background: var(--color-surface);
}

.section-title {
  margin: 0 0 var(--spacing-md) 0;
  color: var(--color-text-primary);
  font-size: 1.125rem;
  font-weight: 600;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: var(--spacing-sm);
}

.nested-section {
  display: grid;
  gap: var(--spacing-md);
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.field-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-primary);
  line-height: 1.4;
}

.field-description {
  color: var(--color-text-muted);
  font-weight: 400;
  font-style: italic;
}

.field-input {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  font-size: 0.875rem;
  background: var(--color-background);
  color: var(--color-text-primary);
  transition: border-color 0.2s ease;
}

.field-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.field-select {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  font-size: 0.875rem;
  background: var(--color-background);
  color: var(--color-text-primary);
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.field-select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.options-list {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--color-background);
  border-radius: var(--border-radius);
  border: 1px solid var(--color-border);
}

/* Card selection variant */
.cards-select {
  padding: var(--spacing-md) var(--spacing-md) var(--spacing-sm);
}

.current-selection {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-secondary);
  margin-bottom: var(--spacing-sm);
}

.cards-grid {
  display: grid;
  gap: var(--spacing-sm);
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.options-title {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.option-item {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius-md, var(--border-radius));
  transition: all 0.18s ease;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  cursor: pointer;
  position: relative;
  outline: none;
}

.option-item:focus-visible {
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.35);
  border-color: var(--color-primary);
}

.option-item.active {
  border-color: var(--color-primary);
  background: linear-gradient(rgba(37, 99, 235, 0.08), rgba(37, 99, 235, 0.08)),
    var(--color-surface);
}

.option-item:hover {
  background: var(--color-surface-hover, var(--color-background));
}

.option-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
}

.option-label {
  font-weight: 500;
  color: var(--color-text-primary);
}

.option-value {
  font-family: "Menlo", "Monaco", "Consolas", monospace;
  font-size: 0.75rem;
  background: var(--color-surface);
  padding: 2px 6px;
  border-radius: 4px;
  color: var(--color-text-secondary);
}

.option-description {
  margin: var(--spacing-xs) 0 0 0;
  font-size: 0.8125rem;
  color: var(--color-text-muted);
  line-height: 1.4;
}

@media (max-width: 768px) {
  .nested-section {
    grid-template-columns: 1fr;
  }

  .option-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-xs);
  }
}
</style>
