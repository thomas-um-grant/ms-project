<template>
  <div v-if="isOpen" class="modal-overlay" @click="handleOverlayClick">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>Advanced Configuration</h3>
        <button class="close-button" @click="closeModal">
          <svg
            class="close-icon"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      <div class="modal-body">
        <div class="config-editor">
          <div class="editor-tabs">
            <button
              class="tab-button"
              :class="{ active: activeTab === 'form' }"
              @click="activeTab = 'form'"
            >
              <svg
                class="tab-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              Form View
            </button>
            <button
              class="tab-button"
              :class="{ active: activeTab === 'json' }"
              @click="activeTab = 'json'"
            >
              <svg
                class="tab-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                />
              </svg>
              JSON View
            </button>
          </div>

          <!-- Form View -->
          <div v-if="activeTab === 'form'" class="form-view">
            <DynamicConfigForm
              v-model="formConfig"
              @update:modelValue="handleFormUpdate"
            />
          </div>

          <!-- JSON View -->
          <div v-else class="json-view">
            <label class="editor-label">Configuration JSON:</label>
            <textarea
              v-model="jsonText"
              class="json-editor"
              :class="{ error: hasError }"
              rows="12"
              spellcheck="false"
              @input="validateJson"
            />
            <div v-if="hasError" class="error-message">
              {{ errorMessage }}
            </div>
            <div
              v-if="!hasError && jsonText !== initialJson"
              class="info-message"
            >
              Configuration modified - click Save to apply changes
            </div>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <Button
          variant="secondary"
          @click="resetToDefaults"
          class="reset-button"
        >
          Reset to Defaults
        </Button>
        <div class="action-buttons">
          <Button variant="secondary" @click="closeModal">Cancel</Button>
          <Button
            variant="primary"
            @click="saveConfiguration"
            :disabled="hasError || !hasChanges"
          >
            Save
          </Button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from "vue";
import Button from "./Button.vue";
import DynamicConfigForm from "./DynamicConfigForm.vue";
import {
  EMBEDDING_MODEL_OPTIONS,
  LLM_MODEL_OPTIONS,
  CHUNKING_OPTIONS,
  RETRIEVAL_OPTIONS,
  RERANKING_OPTIONS,
  DEFAULT_CONFIG,
} from "../../constants/configuration";

export interface DetailedConfig {
  embeddingModel: string;
  llmModel: string;
  chunkingStrategy: string;
  retrievalStrategy: string;
  rerankingStrategy: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  temperature: number;
}

interface Props {
  isOpen: boolean;
  config: DetailedConfig;
}

interface Emits {
  (e: "close"): void;
  (e: "save", config: DetailedConfig): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const jsonText = ref("");
const hasError = ref(false);
const errorMessage = ref("");
const initialJson = ref("");

// Tab management for form/json view
const activeTab = ref<"form" | "json">("form");
const formConfig = ref<Record<string, any>>({});

// Convert configuration to a nicely formatted JSON with comments
const formatConfigJson = (config: DetailedConfig): string => {
  const configWithOptions = {
    embeddingModel: {
      value: config.embeddingModel,
      options: EMBEDDING_MODEL_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    },
    llmModel: {
      value: config.llmModel,
      options: LLM_MODEL_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    },
    chunkingStrategy: {
      value: config.chunkingStrategy,
      options: CHUNKING_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    },
    retrievalStrategy: {
      value: config.retrievalStrategy,
      options: RETRIEVAL_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    },
    rerankingStrategy: {
      value: config.rerankingStrategy,
      options: RERANKING_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    },
    parameters: {
      chunkSize: {
        value: config.chunkSize,
        description: "Size of text chunks (100-2000)",
      },
      chunkOverlap: {
        value: config.chunkOverlap,
        description: "Overlap between chunks (0-500)",
      },
      topK: {
        value: config.topK,
        description: "Number of documents to retrieve (1-20)",
      },
      temperature: {
        value: config.temperature,
        description: "Generation temperature (0-2)",
      },
    },
  };

  return JSON.stringify(configWithOptions, null, 2);
};

// Parse JSON back to configuration
const parseConfigJson = (jsonStr: string): DetailedConfig | null => {
  try {
    const parsed = JSON.parse(jsonStr);

    return {
      embeddingModel:
        parsed.embeddingModel?.value || DEFAULT_CONFIG.selectedEmbeddingModel,
      llmModel: parsed.llmModel?.value || DEFAULT_CONFIG.selectedLLMModel,
      chunkingStrategy:
        parsed.chunkingStrategy?.value || DEFAULT_CONFIG.chunkingStrategy,
      retrievalStrategy:
        parsed.retrievalStrategy?.value || DEFAULT_CONFIG.retrievalStrategy,
      rerankingStrategy:
        parsed.rerankingStrategy?.value || DEFAULT_CONFIG.rerankingStrategy,
      chunkSize:
        parsed.parameters?.chunkSize?.value || DEFAULT_CONFIG.chunkSize,
      chunkOverlap:
        parsed.parameters?.chunkOverlap?.value || DEFAULT_CONFIG.chunkOverlap,
      topK: parsed.parameters?.topK?.value || DEFAULT_CONFIG.topK,
      temperature:
        parsed.parameters?.temperature?.value || DEFAULT_CONFIG.temperature,
    };
  } catch {
    return null;
  }
};

const validateJson = () => {
  try {
    const parsed = parseConfigJson(jsonText.value);
    if (parsed === null) {
      hasError.value = true;
      errorMessage.value = "Invalid JSON format";
    } else {
      hasError.value = false;
      errorMessage.value = "";
    }
  } catch (error) {
    hasError.value = true;
    errorMessage.value = "Invalid JSON format";
  }
};

// Handle form updates
const handleFormUpdate = (newFormData: Record<string, any>) => {
  formConfig.value = newFormData;
  // Sync JSON text with form data
  jsonText.value = JSON.stringify(newFormData, null, 2);
  validateJson();
};

const hasChanges = computed(() => {
  return jsonText.value !== initialJson.value && !hasError.value;
});

const resetToDefaults = () => {
  const defaultConfig: DetailedConfig = {
    embeddingModel: DEFAULT_CONFIG.selectedEmbeddingModel,
    llmModel: DEFAULT_CONFIG.selectedLLMModel,
    chunkingStrategy: "fixed",
    retrievalStrategy: "dense",
    rerankingStrategy: "cross_encoder",
    chunkSize: DEFAULT_CONFIG.chunkSize,
    chunkOverlap: DEFAULT_CONFIG.chunkOverlap,
    topK: DEFAULT_CONFIG.topK,
    temperature: DEFAULT_CONFIG.temperature,
  };

  jsonText.value = formatConfigJson(defaultConfig);
  validateJson();
};

const saveConfiguration = () => {
  if (hasError.value) return;

  const config = parseConfigJson(jsonText.value);
  if (config) {
    emit("save", config);
    closeModal();
  }
};

const closeModal = () => {
  emit("close");
};

const handleOverlayClick = () => {
  closeModal();
};

// Watch for prop changes
watch(
  () => props.config,
  (newConfig) => {
    if (newConfig && props.isOpen) {
      const formattedJson = formatConfigJson(newConfig);
      jsonText.value = formattedJson;
      initialJson.value = formattedJson;
      formConfig.value = JSON.parse(formattedJson);
      validateJson();
    }
  },
  { deep: true, immediate: true }
);

// Reset when modal opens
watch(
  () => props.isOpen,
  (isOpen) => {
    if (isOpen) {
      const formattedJson = formatConfigJson(props.config);
      jsonText.value = formattedJson;
      initialJson.value = formattedJson;
      formConfig.value = JSON.parse(formattedJson);
      validateJson();
    }
  }
);
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: var(--spacing-lg);
}

.modal-content {
  background: var(--color-background);
  border-radius: var(--border-radius-lg);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  max-width: 800px;
  max-height: 90vh;
  width: 100%;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-lg) var(--spacing-lg) 0;
  margin-bottom: var(--spacing-md);
}

.modal-header h3 {
  margin: 0;
  color: var(--color-text-primary);
  font-size: 1.25rem;
  font-weight: 600;
}

.close-button {
  background: none;
  border: none;
  cursor: pointer;
  padding: var(--spacing-xs);
  border-radius: var(--border-radius);
  color: var(--color-text-secondary);
  transition: all 0.2s ease;
}

.close-button:hover {
  background-color: var(--color-surface);
  color: var(--color-text-primary);
}

.close-icon {
  width: 1.25rem;
  height: 1.25rem;
}

.modal-body {
  flex: 1;
  padding: 0 var(--spacing-lg);
  overflow: hidden;
}

.config-editor {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.editor-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-primary);
  margin-bottom: var(--spacing-sm);
}

.json-editor {
  flex: 1;
  font-family: "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New",
    monospace;
  font-size: 0.875rem;
  line-height: 1.5;
  padding: var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  background-color: var(--color-surface);
  color: var(--color-text-primary);
  resize: none;
  outline: none;
  transition: border-color 0.2s ease;
  min-height: 300px;
  max-height: 400px;
}

.json-editor:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.json-editor.error {
  border-color: var(--color-error);
}

.error-message {
  color: var(--color-error);
  font-size: 0.875rem;
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm);
  background-color: rgba(239, 68, 68, 0.1);
  border-radius: var(--border-radius);
}

.info-message {
  color: var(--color-primary);
  font-size: 0.875rem;
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm);
  background-color: rgba(37, 99, 235, 0.1);
  border-radius: var(--border-radius);
}

.modal-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-lg);
  border-top: 1px solid var(--color-border);
  margin-top: var(--spacing-md);
}

.action-buttons {
  display: flex;
  gap: var(--spacing-md);
}

.reset-button {
  color: var(--color-error);
}

.reset-button:hover {
  background-color: rgba(239, 68, 68, 0.1);
}

@media (max-width: 768px) {
  .modal-overlay {
    padding: var(--spacing-md);
  }

  .modal-content {
    max-height: 95vh;
  }

  .modal-footer {
    flex-direction: column;
    gap: var(--spacing-md);
    align-items: stretch;
  }

  .action-buttons {
    justify-content: stretch;
  }

  .action-buttons button {
    flex: 1;
  }
}

/* Tab styles */
.editor-tabs {
  display: flex;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
}

.tab-button {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  font-weight: 500;
  border-radius: var(--border-radius) var(--border-radius) 0 0;
  cursor: pointer;
  transition: all 0.2s ease;
  border-bottom: 2px solid transparent;
}

.tab-button:hover {
  background: var(--color-surface);
  color: var(--color-text-primary);
}

.tab-button.active {
  background: var(--color-surface);
  color: var(--color-primary);
  border-bottom-color: var(--color-primary);
}

.tab-icon {
  width: 1rem;
  height: 1rem;
}

.form-view {
  max-height: 400px;
  overflow-y: auto;
  padding: var(--spacing-sm) 0 var(--spacing-xl) 0;
}

.json-view {
  display: flex;
  flex-direction: column;
}
</style>
