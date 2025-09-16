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
          <!-- Form View -->
          <div class="form-view">
            <DynamicConfigForm
              v-model="formConfig"
              @update:modelValue="handleFormUpdate"
            />
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
            :disabled="!hasChanges"
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
import type { PipelineType } from "../../types/retriever";
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
  pipelineType: PipelineType;
}

interface Emits {
  (e: "close"): void;
  (e: "save", config: DetailedConfig): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const formConfig = ref<Record<string, any>>({});

// Convert configuration to a JSON-like object used to drive the form.
// This function also applies UI constraints based on the selected pipeline.
const formatConfigJson = (config: DetailedConfig): string => {
  // LLM model remains configurable
  const formShape: Record<string, any> = {
    llmModel: {
      value: config.llmModel,
      options: LLM_MODEL_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    },
  };

  // Hide Embedding model: induced from RAG type (do not expose in advanced UI)
  // We intentionally omit embeddingModel from the form.

  // Chunking strategy: only allow "Page" chunking
  formShape.chunkingStrategy = {
    value: "page",
    options: [
      {
        value: "page",
        label: "Page",
        description: "Split by page boundaries",
      },
    ],
  };

  // Retrieval Strategy: show only if pipeline is NOT multimodal
  if (props.pipelineType !== "multimodal") {
    formShape.retrievalStrategy = {
      value: config.retrievalStrategy,
      options: RETRIEVAL_OPTIONS.map((opt) => ({
        value: opt.value,
        label: opt.label,
        description: opt.description,
      })),
    };
  }

  // Reranking Strategy: hide LLM reranking; if MultiRAG, force Jina and make it the only option
  const rerankOptions = RERANKING_OPTIONS.filter((opt) => opt.value !== "llm");
  if (props.pipelineType === "multi") {
    formShape.rerankingStrategy = {
      value: "jina",
      options: rerankOptions.filter((o) => o.value === "jina"),
    };
  } else {
    formShape.rerankingStrategy = {
      value:
        config.rerankingStrategy === "llm" ? "none" : config.rerankingStrategy,
      options: rerankOptions,
    };
  }

  // Parameters: remove chunk size and overlap; keep topK and temperature only
  formShape.parameters = {
    topK: {
      value: config.topK,
      description: "Number of documents to retrieve (1-20)",
    },
    temperature: {
      value: config.temperature,
      description: "Generation temperature (0-2)",
    },
  };

  return JSON.stringify(formShape, null, 2);
};

// Parse JSON back to configuration
// Handle form updates
const handleFormUpdate = (newFormData: Record<string, any>) => {
  formConfig.value = newFormData;
};

const hasChanges = computed(() => {
  // Compare current form state to formatted initial config
  try {
    const initial = JSON.parse(formatConfigJson(props.config));
    return JSON.stringify(formConfig.value) !== JSON.stringify(initial);
  } catch {
    return true;
  }
});

const resetToDefaults = () => {
  const defaultConfig: DetailedConfig = {
    embeddingModel: DEFAULT_CONFIG.selectedEmbeddingModel,
    llmModel: DEFAULT_CONFIG.selectedLLMModel,
    chunkingStrategy: "page",
    retrievalStrategy: "hybrid",
    rerankingStrategy: "none",
    chunkSize: DEFAULT_CONFIG.chunkSize,
    chunkOverlap: DEFAULT_CONFIG.chunkOverlap,
    topK: DEFAULT_CONFIG.topK,
    temperature: DEFAULT_CONFIG.temperature,
  };

  formConfig.value = JSON.parse(formatConfigJson(defaultConfig));
};

const saveConfiguration = () => {
  // Convert the current formConfig back to DetailedConfig
  try {
    const parsed = formConfig.value;
    const cfg: DetailedConfig = {
      // embeddingModel hidden in form -> preserve incoming value
      embeddingModel:
        props.config.embeddingModel || DEFAULT_CONFIG.selectedEmbeddingModel,
      llmModel: parsed.llmModel?.value || DEFAULT_CONFIG.selectedLLMModel,
      chunkingStrategy: parsed.chunkingStrategy?.value || "page",
      // retrievalStrategy may be omitted for multimodal -> preserve or default
      retrievalStrategy:
        parsed.retrievalStrategy?.value ||
        props.config.retrievalStrategy ||
        DEFAULT_CONFIG.retrievalStrategy,
      rerankingStrategy:
        parsed.rerankingStrategy?.value ||
        (props.pipelineType === "multi"
          ? "jina"
          : props.config.rerankingStrategy || DEFAULT_CONFIG.rerankingStrategy),
      // Removed from form; preserve original values when saving
      chunkSize: props.config.chunkSize,
      chunkOverlap: props.config.chunkOverlap,
      topK: parsed.parameters?.topK?.value || DEFAULT_CONFIG.topK,
      temperature:
        parsed.parameters?.temperature?.value || DEFAULT_CONFIG.temperature,
    };

    emit("save", cfg);
    closeModal();
  } catch (e) {
    // If parsing fails, do nothing
    console.error("Failed to save configuration from form", e);
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
      formConfig.value = JSON.parse(formattedJson);
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
      formConfig.value = JSON.parse(formattedJson);
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

/* Removed JSON editor and related messages */

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

/* Removed editor tab styles */

.form-view {
  max-height: 400px;
  overflow-y: auto;
  padding: var(--spacing-sm) 0 var(--spacing-xl) 0;
}

/* Removed JSON view styles */
</style>
