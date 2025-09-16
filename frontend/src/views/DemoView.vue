<template>
  <div class="demo-view">
    <!-- Configuration Panel -->
    <div class="demo-config-panel panel">
      <div class="panel-header">
        <h3>Demo Configuration</h3>
      </div>

      <div class="config-grid">
        <div class="config-group">
          <label class="config-label">Retrieval System</label>
          <Dropdown
            v-model="config.selectedPipeline"
            :options="pipelineOptions"
            placeholder="Select retrieval system..."
            @update:modelValue="updatePipeline"
          />
        </div>

        <div class="config-group">
          <label class="config-label">Knowledge Base</label>
          <div class="knowledge-base-selector">
            <Dropdown
              v-model="selectedKnowledgeBase"
              :options="knowledgeBaseOptions"
              placeholder="Select knowledge base..."
              class="knowledge-base-dropdown"
            />
            <Button
              variant="secondary"
              @click="addKnowledgeBase"
              class="add-button"
              title="Add new knowledge base folder"
            >
              <svg
                class="add-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              Add
            </Button>
          </div>
        </div>

        <div class="config-group">
          <label class="config-label">Advanced Configuration</label>
          <Button
            variant="secondary"
            @click="openConfigModal"
            class="configs-button"
          >
            <svg
              class="config-icon"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
            Configs
          </Button>
        </div>
      </div>
    </div>

    <!-- Tab Navigation -->
    <div class="tab-navigation panel">
      <div class="tab-buttons">
        <button
          class="tab-button"
          :class="{ active: activeTab === 'assistant' }"
          @click="activeTab = 'assistant'"
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
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
          Assistant
        </button>
        <button
          class="tab-button"
          :class="{ active: activeTab === 'finder' }"
          @click="activeTab = 'finder'"
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
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          Finder
        </button>
      </div>
    </div>

    <!-- Main Demo Layout -->
    <div class="demo-layout">
      <!-- Assistant Tab Content -->
      <template v-if="activeTab === 'assistant'">
        <div class="assistant-layout">
          <!-- Chat Panel -->
          <div class="chat-panel panel">
            <div class="panel-header">
              <h3>Chat</h3>
              <div class="chat-actions">
                <Button
                  variant="secondary"
                  size="sm"
                  @click="clearChat"
                  :disabled="messages.length === 0"
                >
                  Clear Chat
                </Button>
              </div>
            </div>

            <div class="chat-container">
              <ChatMessages :messages="messages" :is-loading="isLoading" />
              <ChatInput :loading="isLoading" @send="handleSendMessage" />
            </div>
          </div>

          <!-- Knowledge Sources Panel -->
          <div class="knowledge-panel panel">
            <div class="panel-header">
              <h3>Retrieved Sources</h3>
              <span v-if="currentSources.length > 0" class="source-count">
                {{ currentSources.length }} document{{
                  currentSources.length !== 1 ? "s" : ""
                }}
              </span>
            </div>

            <div v-if="currentSources.length === 0" class="empty-sources">
              <div class="empty-icon">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
              </div>
              <p>Ask a question to see retrieved sources</p>
            </div>

            <div v-else class="sources-list">
              <KnowledgeSource
                v-for="(doc, index) in currentSources"
                :key="doc.id"
                :document="doc"
                :index="index"
              />
            </div>
          </div>
        </div>
      </template>

      <!-- Finder Tab Content -->
      <template v-else-if="activeTab === 'finder'">
        <div class="finder-panel panel">
          <FileFinder @file-selected="handleFileSelected" />
        </div>
      </template>
    </div>

    <!-- Configuration Modal -->
    <ConfigModal
      :is-open="showConfigModal"
      :config="detailedConfig"
      :pipeline-type="config.selectedPipeline"
      @close="closeConfigModal"
      @save="saveConfiguration"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, watch } from "vue";
import ChatMessages from "../components/demo/ChatMessages.vue";
import ChatInput from "../components/demo/ChatInput.vue";
import KnowledgeSource from "../components/demo/KnowledgeSource.vue";
import FileFinder from "../components/demo/FileFinder.vue";
import Button from "../components/common/Button.vue";
import Dropdown from "../components/common/Dropdown.vue";
import ConfigModal from "../components/common/ConfigModal.vue";
import type { DetailedConfig } from "../components/common/ConfigModal.vue";
import { apiClient } from "../services/ApiClient";
import {
  PIPELINE_OPTIONS,
  EMBEDDING_MODEL_OPTIONS,
  LLM_MODEL_OPTIONS,
  CHUNKING_OPTIONS,
  RETRIEVAL_OPTIONS,
  RERANKING_OPTIONS,
  DEFAULT_CONFIG,
} from "../constants/configuration";
import {
  resolveConfigFile,
  getDefaultConfigForPipeline,
} from "../constants/ragConfigs";
import type {
  ChatMessage,
  RetrievedDocument,
  QueryRequest,
} from "../types/api";
import type { ConfigurationState, PipelineConfig } from "../types/retriever";

// Type declaration for Electron API
declare global {
  interface Window {
    electronAPI?: {
      showOpenDialog: (
        options: any
      ) => Promise<{ canceled: boolean; filePaths: string[] }>;
    };
  }
}

// Tab management
const activeTab = ref<"assistant" | "finder">("assistant");

const messages = ref<ChatMessage[]>([]);
const currentSources = ref<RetrievedDocument[]>([]);
const isLoading = ref(false);

// Configuration state
const selectedKnowledgeBase = ref<string>("");

// Configuration modal state
const showConfigModal = ref(false);
const detailedConfig = ref<DetailedConfig>({
  embeddingModel: DEFAULT_CONFIG.selectedEmbeddingModel,
  llmModel: DEFAULT_CONFIG.selectedLLMModel,
  chunkingStrategy: DEFAULT_CONFIG.chunkingStrategy,
  retrievalStrategy: DEFAULT_CONFIG.retrievalStrategy,
  rerankingStrategy: DEFAULT_CONFIG.rerankingStrategy,
  chunkSize: DEFAULT_CONFIG.chunkSize,
  chunkOverlap: DEFAULT_CONFIG.chunkOverlap,
  topK: DEFAULT_CONFIG.topK,
  temperature: DEFAULT_CONFIG.temperature,
});

// Knowledge base options (fake folder names for demo)
const knowledgeBaseOptions = [
  {
    value: "vidore/infovqa_test_subsampled_beir",
    label: "InfoVQA",
  },
  {
    value: "beir/nfcorpus",
    label: "NF Corpus",
  },
];

const config = reactive<ConfigurationState>({
  selectedPipeline: DEFAULT_CONFIG.selectedPipeline,
  selectedEmbeddingModel: DEFAULT_CONFIG.selectedEmbeddingModel,
  selectedLLMModel: DEFAULT_CONFIG.selectedLLMModel,
  chunkSize: DEFAULT_CONFIG.chunkSize,
  chunkOverlap: DEFAULT_CONFIG.chunkOverlap,
  topK: DEFAULT_CONFIG.topK,
  temperature: DEFAULT_CONFIG.temperature,
});

// Use centralized configuration options
const pipelineOptions = PIPELINE_OPTIONS;

/**
 * Updates the pipeline configuration
 */
const updatePipeline = (value: string | number | (string | number)[]) => {
  // Handle array values (shouldn't happen for single-select, but type safety)
  const singleValue = Array.isArray(value) ? value[0] : value;
  config.selectedPipeline = singleValue as any;
};

// Helper: activate a KB on backend
const selectActiveRag = async (kb: string) => {
  if (!kb) return;
  // Resolve backend config file based on current pipeline and knowledge base
  const configFile =
    resolveConfigFile(config.selectedPipeline, kb) ||
    getDefaultConfigForPipeline(config.selectedPipeline);
  if (!configFile) {
    console.warn("No config mapping available for current selection.");
    return;
  }
  try {
    await apiClient.selectRag({ config: configFile, knowledge_base: kb });
    console.log("Active RAG set:", { configFile, kb });
  } catch (e) {
    console.error("Failed to select RAG", e);
  }
};

// Auto-activate RAG when knowledge base changes (only for non-path values)
watch(selectedKnowledgeBase, (kb) => {
  // Only treat absolute filesystem paths as paths; otherwise, auto-select on backend
  const looksLikeAbsoluteUnix = kb.startsWith("/");
  const looksLikeAbsoluteWin = /^[A-Za-z]:\\\\/.test(kb);
  const looksLikePath = looksLikeAbsoluteUnix || looksLikeAbsoluteWin;
  if (!looksLikePath) {
    selectActiveRag(kb);
  }
});

// Auto-activate RAG when pipeline (system) changes as well
watch(
  () => config.selectedPipeline,
  () => {
    const kb = selectedKnowledgeBase.value || "";
    const looksLikeAbsoluteUnix = kb.startsWith("/");
    const looksLikeAbsoluteWin = /^[A-Za-z]:\\\\/.test(kb);
    const looksLikePath = looksLikeAbsoluteUnix || looksLikeAbsoluteWin;
    if (!looksLikePath && kb) {
      selectActiveRag(kb);
    }
  }
);

/**
 * Opens the advanced configuration modal
 */
const openConfigModal = () => {
  // Sync current config with detailed config
  detailedConfig.value = {
    embeddingModel: config.selectedEmbeddingModel,
    llmModel: config.selectedLLMModel,
    chunkingStrategy: detailedConfig.value.chunkingStrategy,
    retrievalStrategy: detailedConfig.value.retrievalStrategy,
    rerankingStrategy: detailedConfig.value.rerankingStrategy,
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
    topK: config.topK,
    temperature: config.temperature,
  };
  showConfigModal.value = true;
};

/**
 * Closes the configuration modal
 */
const closeConfigModal = () => {
  showConfigModal.value = false;
};

/**
 * Saves the configuration from the modal
 */
const saveConfiguration = (newConfig: DetailedConfig) => {
  // Update main config with values from detailed config
  config.selectedEmbeddingModel = newConfig.embeddingModel;
  config.selectedLLMModel = newConfig.llmModel;
  config.chunkSize = newConfig.chunkSize;
  config.chunkOverlap = newConfig.chunkOverlap;
  config.topK = newConfig.topK;
  config.temperature = newConfig.temperature;

  // Update detailed config
  detailedConfig.value = { ...newConfig };

  console.log("Configuration updated:", newConfig);
};

/**
 * Handles adding a new knowledge base folder
 * Opens the system folder picker to select a new knowledge base
 */
const addKnowledgeBase = async () => {
  try {
    // Check if we're on macOS or Windows and use appropriate method
    const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;

    if (window.electronAPI?.showOpenDialog) {
      // If running in Electron, use the native dialog
      const result = await window.electronAPI.showOpenDialog({
        properties: ["openDirectory"],
        title: "Select Knowledge Base Folder",
      });

      if (!result.canceled && result.filePaths.length > 0) {
        const folderPath = result.filePaths[0];
        const folderName =
          folderPath.split("/").pop() || folderPath.split("\\").pop();

        // Add to options if not already present
        const newOption = {
          value: folderPath,
          label: folderName || "Custom Folder",
        };

        if (!knowledgeBaseOptions.find((opt) => opt.value === folderPath)) {
          knowledgeBaseOptions.push(newOption);
        }

        selectedKnowledgeBase.value = folderPath;
        console.log("Added knowledge base:", folderPath);

        // Kick off indexing on backend and set active
        try {
          const resolved =
            resolveConfigFile(
              config.selectedPipeline,
              folderName || "custom_kb"
            ) || getDefaultConfigForPipeline(config.selectedPipeline);
          if (!resolved) throw new Error("No backend config mapping available");
          await apiClient.index({
            config: resolved,
            knowledge_base: folderName || "custom_kb",
            folder_path: folderPath,
            set_current: true,
          });
          // After indexing, switch selection to the KB name (not the path)
          selectedKnowledgeBase.value = folderName || "custom_kb";
        } catch (e) {
          console.error("Indexing failed:", e);
          alert("Indexing failed. Please check backend logs.");
        }
      }
    } else {
      // Fallback for web browsers - use HTML5 file input with webkitdirectory
      const input = document.createElement("input");
      input.type = "file";
      input.webkitdirectory = true;
      input.multiple = true;

      input.onchange = (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (files && files.length > 0) {
          // Get the directory path from the first file
          const firstFile = files[0];
          const pathParts = firstFile.webkitRelativePath.split("/");
          const folderName = pathParts[0];

          const newOption = {
            value: folderName,
            label: folderName,
          };

          if (!knowledgeBaseOptions.find((opt) => opt.value === folderName)) {
            knowledgeBaseOptions.push(newOption);
          }

          selectedKnowledgeBase.value = folderName;
          console.log("Added knowledge base:", folderName);

          // Browser fallback cannot provide absolute path; we'll just attempt selectRag
          void selectActiveRag(folderName);
        }
      };

      input.click();
    }
  } catch (error) {
    console.error("Failed to add knowledge base:", error);
    alert("Failed to add knowledge base folder");
  }
};

/**
 * Handles sending a chat message and getting a response
 */
const handleSendMessage = async (message: string) => {
  if (!message.trim() || isLoading.value) return;

  // Add user message
  const userMessage: ChatMessage = {
    id: Date.now().toString(),
    type: "user",
    content: message,
    timestamp: new Date(),
  };

  messages.value.push(userMessage);
  isLoading.value = true;

  try {
    // Always call backend
    const queryRequest: QueryRequest = {
      query: message,
      pipeline_config: {
        pipeline_type: config.selectedPipeline,
        embedding_model: detailedConfig.value.embeddingModel,
        llm_model: detailedConfig.value.llmModel,
        chunk_size: detailedConfig.value.chunkSize,
        chunk_overlap: detailedConfig.value.chunkOverlap,
        top_k: detailedConfig.value.topK,
        temperature: detailedConfig.value.temperature,
        additional_params: {
          chunking_strategy: detailedConfig.value.chunkingStrategy,
          retrieval_strategy: detailedConfig.value.retrievalStrategy,
          reranking_strategy: detailedConfig.value.rerankingStrategy,
        },
      } as PipelineConfig,
      top_k: detailedConfig.value.topK,
    };

    const response = await apiClient.query(queryRequest);

    // Update current sources
    currentSources.value = response.retrieved_documents;

    // Add assistant message
    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: "assistant",
      content: response.answer,
      timestamp: new Date(),
      retrieved_documents: response.retrieved_documents,
    };

    messages.value.push(assistantMessage);
  } catch (error) {
    console.error("Query failed:", error);

    // Add error message
    const errorMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: "assistant",
      content:
        "Sorry, I encountered an error while processing your question. Please try again.",
      timestamp: new Date(),
    };

    messages.value.push(errorMessage);
  } finally {
    isLoading.value = false;
  }
};

/**
 * Clears the chat history and sources
 */
const clearChat = () => {
  messages.value = [];
  currentSources.value = [];
};

/**
 * Handles file selection from the FileFinder component
 */
const handleFileSelected = (file: any) => {
  console.log("File selected:", file);
  // In a real application, this would handle file processing
  // For example: preview, add to context, or upload for indexing
};
</script>

<style scoped>
.demo-view {
  padding: var(--spacing-lg);
  padding-right: calc(var(--spacing-lg) * 2);
  max-width: 1400px;
  margin: 0 auto;
}

.demo-config-panel {
  margin-bottom: var(--spacing-lg);
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--spacing-md);
}

.config-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.config-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-primary);
}

.config-label.disabled {
  color: var(--color-text-muted);
}

.config-input {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  font-size: 0.875rem;
  transition: border-color 0.2s ease;
}

.config-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.config-input:disabled {
  background-color: var(--color-surface);
  color: var(--color-text-muted);
  cursor: not-allowed;
}

.knowledge-base-selector {
  display: flex;
  gap: var(--spacing-sm);
  align-items: stretch;
}

.knowledge-base-dropdown {
  flex: 1;
}

.add-button {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm) var(--spacing-md);
}

.add-icon {
  width: 1rem;
  height: 1rem;
}

.configs-button {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  width: 100%;
  justify-content: center;
}

.config-icon {
  width: 1.25rem;
  height: 1.25rem;
}

.tab-navigation {
  margin-bottom: var(--spacing-lg);
  padding: var(--spacing-sm);
}

.tab-buttons {
  display: flex;
  gap: var(--spacing-xs);
}

.tab-button {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-lg);
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  font-weight: 500;
  border-radius: var(--border-radius);
  cursor: pointer;
  transition: all 0.2s ease;
}

.tab-button:hover {
  background: var(--color-surface);
  color: var(--color-text-primary);
}

.tab-button.active {
  background: var(--color-primary);
  color: white;
}

.tab-icon {
  width: 1.25rem;
  height: 1.25rem;
}

.demo-layout {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  height: calc(100vh - 250px);
  min-height: 600px;
}

/* Assistant tab layout - side by side */
.assistant-layout {
  display: flex;
  gap: var(--spacing-lg);
  height: 100%;
  min-height: 0;
}

.chat-panel {
  flex: 0 0 80%;
  display: flex;
  flex-direction: column;
  max-height: 100%;
}

.knowledge-panel {
  flex: 0 0 20%;
  display: flex;
  flex-direction: column;
  max-height: 100%;
  min-width: 0; /* Allow panel to shrink below content width */
}

/* Finder tab layout */
.finder-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.source-count {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  background: var(--color-surface);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius);
}

.empty-sources {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--color-text-muted);
  text-align: center;
}

.empty-icon {
  width: 3rem;
  height: 3rem;
  margin-bottom: var(--spacing-md);
  opacity: 0.5;
}

.sources-list {
  flex: 1;
  overflow-y: auto;
  padding-right: var(--spacing-sm);
}

.chat-actions {
  display: flex;
  gap: var(--spacing-sm);
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

@media (max-width: 1024px) {
  .assistant-layout {
    flex-direction: column;
    gap: var(--spacing-md);
  }

  .chat-panel {
    flex: 1;
    min-height: 400px;
  }

  .knowledge-panel {
    flex: 0 0 300px;
    max-height: 300px;
  }

  .tab-buttons {
    flex-direction: column;
  }

  .tab-button {
    justify-content: center;
  }
}
</style>
