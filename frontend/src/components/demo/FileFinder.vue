<template>
  <div class="file-finder">
    <!-- Search Input -->
    <div class="search-section">
      <div class="search-input-wrapper">
        <svg
          class="search-icon"
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
        <input
          v-model="searchQuery"
          class="search-input"
          type="text"
          placeholder="Search for files by content, name, or description..."
          @input="handleSearch"
          @keydown.enter="performSearch"
        />
        <Button
          v-if="searchQuery"
          variant="secondary"
          size="sm"
          @click="clearSearch"
          class="clear-button"
        >
          Clear
        </Button>
      </div>
      <Button
        variant="primary"
        @click="performSearch"
        :loading="isSearching"
        :disabled="!searchQuery.trim()"
        class="search-button"
      >
        Search Files
      </Button>
    </div>

    <!-- Search Results -->
    <div class="results-section">
      <div v-if="!hasSearched && !isSearching" class="empty-state">
        <div class="empty-icon">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
            />
          </svg>
        </div>
        <h3>Find Documents</h3>
        <p>
          Search through your document collection to find specific files and
          content.
        </p>
      </div>

      <div v-else-if="isSearching" class="loading-state">
        <div class="loading-spinner"></div>
        <p>Searching documents...</p>
      </div>

      <div v-else-if="searchResults.length === 0" class="no-results">
        <div class="no-results-icon">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.29-1.009-5.674-2.326M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
            />
          </svg>
        </div>
        <h3>No Files Found</h3>
        <p>
          No documents match your search criteria. Try different keywords or
          check your spelling.
        </p>
      </div>

      <div v-else class="results-list">
        <div class="results-header">
          <span class="results-count">
            {{ searchResults.length }} file{{
              searchResults.length !== 1 ? "s" : ""
            }}
            found
          </span>
          <span class="search-time" v-if="searchTime">
            ({{ searchTime }}ms)
          </span>
        </div>

        <div
          v-for="(file, index) in searchResults"
          :key="file.id"
          class="file-item"
          @click="openFile(file)"
        >
          <div class="file-header">
            <div class="file-info">
              <h4 class="file-name">{{ file.name }}</h4>
              <span class="file-score"
                >{{ (file.score * 100).toFixed(1) }}% match</span
              >
            </div>
            <div class="file-actions">
              <Button
                variant="secondary"
                size="sm"
                @click.stop="downloadFile(file)"
                class="download-btn"
              >
                <svg
                  class="download-icon"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                Download
              </Button>
            </div>
          </div>

          <p class="file-description">{{ file.description }}</p>

          <div class="file-metadata">
            <span class="metadata-item">
              <svg
                class="metadata-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                />
              </svg>
              {{ file.type }}
            </span>
            <span class="metadata-item">
              <svg
                class="metadata-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M8 7V3a2 2 0 012-2h4a2 2 0 012 2v4m-4 8h4m-4-4h4m-4-4h4M7 21h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              {{ formatFileSize(file.size) }}
            </span>
            <span class="metadata-item">
              <svg
                class="metadata-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M8 7V3a2 2 0 012-2h4a2 2 0 012 2v4m-6 9l2 2 4-4m6-2V7a2 2 0 00-2-2H9a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2z"
                />
              </svg>
              {{ formatDate(file.lastModified) }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import Button from "../common/Button.vue";

interface FileResult {
  id: string;
  name: string;
  description: string;
  score: number;
  type: string;
  size: number;
  lastModified: Date;
  path: string;
  url?: string;
}

interface Emits {
  (e: "fileSelected", file: FileResult): void;
}

const emit = defineEmits<Emits>();

const searchQuery = ref("");
const searchResults = ref<FileResult[]>([]);
const isSearching = ref(false);
const hasSearched = ref(false);
const searchTime = ref<number | null>(null);

const handleSearch = () => {
  // Debounce search - could implement auto-search here
};

const performSearch = async () => {
  if (!searchQuery.value.trim() || isSearching.value) return;

  isSearching.value = true;
  hasSearched.value = true;
  const startTime = Date.now();

  try {
    // Simulate API call - replace with actual search endpoint
    await new Promise((resolve) => setTimeout(resolve, 800));

    // Mock search results - replace with actual API call
    searchResults.value = generateMockResults(searchQuery.value);
    searchTime.value = Date.now() - startTime;
  } catch (error) {
    console.error("Search failed:", error);
    searchResults.value = [];
  } finally {
    isSearching.value = false;
  }
};

const clearSearch = () => {
  searchQuery.value = "";
  searchResults.value = [];
  hasSearched.value = false;
  searchTime.value = null;
};

const openFile = (file: FileResult) => {
  emit("fileSelected", file);
  // Could also open file in new tab or preview modal
};

const downloadFile = async (file: FileResult) => {
  try {
    // Simulate file download - replace with actual download logic
    console.log("Downloading file:", file.name);

    // For now, create a mock download
    const element = document.createElement("a");
    element.href = file.url || "#";
    element.download = file.name;
    element.click();
  } catch (error) {
    console.error("Download failed:", error);
    alert("Download failed. Please try again.");
  }
};

const formatFileSize = (bytes: number): string => {
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
};

const formatDate = (date: Date): string => {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
};

// Mock data generator - replace with actual API integration
const generateMockResults = (query: string): FileResult[] => {
  const mockFiles = [
    {
      id: "1",
      name: "research_paper_nlp.pdf",
      description:
        "Natural Language Processing research paper discussing transformer architectures and their applications in modern AI systems.",
      score: 0.95,
      type: "PDF",
      size: 2048576,
      lastModified: new Date("2024-01-15"),
      path: "/documents/research/research_paper_nlp.pdf",
    },
    {
      id: "2",
      name: "project_documentation.md",
      description:
        "Complete project documentation including setup instructions, API reference, and deployment guidelines.",
      score: 0.87,
      type: "Markdown",
      size: 45632,
      lastModified: new Date("2024-02-20"),
      path: "/documents/projects/project_documentation.md",
    },
    {
      id: "3",
      name: "data_analysis_report.xlsx",
      description:
        "Comprehensive data analysis report with charts, tables, and statistical insights from the quarterly review.",
      score: 0.78,
      type: "Excel",
      size: 1536000,
      lastModified: new Date("2024-03-10"),
      path: "/documents/reports/data_analysis_report.xlsx",
    },
  ];

  // Filter based on query (simple contains check)
  return mockFiles.filter(
    (file) =>
      file.name.toLowerCase().includes(query.toLowerCase()) ||
      file.description.toLowerCase().includes(query.toLowerCase())
  );
};
</script>

<style scoped>
.file-finder {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.search-section {
  display: flex;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-lg);
}

.search-input-wrapper {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: var(--spacing-md);
  width: 1.25rem;
  height: 1.25rem;
  color: var(--color-text-muted);
  pointer-events: none;
}

.search-input {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md) var(--spacing-sm) 3rem;
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  font-size: 0.9rem;
  transition: border-color 0.2s ease;
}

.search-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.clear-button {
  position: absolute;
  right: var(--spacing-xs);
}

.search-button {
  flex-shrink: 0;
}

.results-section {
  flex: 1;
  overflow-y: auto;
}

.empty-state,
.loading-state,
.no-results {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: var(--spacing-2xl);
  color: var(--color-text-muted);
  height: 100%;
}

.empty-icon,
.no-results-icon {
  width: 4rem;
  height: 4rem;
  margin-bottom: var(--spacing-md);
  opacity: 0.5;
}

.loading-spinner {
  width: 2rem;
  height: 2rem;
  border: 3px solid var(--color-border);
  border-top: 3px solid var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: var(--spacing-md);
}

.empty-state h3,
.no-results h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: var(--spacing-sm);
  color: var(--color-text-primary);
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-sm);
  border-bottom: 1px solid var(--color-border);
}

.results-count {
  font-weight: 500;
  color: var(--color-text-primary);
}

.search-time {
  font-size: 0.875rem;
  color: var(--color-text-muted);
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.file-item {
  padding: var(--spacing-lg);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  background: var(--color-background);
  cursor: pointer;
  transition: all 0.2s ease;
}

.file-item:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-md);
}

.file-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-sm);
}

.file-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.file-name {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

.file-score {
  background: var(--color-primary);
  color: white;
  padding: 2px var(--spacing-xs);
  border-radius: calc(var(--border-radius) / 2);
  font-size: 0.75rem;
  font-weight: 500;
}

.file-actions {
  flex-shrink: 0;
}

.download-btn {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.download-icon {
  width: 1rem;
  height: 1rem;
}

.file-description {
  margin: var(--spacing-sm) 0;
  color: var(--color-text-secondary);
  line-height: 1.5;
}

.file-metadata {
  display: flex;
  gap: var(--spacing-lg);
  margin-top: var(--spacing-sm);
}

.metadata-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 0.875rem;
  color: var(--color-text-muted);
}

.metadata-icon {
  width: 1rem;
  height: 1rem;
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
