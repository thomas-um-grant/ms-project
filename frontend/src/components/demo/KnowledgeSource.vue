<template>
  <div class="knowledge-source panel">
    <div class="source-header">
      <div class="source-title">
        <span class="source-number">Doc {{ index + 1 }}</span>
        <span class="source-score"
          >{{ (document.score * 100).toFixed(1) }}% match</span
        >
      </div>
      <button
        @click="toggleExpanded"
        class="expand-button"
        :class="{ expanded: isExpanded }"
      >
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>
    </div>

    <div class="source-content" :class="{ expanded: isExpanded }">
      <p class="source-text">{{ truncatedContent }}</p>

      <div v-if="document.metadata" class="source-metadata">
        <div
          v-for="(value, key) in document.metadata"
          :key="key"
          class="metadata-item"
        >
          <span class="metadata-key">{{ formatMetadataKey(key) }}:</span>
          <span class="metadata-value">{{ value }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import type { RetrievedDocument } from "../../types/api";

interface Props {
  document: RetrievedDocument;
  index: number;
}

const props = defineProps<Props>();

const isExpanded = ref(false);
const maxLength = 200;

const truncatedContent = computed(() => {
  if (isExpanded.value || props.document.content.length <= maxLength) {
    return props.document.content;
  }
  return props.document.content.substring(0, maxLength) + "...";
});

const toggleExpanded = () => {
  isExpanded.value = !isExpanded.value;
};

const formatMetadataKey = (key: string) => {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};
</script>

<style scoped>
.knowledge-source {
  background: var(--color-surface);
  border-left: 4px solid var(--color-primary);
  margin-bottom: var(--spacing-sm);
  transition: all 0.2s ease;
}

.knowledge-source:hover {
  border-left-color: #1d4ed8;
  box-shadow: var(--shadow-md);
}

.source-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--spacing-sm);
}

.source-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.source-number {
  font-weight: 600;
  color: var(--color-text-primary);
  font-size: 0.875rem;
}

.source-score {
  background: var(--color-primary);
  color: white;
  padding: 2px var(--spacing-xs);
  border-radius: calc(var(--border-radius) / 2);
  font-size: 0.75rem;
  font-weight: 500;
}

.expand-button {
  background: none;
  border: none;
  cursor: pointer;
  padding: var(--spacing-xs);
  border-radius: var(--border-radius);
  color: var(--color-text-secondary);
  transition: all 0.2s ease;
}

.expand-button:hover {
  background: var(--color-background);
  color: var(--color-text-primary);
}

.expand-button svg {
  width: 1rem;
  height: 1rem;
  transition: transform 0.2s ease;
}

.expand-button.expanded svg {
  transform: rotate(180deg);
}

.source-content {
  overflow: hidden;
  transition: all 0.3s ease;
}

.source-text {
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--color-text-primary);
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.source-metadata {
  margin-top: var(--spacing-sm);
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--color-border);
}

.metadata-item {
  display: flex;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-xs);
  font-size: 0.75rem;
}

.metadata-key {
  font-weight: 500;
  color: var(--color-text-secondary);
  min-width: 80px;
}

.metadata-value {
  color: var(--color-text-muted);
  word-break: break-word;
}
</style>
