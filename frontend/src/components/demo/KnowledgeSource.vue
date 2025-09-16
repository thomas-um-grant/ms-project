<template>
  <div class="knowledge-source panel">
    <div class="source-header">
      <div class="source-title">
        <span class="source-number">Rank {{ index + 1 }}</span>
      </div>
    </div>

    <div class="source-content" :class="{ expanded: isExpanded }">
      <p class="source-text">{{ truncatedContent }}</p>

      <div v-if="displayMetadata.length" class="source-metadata">
        <div
          v-for="([key, value], idx) in displayMetadata"
          :key="String(key) + idx"
          class="metadata-item"
        >
          <span class="metadata-key"
            >{{ formatMetadataKey(key as string) }}:</span
          >
          <span class="metadata-value">{{ value as any }}</span>
        </div>
      </div>

      <div class="source-footer">
        <button
          class="open-button"
          @click="openInFileManager"
          :title="openButtonTitle"
        >
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M3 7h18M3 7l2 12a2 2 0 002 2h10a2 2 0 002-2l2-12M3 7l2-3h14l2 3"
            />
          </svg>
          <span>open doc</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import type { RetrievedDocument } from "../../types/api";
import { apiClient } from "../../services/ApiClient";

interface Props {
  document: RetrievedDocument;
  index: number;
}

const props = defineProps<Props>();

const isExpanded = ref(false);
const maxLength = 200;

const contentText = computed(() => {
  const c: any = (props.document as any).content;
  if (typeof c === "string") return c;
  if (c && typeof c === "object") {
    return c.content ?? c.text ?? c.snippet ?? c.page_content ?? "";
  }
  return c != null ? String(c) : "";
});

const truncatedContent = computed(() => {
  const text = contentText.value || "";
  if (isExpanded.value || text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "...";
});

const displayMetadata = computed(() => {
  const md = props.document.metadata || {};

  const isPathLikeKey = (key: string) => {
    const k = key.toLowerCase();
    return (
      k.includes("path") ||
      k.includes("file") ||
      k.includes("dir") ||
      k.includes("folder") ||
      k.includes("uri") ||
      k.includes("url")
    );
  };

  const isAbsolutePathString = (val: unknown) => {
    if (typeof val !== "string") return false;
    return (
      val.startsWith("/") || // Unix-like absolute path
      /^[A-Za-z]:[\\/]/.test(val) || // Windows absolute path
      val.startsWith("file://")
    );
  };

  const entries = Object.entries(md).filter(([k, v]) => {
    // Only primitive displayable values
    if (!["string", "number", "boolean"].includes(typeof v as string))
      return false;
    // Drop generic description
    if (k === "description") return false;
    // Hide any explicit path/file hints
    if (isPathLikeKey(k)) return false;
    // Hide source when it's actually a full path
    if (k === "source" && isAbsolutePathString(v)) return false;
    return true;
  });

  const priority = ["source", "page", "section", "title"];
  entries.sort(([a], [b]) => {
    const pa = priority.indexOf(a);
    const pb = priority.indexOf(b);
    const ra = pa === -1 ? 999 : pa;
    const rb = pb === -1 ? 999 : pb;
    return ra - rb || a.localeCompare(b);
  });

  return entries;
});

const openButtonTitle = computed(() => `Reveal this file in your file manager`);

const getPossiblePath = (): string | null => {
  const md = props.document.metadata || ({} as Record<string, any>);
  const candidates = [
    md.path,
    md.file_path,
    md.filepath,
    md.full_path,
    md.source_path,
    md.absolute_path,
  ].filter(Boolean);
  if (candidates.length > 0 && typeof candidates[0] === "string")
    return candidates[0] as string;
  // If only a name is present, we cannot infer absolute path reliably in browser
  return null;
};

const openInFileManager = async () => {
  const filePath = getPossiblePath();
  // Prefer Electron, if available
  const anyWin: any = window as any;
  const api = anyWin.electronAPI;
  if (api?.showItemInFolder && filePath) {
    try {
      await api.showItemInFolder(filePath);
      return;
    } catch (e) {
      console.error("Failed to reveal file via Electron showItemInFolder", e);
    }
  }
  if (api?.openPath && filePath) {
    try {
      await api.openPath(filePath);
      return;
    } catch (e) {
      console.error("Failed to open file via Electron openPath", e);
    }
  }
  // Web fallback: ask backend to reveal/open on host OS
  if (filePath) {
    try {
      await apiClient.openPath({ path: filePath, reveal: true });
      return;
    } catch (e) {
      console.error("Backend open-path failed", e);
    }
  }
  alert(
    filePath
      ? `Cannot open the file manager from the browser. File path: ${filePath}`
      : "No file path is available in the document metadata."
  );
};

// Expand/collapse control removed from UI; keep default collapsed behavior

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

/* Removed score and expand button styles */

.open-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  padding: 4px 8px;
  border-radius: var(--border-radius);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.2s ease;
}

.open-button:hover {
  background: var(--color-background);
  color: var(--color-text-primary);
}

.open-button svg {
  width: 1rem;
  height: 1rem;
}

/* expand button removed */

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

.source-footer {
  display: flex;
  justify-content: flex-end;
  margin-top: var(--spacing-sm);
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
