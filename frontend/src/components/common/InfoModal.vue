<template>
  <Teleport to="body">
    <div v-if="show" class="modal-overlay" @click="handleOverlayClick">
      <div
        class="modal-container"
        :style="{ '--accent-color': item?.color || '#4CAF50' }"
        @click.stop
      >
        <div class="modal-header">
          <div class="modal-title-section">
            <div class="modal-icon">{{ item?.icon }}</div>
            <div>
              <h2 class="modal-title">{{ item?.displayName }}</h2>
              <span class="modal-category">{{ item?.category }}</span>
            </div>
          </div>
          <button class="modal-close" @click="$emit('close')">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path
                d="M18 6L6 18M6 6L18 18"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </button>
        </div>

        <div class="modal-content">
          <div class="modal-description">
            <p>{{ item?.description }}</p>
          </div>

          <div v-if="item?.details" class="modal-details">
            <!-- For datasets -->
            <div v-if="type === 'dataset'" class="details-grid">
              <div class="detail-item">
                <strong>Purpose:</strong>
                <span>{{ item.details.purpose }}</span>
              </div>
              <div class="detail-item">
                <strong>Domain:</strong>
                <span>{{ item.details.domain }}</span>
              </div>
              <div class="detail-item">
                <strong>Size:</strong>
                <span class="multiline">{{ item.details.size }}</span>
              </div>
              <div class="detail-item">
                <strong>Format:</strong>
                <span>{{ item.details.format }}</span>
              </div>
              <div class="detail-item full-width">
                <strong>Key Challenges:</strong>
                <span>{{ item.details.challenges }}</span>
              </div>
            </div>

            <!-- For metrics -->
            <div v-else-if="type === 'metric'" class="details-grid">
              <div class="detail-item">
                <strong>Formula:</strong>
                <MathFormula :formula="formattedFormula" />
              </div>
              <div class="detail-item">
                <strong>Range:</strong>
                <span>{{ item.details.range }}</span>
              </div>
              <div class="detail-item full-width">
                <strong>Interpretation:</strong>
                <span>{{ item.details.interpretation }}</span>
              </div>
              <div class="detail-item full-width">
                <strong>Key Advantages:</strong>
                <span>{{ item.details.advantages }}</span>
              </div>
              <div class="detail-item full-width">
                <strong>Best Use Case:</strong>
                <span>{{ item.details.use_case }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, watch, onBeforeUnmount } from "vue";
import MathFormula from "./MathFormula.vue";

interface ModalProps {
  show: boolean;
  item: any | null;
  type: "dataset" | "metric";
}

const props = defineProps<ModalProps>();
const emit = defineEmits(["close"]);

// --- Body scroll lock handling (supports multiple modals) ---
let openModalCount = 0; // module-level across all InfoModal instances
let originalOverflow: string | null = null;
let originalPaddingRight: string | null = null;

const lockBodyScroll = () => {
  if (openModalCount === 0) {
    const body = document.body;
    // Store original styles to restore later
    originalOverflow = body.style.overflow || null;
    originalPaddingRight = body.style.paddingRight || null;
    const scrollBarWidth =
      window.innerWidth - document.documentElement.clientWidth;
    body.style.overflow = "hidden";
    if (scrollBarWidth > 0) {
      body.style.paddingRight = `${scrollBarWidth}px`;
    }
  }
  openModalCount++;
};

const unlockBodyScroll = () => {
  if (openModalCount > 0) openModalCount--;
  if (openModalCount === 0) {
    const body = document.body;
    // Restore original styles
    if (originalOverflow !== null) body.style.overflow = originalOverflow;
    else body.style.removeProperty("overflow");
    if (originalPaddingRight !== null)
      body.style.paddingRight = originalPaddingRight;
    else body.style.removeProperty("padding-right");
    originalOverflow = null;
    originalPaddingRight = null;
  }
};

watch(
  () => props.show,
  (val, oldVal) => {
    if (val && !oldVal) {
      lockBodyScroll();
    } else if (!val && oldVal) {
      unlockBodyScroll();
    }
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  if (props.show) {
    unlockBodyScroll();
  }
});

const formattedFormula = computed(() => {
  const details = props.item?.details || {};
  // Prefer explicit latex field
  if (details.formulaLatex) return details.formulaLatex as string;
  const raw = details.formula || details.formulaPlain || "";
  if (!raw) return "";
  if (/\\[a-zA-Z]+|\^|_\{/g.test(raw)) return raw; // already TeX
  // Minimal transformation otherwise
  return raw
    .replace(/@k/gi, "_k")
    .replace(
      /\b(NDCG|MAP|MRR|Recall|Precision)\b/gi,
      (m: string) => `\\mathrm{${m.toUpperCase()}}`
    );
});

const handleOverlayClick = (event: MouseEvent) => {
  if (event.target === event.currentTarget) {
    emit("close");
  }
};
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-lg);
  z-index: 1000;
  animation: fadeIn 0.2s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-container {
  background: var(--color-background);
  border-radius: 16px;
  box-shadow: var(--shadow-lg);
  max-width: 600px;
  width: 100%;
  max-height: 90vh;
  overflow: hidden;
  position: relative;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: var(--spacing-xl);
  border-bottom: 1px solid var(--color-border);
  background: linear-gradient(135deg, var(--accent-color) 15, transparent);
  position: relative;
}

.modal-header::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--accent-color);
}

.modal-title-section {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.modal-icon {
  font-size: 2.5rem;
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-color);
  color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.modal-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text-primary);
  margin: 0 0 var(--spacing-xs) 0;
}

.modal-category {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 500;
}

.modal-close {
  background: none;
  border: none;
  color: var(--color-text-muted);
  cursor: pointer;
  padding: var(--spacing-sm);
  border-radius: 6px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-close:hover {
  background: var(--color-background-hover);
  color: var(--color-text-primary);
}

.modal-content {
  padding: var(--spacing-xl);
  overflow-y: auto;
  max-height: calc(90vh - 140px);
}

.modal-description {
  margin-bottom: var(--spacing-lg);
}

.modal-description p {
  font-size: 1rem;
  line-height: 1.6;
  color: var(--color-text-primary);
  margin: 0;
}

.modal-details {
  border-top: 1px solid var(--color-border);
  padding-top: var(--spacing-lg);
}

.details-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--spacing-md);
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.detail-item.full-width {
  grid-column: 1 / -1;
}

.detail-item strong {
  color: var(--accent-color);
  font-weight: 600;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.detail-item span {
  color: var(--color-text-secondary);
  line-height: 1.5;
}

.detail-item span.multiline {
  white-space: pre-line; /* preserve \n line breaks */
}

/* old formula style removed in favor of MathFormula component */

/* Responsive design */
@media (max-width: 768px) {
  .modal-overlay {
    padding: var(--spacing-md);
  }

  .modal-header {
    padding: var(--spacing-lg);
  }

  .modal-content {
    padding: var(--spacing-lg);
  }

  .modal-title-section {
    gap: var(--spacing-sm);
  }

  .modal-icon {
    width: 48px;
    height: 48px;
    font-size: 2rem;
  }

  .modal-title {
    font-size: 1.25rem;
  }
}
</style>
