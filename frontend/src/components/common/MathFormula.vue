<template>
  <span
    ref="container"
    class="math-formula"
    :class="displayMode ? 'block' : ''"
  ></span>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from "vue";
import * as katex from "katex";
import "katex/dist/katex.min.css";

interface Props {
  formula: string;
  displayMode?: boolean;
  errorColor?: string;
}

const props = withDefaults(defineProps<Props>(), {
  displayMode: false,
  errorColor: "#cc0000",
});

const container = ref<HTMLElement | null>(null);
let renderTimeout: number | null = null;

const renderFormula = () => {
  if (!container.value || !props.formula) return;
  try {
    katex.render(props.formula, container.value, {
      displayMode: props.displayMode,
      throwOnError: false,
      errorColor: props.errorColor,
      strict: "ignore",
      trust: true,
      macros: {
        "\\E": "\\mathbb{E}",
      },
    });
  } catch (e) {
    // Fallback to plain text if unexpected error
    container.value.textContent = props.formula;
  }
};

const scheduleRender = () => {
  if (renderTimeout) cancelAnimationFrame(renderTimeout);
  renderTimeout = requestAnimationFrame(renderFormula);
};

watch(() => [props.formula, props.displayMode], scheduleRender, {
  immediate: true,
});

onMounted(renderFormula);

onBeforeUnmount(() => {
  if (renderTimeout) cancelAnimationFrame(renderTimeout);
});
</script>

<style scoped>
.math-formula {
  font-size: 0.9rem;
  line-height: 1.4;
  display: inline-block;
  padding: 4px 8px;
  border-radius: 6px;
  background: var(--color-background-elevated, var(--color-surface));
  border: 1px solid var(--color-border);
  overflow-x: auto;
  max-width: 100%;
}
.block {
  display: block;
}
:deep(.katex-display) {
  margin: 0;
}
</style>
