<template>
  <header class="app-header">
    <div class="container">
      <nav class="nav-container">
        <div class="logo">
          <h1>RAG Optimization & Evaluation - MSc Project</h1>
          <h4>
            MultiRAG: A Framework to Fuse Expert Retrievers via Hybrid Reranking
          </h4>
        </div>
        <div class="nav-links">
          <router-link
            to="/demo"
            class="nav-link"
            :class="{ active: $route.path === '/demo' }"
          >
            DEMO
          </router-link>
          <router-link
            to="/evaluation"
            class="nav-link"
            :class="{ active: $route.path === '/evaluation' }"
          >
            EVALUATION
          </router-link>
          <ThemeToggle />
        </div>
      </nav>
    </div>
  </header>
</template>

<script setup lang="ts">
import ThemeToggle from "./ThemeToggle.vue";
</script>

<style scoped>
.app-header {
  background: var(--color-background);
  border-bottom: 1px solid var(--color-border);
  padding: var(--spacing-md) 0;
  position: sticky;
  top: 0;
  z-index: 100;
  transition: background-color var(--transition-normal),
    border-color var(--transition-normal);
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 var(--spacing-lg);
}

.nav-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo h1 {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-primary);
  margin: 0;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: var(--spacing-lg);
}

.nav-link {
  text-decoration: none;
  color: var(--color-text-secondary);
  font-weight: 500;
  font-size: 1rem;
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius);
  transition: all 0.2s ease;
}

.nav-link:hover {
  color: var(--color-primary);
  background-color: var(--color-surface);
}

.nav-link.active {
  color: var(--color-primary);
  background-color: var(--color-primary-light);
}
</style>
