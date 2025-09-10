<template>
  <div class="stats-view">
    <!-- Evaluation Explanation Panel -->
    <div class="explanation-panel panel">
      <div class="panel-header">
        <h3>Evaluation Metrics & Datasets</h3>
      </div>
      <div class="explanation-content">
        <p>
          This evaluation system tests retrieval performance across diverse
          datasets and measures quality using multiple metrics. Explore the
          datasets and metrics below to understand what each measures and why
          it's important for evaluation.
        </p>

        <p>
          Note: Some dataset were produced with pages coming from the same
          documents. Since models have large context windows allowing them to
          support a few files at once, it is legitimate to assert that it would
          be sufficient to retrieve the right document only, even if the page
          retrieved is not the right one. Therefore, we created a v1 and v2
          versions for these datasets. The v1 will count partial matches (good
          document retrieved) as correct, whereas the v2 will require exact
          matches (correct document and page) to be considered correct.
        </p>

        <!-- Datasets Carousel -->
        <Carousel
          title="Available Datasets"
          :items="datasetInfo"
          @item-click="openDatasetModal"
        />

        <!-- Metrics Carousel -->
        <Carousel
          title="Evaluation Metrics"
          :items="metricInfo"
          @item-click="openMetricModal"
        />
      </div>
    </div>

    <!-- Dataset Modal -->
    <InfoModal
      :show="showDatasetModal"
      :item="selectedDatasetItem"
      type="dataset"
      @close="closeDatasetModal"
    />

    <!-- Metric Modal -->
    <InfoModal
      :show="showMetricModal"
      :item="selectedMetricItem"
      type="metric"
      @close="closeMetricModal"
    />

    <!-- Results Display -->
    <div class="results-section">
      <div class="results-header">
        <h2>Evaluation Results</h2>
        <!-- CSV Export -->
        <div class="results-actions">
          <Button
            variant="secondary"
            size="sm"
            @click="exportMetricsJson"
            :disabled="!dashboardData || isDashboardLoading"
          >
            Export Data
          </Button>
        </div>
      </div>

      <!-- Performance Table - Now at the top -->
      <div v-if="!isDashboardLoading" class="card performance-table-card">
        <!-- Shared Controls -->
        <div class="shared-controls">
          <div class="shared-datasets">
            <label>Datasets:</label>
            <div
              class="shared-multiselect"
              @click="toggleGlobalDatasetDropdown"
            >
              <span
                v-if="globalSelectedDatasets.length === 0"
                class="placeholder"
                >Select datasets...</span
              >
              <span v-else class="selected-count"
                >{{ globalSelectedDatasets.length }} selected</span
              >
              <span
                class="dropdown-arrow"
                :class="{ open: isGlobalDatasetDropdownOpen }"
                >▼</span
              >
            </div>
            <div v-if="isGlobalDatasetDropdownOpen" class="shared-options">
              <div
                v-for="d in datasetOptions"
                :key="d.value"
                class="shared-option"
                @click.stop="toggleGlobalDataset(d.value)"
              >
                <input
                  type="checkbox"
                  :checked="globalSelectedDatasets.includes(d.value)"
                  @click.stop
                />
                <span>{{ d.label }}</span>
              </div>
            </div>
          </div>
          <div class="shared-metrics">
            <label>Metric:</label>
            <select v-model="globalSelectedMetric" class="metric-dropdown">
              <option
                v-for="m in metricOptions"
                :key="m.value"
                :value="m.value"
              >
                {{ m.label }}
              </option>
            </select>
          </div>
        </div>

        <!-- Grouped Tables -->
        <div class="grouped-tables">
          <div class="group-card" v-for="group in ragGroups" :key="group.name">
            <h3 class="group-title">{{ group.name }}</h3>
            <EvaluationResultsTable
              :selectedDatasets="globalSelectedDatasets"
              :selectedTopK="selectedTopK"
              :externalControls="true"
              :externalSelectedDatasets="globalSelectedDatasets"
              :externalSelectedMetric="globalSelectedMetric"
              :ragSystemsGroup="group.systems"
            />
          </div>
        </div>
      </div>

      <!-- Loading state for table when dashboard is loading -->
      <div v-else-if="isDashboardLoading" class="loading-section">
        <div class="loading-container">
          <div class="loading-spinner"></div>
          <p>Loading evaluation data...</p>
        </div>
      </div>

      <!-- Detailed Metrics Dashboard -->
      <div v-if="dashboardData" class="metrics-dashboard">
        <div class="dashboard-section-header">
          <h2>Detailed Analysis</h2>
        </div>

        <!-- Configuration Panel for Detailed Analysis -->
        <div class="evaluation-config-panel panel">
          <div class="panel-header">
            <h3>Configuration</h3>
          </div>

          <div class="config-grid">
            <!-- Retrieval System Selection -->
            <div class="config-group">
              <label class="config-label">RAG Systems</label>
              <Dropdown
                v-model="selectedRetrievalSystems"
                :options="retrievalSystemOptions"
                placeholder="Select RAG systems..."
                multiple
                @update:modelValue="refreshAnalysis"
              />
            </div>

            <!-- Dataset Selection (multi) -->
            <div class="config-group">
              <label class="config-label">Datasets</label>
              <div
                style="
                  display: flex;
                  gap: 8px;
                  align-items: center;
                  width: 100%;
                "
              >
                <Dropdown
                  v-model="selectedAnalysisDatasets"
                  :options="datasetOptions"
                  placeholder="Select datasets..."
                  multiple
                  :disabled="useAllDatasets"
                  @update:modelValue="refreshAnalysis"
                />
                <label
                  class="all-datasets-toggle"
                  :title="'Average over all datasets'"
                >
                  <input
                    type="checkbox"
                    v-model="useAllDatasets"
                    @change="onToggleAllDatasets"
                    aria-label="Use all datasets"
                  />
                  <span
                    class="toggle-pill"
                    :class="{ checked: useAllDatasets }"
                  >
                    <span class="toggle-handle" />
                    <span class="toggle-text">All</span>
                  </span>
                </label>
              </div>
            </div>

            <!-- Top-K Selection -->
            <div class="config-group">
              <label class="config-label">Top-K Value</label>
              <Dropdown
                v-model="selectedTopK"
                :options="topKOptions"
                placeholder="Select top-K value..."
                @update:modelValue="refreshAnalysis"
              />
            </div>

            <!-- Max K (Trend) -->
            <div class="config-group">
              <label class="config-label">Max K (Trend)</label>
              <Dropdown
                v-model="selectedMaxK"
                :options="maxKOptions"
                placeholder="Select max K..."
                @update:modelValue="refreshAnalysis"
              />
            </div>
          </div>
        </div>

        <!-- RAG System Comparison Charts or Placeholder -->
        <div
          class="card"
          v-if="useAllDatasets || selectedAnalysisDatasets.length > 0"
        >
          <h3>
            Key Metrics Across RAG Systems (Dataset:
            {{ formattedAnalysisDataset }}) @K={{ currentK }}
          </h3>
          <div class="comparison-grid">
            <DatasetComparisonChart
              :data="analysisChartData"
              :metric="`ndcg_at_${currentK}`"
              :title="`NDCG@${currentK}`"
            />
            <DatasetComparisonChart
              :data="analysisChartData"
              :metric="`recall_at_${currentK}`"
              :title="`Recall@${currentK}`"
            />
            <DatasetComparisonChart
              :data="analysisChartData"
              :metric="`precision_at_${currentK}`"
              :title="`Precision@${currentK}`"
            />
          </div>
        </div>
        <div class="card" v-else aria-live="polite">
          <h3>Key Metrics Across RAG Systems</h3>
          <p class="no-selection">
            No datasets selected. Choose one or more datasets from the dropdown
            above or enable the 'All' toggle to view the analysis.
          </p>
        </div>

        <!-- RAG System Trends -->
        <div
          class="card"
          v-if="
            (useAllDatasets || selectedAnalysisDatasets.length > 0) &&
            Object.keys(analysisChartData).length > 0
          "
        >
          <h3>
            Performance Trends Across K (Dataset:
            {{ formattedAnalysisDataset }})
          </h3>
          <div class="line-charts-grid">
            <MultiDatasetLineChart
              :data="analysisChartData"
              metricBase="ndcg"
              title="NDCG Trends"
              :ks="truncatedKs"
            />
            <MultiDatasetLineChart
              :data="analysisChartData"
              metricBase="recall"
              title="Recall Trends"
              :ks="truncatedKs"
            />
          </div>
        </div>
        <div
          class="card"
          v-else-if="!(useAllDatasets || selectedAnalysisDatasets.length > 0)"
        >
          <h3>Performance Trends Across K</h3>
          <p class="no-selection">
            Select at least one dataset (or enable 'All') to display trend
            charts.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, watch, computed } from "vue";
import Button from "../components/common/Button.vue";
import Dropdown from "../components/common/Dropdown.vue";
import Carousel from "../components/common/Carousel.vue";
import InfoModal from "../components/common/InfoModal.vue";
import DatasetComparisonChart from "../components/evaluation/DatasetComparisonChart.vue";
import MultiDatasetLineChart from "../components/evaluation/MultiDatasetLineChart.vue";
import EvaluationResultsTable from "../components/evaluation/EvaluationResultsTable.vue";
// Removed unused EvaluationService (was mock wrapper not invoked anywhere)
import {
  AVAILABLE_DATASETS,
  AVAILABLE_METRICS,
  DEFAULT_SELECTED_DATASETS,
  DEFAULT_SELECTED_RAG_SYSTEMS,
  DEFAULT_ANALYSIS_DATASET,
} from "../constants/configuration";
import {
  generateMockEvaluationResults,
  simulateDelay,
  MOCK_DATASETS,
  MOCK_METRICS,
} from "../utils/mockData";
import type { EvaluationResult, EvaluateRequest } from "../types/api";
import type { BenchmarkDataset } from "../types/retriever";

const isEvaluating = ref(false);
const evaluationResults = ref<EvaluationResult[]>([]);
const availableDatasets = ref<BenchmarkDataset[]>([]);

// Dashboard-specific reactive variables
const dashboardData = ref(null);
const isDashboardLoading = ref(false);

// Flag to prevent multiple simultaneous calls
let fetchInProgress = false;

// Multiple selections for dropdowns
const selectedRetrievalSystems = ref<string[]>([]);
const selectedDatasets = ref<string[]>([]);
const selectedTopK = ref<number | null>(null);
const selectedMetrics = ref<string[]>([]);

// Multi dataset selection for analysis section
const selectedAnalysisDatasets = ref<string[]>([]);
// When true, aggregate (average) metrics across ALL datasets for charts
const useAllDatasets = ref(false);

// Separate selections for the performance table
const selectedTableDatasets = ref<string[]>([]);
const selectedTableMetric = ref<string | null>(null);

// Modal state for datasets and metrics info
const showDatasetModal = ref(false);
const showMetricModal = ref(false);
const selectedDatasetItem = ref<any>(null);
const selectedMetricItem = ref<any>(null);

// Information data loaded from JSON
const datasetInfo = ref<any[]>([]);
const metricInfo = ref<any[]>([]);

// Options for dropdowns - using centralized constants
const topKOptions = [
  { value: 1, label: "1" },
  { value: 3, label: "3" },
  { value: 5, label: "5" },
  { value: 10, label: "10" },
  { value: 20, label: "20" },
  { value: 50, label: "50" },
  { value: 100, label: "100" },
];

// Replace hardcoded retrievalSystemOptions with dynamic list built from metrics.json keys (systemMap formatting)
const retrievalSystemOptions = ref<{ value: string; label: string }[]>([]);

// Mapping for nicer RAG system labels (duplicate of table systemMap for consistency)
const systemMap: Record<string, string> = {
  traditional_vector: "Traditional Vector",
  traditional_bm25: "Traditional BM25",
  traditional_hybrid: "Traditional",
  traditional_hybrid_rerank_llm: "Traditional (LLM)",
  traditional_hybrid_rerank_jina: "Traditional (Jina)",
  multimodal: "Multimodal",
  multimodal_rerank_llm: "Multimodal (LLM)",
  multimodal_rerank_jina: "Multimodal (Jina)",
  jina_text: "Jina text embed",
  jina_image: "Jina image embed",
  multi_rank_max: "MultiRAG (max-rank)",
  multi_norm_avg_l1: "MultiRAG (L1 norm-avg)",
  multi_norm_avg_l2: "MultiRAG (L2 norm-avg)",
  multi_rank_fuse_l1: "MultiRAG (L1 norm-rank)",
  multi_rank_fuse_l2: "MultiRAG (L2 norm-rank)",
  multi_rrf: "MultiRAG (RRF k=100)",
  multi_rrf_30: "MultiRAG (RRF k=30)",
};

const formatRagSystemName = (system: string) =>
  systemMap[system] ||
  system.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

// Formatted dataset name for headings
const formattedAnalysisDataset = computed(() => {
  if (useAllDatasets.value) return "All";
  const count = selectedAnalysisDatasets.value.length;
  if (count === 0) return "-";
  if (count === 1) {
    return selectedAnalysisDatasets.value[0]
      .replace(/^vidore\//, "")
      .replace(/_/g, " ")
      .replace(/\b(v\d+)$/i, (m) => m.toUpperCase())
      .replace(/\b([a-z])/g, (c) => c.toUpperCase());
  }
  return `${count} datasets`;
});

// Computed properties for dashboard
const currentK = computed(() => {
  return selectedTopK.value || 10;
});

const availableKs = computed(() => {
  // Line charts should always show all K values for trend analysis
  return [1, 3, 5, 10, 20, 50, 100];
});

// Max K selection for trimming trend lines
const maxKOptions = [
  { value: 3, label: "Up to 3" },
  { value: 5, label: "Up to 5" },
  { value: 10, label: "Up to 10" },
  { value: 20, label: "Up to 20" },
  { value: 50, label: "Up to 50" },
  { value: 100, label: "Up to 100" },
];
const selectedMaxK = ref<number>(100);
const truncatedKs = computed(() =>
  availableKs.value.filter((k) => k <= (selectedMaxK.value || 100))
);

// Build analysis data object: keys are formatted RAG system names; each contains metrics for the selected dataset
const analysisChartData = computed(() => {
  if (!dashboardData.value) return {};
  const systems = selectedRetrievalSystems.value.length
    ? selectedRetrievalSystems.value
    : Object.keys(dashboardData.value);
  const result: Record<string, any> = {};
  // Determine which metric keys we actually need:
  // For bar charts: ndcg/recall/precision at currentK
  // For line charts: ndcg/recall across truncatedKs
  const primaryBases = ["ndcg", "recall", "precision"]; // precision only for bar chart
  const trendBases = ["ndcg", "recall"]; // line charts
  const neededKeys = new Set<string>();
  primaryBases.forEach((b) => neededKeys.add(`${b}_at_${currentK.value}`));
  truncatedKs.value.forEach((k) =>
    trendBases.forEach((b) => neededKeys.add(`${b}_at_${k}`))
  );

  systems.forEach((system) => {
    const systemData = (dashboardData.value as any)[system];
    if (!systemData) return;

    const accumulateAvg = (datasets: string[]) => {
      const acc: Record<string, { sum: number; count: number }> = {};
      datasets.forEach((ds) => {
        const metrics = systemData[ds];
        if (!metrics) return;
        Object.entries(metrics).forEach(([k, v]) => {
          if (neededKeys.has(k) && typeof v === "number" && !isNaN(v)) {
            if (!acc[k]) acc[k] = { sum: 0, count: 0 };
            acc[k].sum += v as number;
            acc[k].count += 1;
          }
        });
      });
      const out: Record<string, number> = {};
      Object.entries(acc).forEach(([k, { sum, count }]) => {
        out[k] = count ? sum / count : 0;
      });
      return out;
    };

    if (useAllDatasets.value) {
      result[formatRagSystemName(system)] = accumulateAvg(
        Object.keys(systemData)
      );
    } else if (selectedAnalysisDatasets.value.length === 1) {
      const ds = selectedAnalysisDatasets.value[0];
      const metrics = systemData[ds];
      if (!metrics) return;
      const filtered: Record<string, number> = {};
      Object.entries(metrics).forEach(([k, v]) => {
        if (neededKeys.has(k) && typeof v === "number")
          filtered[k] = v as number;
      });
      result[formatRagSystemName(system)] = filtered;
    } else if (selectedAnalysisDatasets.value.length > 1) {
      const valid = selectedAnalysisDatasets.value.filter(
        (ds) => systemData[ds]
      );
      if (valid.length === 0) return;
      result[formatRagSystemName(system)] = accumulateAvg(valid);
    }
  });
  return result;
});

// Refresh handler (currently just triggers dependent computeds/log)
const refreshAnalysis = () => {
  // Intentionally lightweight; computeds react automatically
  console.log("Analysis updated", {
    datasets: selectedAnalysisDatasets.value,
    all: useAllDatasets.value,
    systems: selectedRetrievalSystems.value,
    k: selectedTopK.value,
  });
};

const onToggleAllDatasets = () => {
  if (useAllDatasets.value) {
    selectedAnalysisDatasets.value = [];
  }
  refreshAnalysis();
};

// Remove filteredDashboardData (dataset-focused) since charts are system-focused now

// Remove unused configuration state and handlers
// Configuration is now handled by the dropdown selections directly

/**
 * Fetches evaluation results based on current selections
 * Uses mock data generation for consistent testing
 */
const fetchResults = async () => {
  console.log("fetchResults called", {
    retrievalSystems: selectedRetrievalSystems.value.length,
    datasets: selectedAnalysisDatasets.value,
    topK: selectedTopK.value,
    metrics: selectedMetrics.value.length,
    fetchInProgress,
  });

  // Prevent multiple simultaneous calls
  if (fetchInProgress) {
    console.log("Fetch already in progress, skipping");
    return;
  }

  if (
    selectedRetrievalSystems.value.length === 0 ||
    (selectedAnalysisDatasets.value.length === 0 && !useAllDatasets.value) ||
    selectedTopK.value === null ||
    selectedMetrics.value.length === 0
  ) {
    // Don't fetch if no options are selected
    console.log("No options selected, clearing results");
    evaluationResults.value = [];
    isEvaluating.value = false;
    return;
  }

  console.log("Setting isEvaluating to true");
  fetchInProgress = true;
  isEvaluating.value = true;

  try {
    // Use centralized mock data generation
    const datasetsForMock = useAllDatasets.value
      ? AVAILABLE_DATASETS
      : selectedAnalysisDatasets.value;
    const results = generateMockEvaluationResults(
      datasetsForMock,
      selectedMetrics.value,
      selectedRetrievalSystems.value
    );

    console.log("Generated results:", results.length);

    // Simulate network delay for realistic UX
    await simulateDelay(500, 1200);

    evaluationResults.value = results;
    console.log(
      "Results set, evaluationResults length:",
      evaluationResults.value.length
    );
  } catch (error) {
    console.error("Failed to fetch evaluation results:", error);
    evaluationResults.value = [];
  } finally {
    console.log("Setting isEvaluating to false");
    fetchInProgress = false;
    isEvaluating.value = false;
  }
};

// Export the full raw metrics.json currently loaded in the dashboard
const exportMetricsJson = async () => {
  try {
    // If already loaded in memory, serialize directly to avoid refetch
    if (dashboardData.value) {
      const blob = new Blob([JSON.stringify(dashboardData.value, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "metrics.json";
      a.click();
      URL.revokeObjectURL(url);
      return;
    }
    const resp = await fetch("/metrics.json");
    if (!resp.ok)
      throw new Error(`Failed to fetch metrics.json: ${resp.status}`);
    const text = await resp.text();
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "metrics.json";
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error("Export metrics.json failed", e);
  }
};

// Modal handlers
const openDatasetModal = (dataset: any) => {
  selectedDatasetItem.value = dataset;
  showDatasetModal.value = true;
};

const closeDatasetModal = () => {
  showDatasetModal.value = false;
  selectedDatasetItem.value = null;
};

const openMetricModal = (metric: any) => {
  selectedMetricItem.value = metric;
  showMetricModal.value = true;
};

const closeMetricModal = () => {
  showMetricModal.value = false;
  selectedMetricItem.value = null;
};

// Load evaluation information from JSON
const loadEvaluationInfo = async () => {
  try {
    const response = await fetch("/evaluation-info.json");
    const data = await response.json();
    datasetInfo.value = data.datasets || [];
    metricInfo.value = data.metrics || [];
  } catch (error) {
    console.error("Failed to load evaluation info:", error);
    // Fallback to empty arrays
    datasetInfo.value = [];
    metricInfo.value = [];
  }
};

// Load dashboard data from metrics.json
const loadDashboardData = async () => {
  try {
    isDashboardLoading.value = true;
    const response = await fetch("/metrics.json");

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get the raw text first to handle NaN values
    const rawText = await response.text();

    // Replace NaN values with null before parsing JSON
    const cleanedText = rawText.replace(/:\s*NaN/g, ": null");

    const data = JSON.parse(cleanedText);
    dashboardData.value = data;

    // Derive RAG system options
    const systems = Object.keys(data);
    retrievalSystemOptions.value = systems.map((s) => ({
      value: s,
      label: formatRagSystemName(s),
    }));
  } catch (error) {
    console.error("Failed to load dashboard data:", error);
  } finally {
    isDashboardLoading.value = false;
  }
};

// Options derived from AVAILABLE_DATASETS / AVAILABLE_METRICS for dropdowns
interface SimpleOption {
  value: string;
  label: string;
}
const datasetOptions: SimpleOption[] = AVAILABLE_DATASETS.map((dataset) => ({
  value: dataset,
  label: dataset
    .replace(/^vidore\//, "")
    .replace(/_/g, " ")
    .replace(/\b(v\d+)$/i, (m) => m.toUpperCase())
    .replace(/\b([a-z])/g, (c) => c.toUpperCase()),
}));
const metricOptions = AVAILABLE_METRICS; // retains full set for CSV export/mock generation

// New global shared state for grouped tables
const globalSelectedDatasets = ref<string[]>([]);
const globalSelectedMetric = ref<string>("ndcg_at_10");
const isGlobalDatasetDropdownOpen = ref(false);

const ragGroups = [
  {
    name: "Base",
    systems: ["traditional_hybrid", "multimodal"],
  },
  {
    name: "Jina Reranker",
    systems: [
      "traditional_hybrid_rerank_jina",
      "multimodal_rerank_jina",
      "jina_text",
      "jina_image",
    ],
  },
  {
    name: "LLM Reranker",
    systems: ["traditional_hybrid_rerank_llm", "multimodal_rerank_llm"],
  },
  {
    name: "MultiRAG",
    systems: [
      "multi_rank_max",
      "multi_norm_avg_l1",
      "multi_norm_avg_l2",
      "multi_rank_fuse_l1",
      "multi_rank_fuse_l2",
      "multi_rrf",
      "multi_rrf_30",
    ],
  },
];

const toggleGlobalDatasetDropdown = () => {
  isGlobalDatasetDropdownOpen.value = !isGlobalDatasetDropdownOpen.value;
};
const toggleGlobalDataset = (value: string) => {
  const idx = globalSelectedDatasets.value.indexOf(value);
  if (idx > -1) globalSelectedDatasets.value.splice(idx, 1);
  else globalSelectedDatasets.value.push(value);
};

document.addEventListener("click", (e: any) => {
  const target = e.target as HTMLElement;
  if (!target.closest(".shared-datasets")) {
    isGlobalDatasetDropdownOpen.value = false;
  }
});

onMounted(async () => {
  try {
    // Load evaluation info (datasets and metrics descriptions)
    await loadEvaluationInfo();

    // Load dashboard data
    await loadDashboardData();

    // Use mock datasets with proper BenchmarkDataset structure
    availableDatasets.value = MOCK_DATASETS;

    // Auto-select all available options for better initial experience
    if (retrievalSystemOptions.value.length > 0) {
      // Use configured default RAG systems for detailed analysis
      selectedRetrievalSystems.value = DEFAULT_SELECTED_RAG_SYSTEMS.filter(
        (system) =>
          retrievalSystemOptions.value.some((option) => option.value === system)
      );
    }
    if (datasetOptions.length > 0) {
      // Use configured default dataset for detailed analysis
      const initial = AVAILABLE_DATASETS.includes(DEFAULT_ANALYSIS_DATASET)
        ? DEFAULT_ANALYSIS_DATASET
        : (datasetOptions[0].value as string);
      if (initial) selectedAnalysisDatasets.value = [initial];
    }
    if (topKOptions.length > 0) {
      // Set default Top-K to 10
      selectedTopK.value = 10;
    }
    if (metricOptions.length > 0) {
      // Select all metrics by default
      selectedMetrics.value = metricOptions.map(
        (option) => option.value as string
      );
    }

    // Initialize table-specific selections
    if (datasetOptions.length > 0) {
      // Select default datasets from configuration
      selectedTableDatasets.value = DEFAULT_SELECTED_DATASETS.filter(
        (dataset) => AVAILABLE_DATASETS.includes(dataset)
      );
    }
    if (metricOptions.length > 0) {
      // Select a key metric for table by default
      const keyMetrics = [
        "ndcg_at_10",
        "ndcg_at_5",
        "recall_at_10",
        "recall_at_5",
        "map_at_5",
        "mrr_at_5",
      ];
      const foundKeyMetric = metricOptions.find((option) =>
        keyMetrics.includes(option.value as string)
      );

      // If a key metric is found, use it, otherwise use the first metric
      selectedTableMetric.value = foundKeyMetric
        ? (foundKeyMetric.value as string)
        : (metricOptions[0].value as string);
    }

    // initialize global datasets using default configuration
    globalSelectedDatasets.value = DEFAULT_SELECTED_DATASETS.filter((dataset) =>
      AVAILABLE_DATASETS.includes(dataset)
    );
    // Force default table metric to nDCG@10 (override any earlier guess by index)
    const ndcg10 = metricOptions.find((m) => m.value === "ndcg_at_10");
    if (ndcg10) {
      globalSelectedMetric.value = ndcg10.value as string;
    }

    console.log("onMounted completed, selections:", {
      retrievalSystems: selectedRetrievalSystems.value,
      analysisDatasets: selectedAnalysisDatasets.value,
      topK: selectedTopK.value,
      metrics: selectedMetrics.value,
    });
  } catch (error) {
    console.error("Failed to initialize evaluation configuration:", error);
  }
});
</script>

<style scoped>
.stats-view {
  padding: var(--spacing-lg);
  max-width: 1400px;
  margin: 0 auto;
}

.explanation-panel {
  margin-bottom: var(--spacing-lg);
}

.explanation-content {
  line-height: 1.6;
}

.explanation-content > p:first-child {
  margin-bottom: var(--spacing-lg);
  color: var(--color-text-primary);
}

.evaluation-config-panel {
  margin-bottom: var(--spacing-lg);
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
}

.config-group {
  display: flex;
  flex-direction: column;
}

.config-label {
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: var(--spacing-sm);
  font-size: 0.875rem;
}

.explanation-content p {
  margin-bottom: var(--spacing-md);
  color: var(--color-text-primary);
}

.metrics-list {
  margin: var(--spacing-md) 0;
  padding-left: var(--spacing-lg);
}

.metrics-list li {
  margin-bottom: var(--spacing-sm);
  color: var(--color-text-primary);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.results-section {
  margin-top: var(--spacing-xl);
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.results-header h2 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text-primary);
}

.results-actions {
  display: flex;
  gap: var(--spacing-sm);
}

/* Dashboard styles */
.metrics-dashboard {
  margin-top: var(--spacing-xl);
}

.no-selection {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--color-text-muted);
  font-style: italic;
}

.dashboard-section-header {
  margin-bottom: var(--spacing-lg);
}

.dashboard-section-header h2 {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text-primary);
  margin: 0;
  padding-left: var(--spacing-md);
  border-left: 4px solid var(--color-primary);
}

.performance-table-card {
  margin-bottom: var(--spacing-xl);
}

.performance-table-card h3 {
  display: none; /* Hide the duplicate title since it's already in the table component */
}

.card {
  background: var(--color-background);
  border: 1px solid var(--color-border);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: var(--shadow-md);
  transition: all 0.3s ease;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
  border-color: var(--color-border-hover);
}

.card h3 {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.card h3::before {
  content: "";
  width: 4px;
  height: 16px;
  background: linear-gradient(
    135deg,
    var(--color-primary) 0%,
    var(--color-primary-hover) 100%
  );
  border-radius: 2px;
}

.comparison-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.line-charts-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
}

.loading-section {
  text-align: center;
  padding: 60px 20px;
  color: var(--color-text-muted);
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--color-border);
  border-top: 3px solid var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/* Responsive design */
@media (min-width: 1200px) {
  .line-charts-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .comparison-grid {
    grid-template-columns: 1fr;
  }
}

.shared-controls {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  margin-bottom: 24px;
  align-items: flex-end;
}

.shared-datasets,
.shared-metrics {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.shared-multiselect {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-surface);
  cursor: pointer;
  min-width: 220px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: border-color 0.2s ease, background-color 0.3s ease,
    box-shadow 0.2s ease;
}

.shared-multiselect:hover {
  border-color: var(--color-primary);
  background-color: var(--color-surface-hover);
}

.shared-multiselect:focus-within {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-light);
}

.shared-options {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  max-height: 220px;
  overflow-y: auto;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  box-shadow: var(--shadow-lg);
  z-index: 50;
  margin-top: 4px;
}

.shared-option {
  padding: 8px 12px;
  display: flex;
  gap: 8px;
  align-items: center;
  cursor: pointer;
}

.shared-option:hover {
  background: var(--color-background-secondary);
}

/* Stack each group table on its own row */
.grouped-tables {
  display: flex;
  flex-direction: column;
  gap: 32px;
}
.group-card {
  width: 100%;
  background: var(--color-background);
  border: 1px solid var(--color-border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow-sm);
}
.group-title {
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

.metric-dropdown {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-surface);
  font-size: 0.875rem;
  color: var(--color-text-primary);
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.3s ease,
    box-shadow 0.2s ease;
  min-width: 160px;
  appearance: none;
  background-image: url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23666' d='M6 8.5 2.5 5h7z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 8px center;
  padding-right: 32px;
}

.metric-dropdown:hover {
  border-color: var(--color-primary);
  background-color: var(--color-surface-hover);
}

.metric-dropdown:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-light);
}

/* All datasets pill toggle */
.all-datasets-toggle {
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  user-select: none;
}
.all-datasets-toggle input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}
.toggle-pill {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  padding: 4px 10px 4px 34px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 500;
  color: var(--color-text-secondary);
  transition: background 0.25s ease, border-color 0.25s ease, color 0.25s ease;
  min-height: 28px;
}
.toggle-pill:hover {
  border-color: var(--color-primary);
}
.toggle-pill.checked {
  background: linear-gradient(
    135deg,
    var(--color-primary) 0%,
    var(--color-primary-hover) 100%
  );
  border-color: var(--color-primary-hover);
  color: #fff;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}
.toggle-pill .toggle-handle {
  position: absolute;
  left: 6px;
  top: 50%;
  transform: translateY(-50%);
  width: 18px;
  height: 18px;
  background: var(--color-background);
  border-radius: 50%;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2),
    0 0 0 2px rgba(255, 255, 255, 0.6) inset;
  transition: background 0.25s ease;
}
.toggle-pill.checked .toggle-handle {
  background: #fff;
}
.toggle-pill .toggle-text {
  line-height: 1;
  position: relative;
  top: 1px;
}
</style>
