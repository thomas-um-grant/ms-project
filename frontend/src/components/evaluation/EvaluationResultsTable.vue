<template>
  <div class="evaluation-table-container">
    <!-- Loading State -->
    <div v-if="isLoading" class="loading-section">
      <div class="loading-container">
        <div class="loading-spinner"></div>
        <p>Loading evaluation table...</p>
      </div>
    </div>

    <!-- Main Table Content -->
    <template v-else>
      <!-- Header with Dataset and Metric Selection (hidden when external controls used) -->
      <div class="table-header" v-if="!externalControls">
        <div class="dataset-selector">
          <label for="dataset-select">Datasets:</label>
          <div class="dataset-multiselect">
            <div class="multiselect-display" @click="toggleDatasetDropdown">
              <span v-if="selectedDatasetsList.length === 0" class="placeholder"
                >Select datasets...</span
              >
              <span v-else class="selected-count">
                {{ selectedDatasetsList.length }} dataset{{
                  selectedDatasetsList.length === 1 ? "" : "s"
                }}
                selected
              </span>
              <span
                class="dropdown-arrow"
                :class="{ open: isDatasetDropdownOpen }"
                >▼</span
              >
            </div>
            <div v-if="isDatasetDropdownOpen" class="multiselect-options">
              <div
                v-for="dataset in availableDatasets"
                :key="dataset"
                class="multiselect-option"
                @click="toggleDatasetSelection(dataset)"
              >
                <input
                  type="checkbox"
                  :checked="selectedDatasetsList.includes(dataset)"
                  @click.stop
                />
                <span>{{ formatDatasetName(dataset) }}</span>
              </div>
            </div>
          </div>
        </div>
        <div class="metric-selector">
          <label for="metric-select">Metric:</label>
          <select
            id="metric-select"
            v-model="selectedMetric"
            @change="updateTable"
            class="metric-dropdown"
          >
            <option
              v-for="metric in availableMetrics"
              :key="metric"
              :value="metric"
            >
              {{ formatMetricName(metric) }}
            </option>
          </select>
        </div>
      </div>

      <!-- Results Table -->
      <div v-if="datasets.length > 0" class="table-wrapper">
        <table class="leaderboard-table">
          <thead>
            <tr>
              <th class="model-header">RAG System</th>
              <th
                v-for="dataset in datasets"
                :key="dataset"
                class="dataset-header"
                :title="dataset"
              >
                {{ formatDatasetName(dataset) }}
              </th>
              <th class="average-header">Average</th>
              <th class="stddev-header">Std Dev</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(ragSystem, index) in ragSystems" :key="ragSystem">
              <td class="model-cell">
                <div class="model-info">
                  <span class="rank-badge">{{ index + 1 }}</span>
                  <span class="model-name">{{
                    formatRagSystemName(ragSystem)
                  }}</span>
                </div>
              </td>
              <td
                v-for="dataset in datasets"
                :key="`${ragSystem}-${dataset}`"
                class="score-cell"
              >
                <span class="score-value">
                  {{ formatScore(getScore(ragSystem, dataset)) }}
                </span>
              </td>
              <td class="average-cell">
                <span class="average-value">
                  {{ formatScore(getAverageScore(ragSystem)) }}
                </span>
              </td>
              <td class="stddev-cell">
                <span class="stddev-value">{{
                  formatScore(getStdDevScore(ragSystem))
                }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- No datasets selected message -->
      <div v-else class="no-datasets-message">
        <div class="no-datasets-content">
          <p>📊 No datasets selected</p>
          <p class="no-datasets-subtitle">
            Please select one or more datasets from the dropdown above to view
            the evaluation results.
          </p>
        </div>
      </div>

      <!-- Legend -->
      <div class="table-legend">
        <div class="legend-item">
          <span>
            Scores shown as percentages when < 1.0, otherwise as raw values.
            "N/A" indicates missing or invalid data.
          </span>
        </div>
        <div class="legend-item">
          <span>
            Systems ranked by average performance across
            {{ datasets.length }} selected datasets using
            {{ formatMetricName(metricKey) }}.
          </span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import { AVAILABLE_DATASETS } from "../../constants/configuration";

interface Props {
  selectedDatasets: string[];
  selectedTopK: number | null;
  externalControls?: boolean; // when true, hide internal selectors and use externalSelectedDatasets/externalSelectedMetric
  externalSelectedDatasets?: string[];
  externalSelectedMetric?: string;
  ragSystemsGroup?: string[]; // optional subset filter
}

const props = defineProps<Props>();

// Types for the metrics.json data structure
type MetricsData = {
  [ragSystem: string]: {
    [dataset: string]: {
      [metric: string]: number;
    };
  };
};

type TableData = {
  [metric: string]: {
    [ragSystem: string]: {
      [dataset: string]: number;
    };
  };
};

// Reactive data
const rawMetricsData = ref<MetricsData>({});
const tableData = ref<TableData>({});
const selectedMetric = ref("ndcg_at_10"); // internal metric (ignored when externalControls)
const selectedDatasetsList = ref<string[]>([]); // internal datasets selection
const isLoading = ref(true);
const isDatasetDropdownOpen = ref(false);
const datasetKeyMap = ref<Record<string, string | null>>({});

// External controls flags and active values
const externalControls = computed(() => !!props.externalControls);
const activeSelectedDatasets = computed(() =>
  externalControls.value && props.externalSelectedDatasets
    ? props.externalSelectedDatasets
    : selectedDatasetsList.value
);
const metricKey = computed(() =>
  externalControls.value && props.externalSelectedMetric
    ? props.externalSelectedMetric
    : selectedMetric.value
);

// Load data from metrics.json file
const loadTableData = async () => {
  try {
    isLoading.value = true;
    const response = await fetch("/metrics.json");
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const rawText = await response.text();
    const cleanedText = rawText.replace(/:\s*NaN/g, ": null");
    const data = JSON.parse(cleanedText);
    if (!data || typeof data !== "object")
      throw new Error("Invalid data format: expected object");
    rawMetricsData.value = data;
    buildDatasetKeyMap(data);
    transformDataForTable(data);

    if (!externalControls.value) {
      if (props.selectedDatasets.length > 0) {
        selectedDatasetsList.value = props.selectedDatasets.filter((d) =>
          AVAILABLE_DATASETS.includes(d)
        );
      } else if (selectedDatasetsList.value.length === 0) {
        const preferredDefaults = [
          "infovqa",
          "docvqa",
          "esg_reports_v2",
          "economics_reports_v2",
          "biomedical_lectures_v2",
          "consulting_light_v2",
          "consulting_v2",
        ];
        const defaults = preferredDefaults.filter((d) =>
          AVAILABLE_DATASETS.includes(d)
        );
        selectedDatasetsList.value =
          defaults.length > 0 ? defaults : AVAILABLE_DATASETS.slice(0, 4);
      }
    }
  } catch (error) {
    console.error("Failed to load evaluation table data:", error);
    rawMetricsData.value = {};
    tableData.value = {};
    datasetKeyMap.value = {};
  } finally {
    isLoading.value = false;
  }
};

const buildDatasetKeyMap = (data: MetricsData) => {
  const map: Record<string, string | null> = {};
  const allMetricDatasetKeys = new Set<string>();
  Object.values(data).forEach((ragSystemData) => {
    Object.keys(ragSystemData).forEach((k) => allMetricDatasetKeys.add(k));
  });
  const normalized = (name: string) =>
    name
      .toLowerCase()
      .replace(/^vidore\//, "")
      .replace(/^sherpa_/, "")
      .replace(/_test_subsampled_beir$/, "")
      .replace(/_test_beir$/, "")
      .replace(/_test_subsampled$/, "")
      .replace(/_test$/, "")
      .replace(/_dataset_light$/, "_light")
      .replace(/_dataset$/, "")
      .replace(/_v(\d+)$/, "_v$1")
      .replace(/[_\s]+/g, "_");

  const metricKeysArray = Array.from(allMetricDatasetKeys);

  AVAILABLE_DATASETS.forEach((cfg) => {
    const cfgNorm = normalized(cfg);
    // Exact first
    let match = metricKeysArray.find((k) => normalized(k) === cfgNorm);
    if (!match) {
      // Contains relation (prefer longest key)
      match = metricKeysArray
        .filter(
          (k) =>
            normalized(k).includes(cfgNorm) || cfgNorm.includes(normalized(k))
        )
        .sort((a, b) => b.length - a.length)[0];
    }
    map[cfg] = match || null; // null indicates missing in metrics
  });
  datasetKeyMap.value = map;
};

const transformDataForTable = (data: MetricsData) => {
  const transformed: TableData = {};

  // Get all unique metrics from all systems and datasets
  const allMetrics = new Set<string>();
  const allDatasets = new Set<string>();
  const allRagSystems = new Set<string>();

  // First pass: collect all metrics, datasets, and RAG systems
  Object.entries(data).forEach(([ragSystem, ragSystemData]) => {
    allRagSystems.add(ragSystem);

    Object.entries(ragSystemData).forEach(([dataset, datasetData]) => {
      allDatasets.add(dataset);

      Object.entries(datasetData).forEach(([metric, value]) => {
        // Only include valid metrics (non-null, non-NaN, finite numbers)
        if (typeof value === "number" && !isNaN(value) && isFinite(value)) {
          allMetrics.add(metric);
        }
      });
    });
  });

  // Initialize the transformed structure
  allMetrics.forEach((metric) => {
    transformed[metric] = {};
    allRagSystems.forEach((ragSystem) => {
      transformed[metric][ragSystem] = {};
    });
  });

  // Fill in the data, handling missing values and null values gracefully
  Object.entries(data).forEach(([ragSystem, ragSystemData]) => {
    Object.entries(ragSystemData).forEach(([dataset, datasetData]) => {
      Object.entries(datasetData).forEach(([metric, value]) => {
        // Only store valid numeric values, skip null (converted from NaN) and invalid values
        if (
          value !== null &&
          typeof value === "number" &&
          !isNaN(value) &&
          isFinite(value)
        ) {
          if (!transformed[metric]) {
            transformed[metric] = {};
          }
          if (!transformed[metric][ragSystem]) {
            transformed[metric][ragSystem] = {};
          }
          transformed[metric][ragSystem][dataset] = value;
        }
      });
    });
  });

  tableData.value = transformed;

  // Set default metric to the first commonly available one
  const preferredMetrics = [
    "ndcg_at_10",
    "ndcg_at_5",
    "recall_at_10",
    "recall_at_5",
  ];
  const availableMetrics = Object.keys(transformed);

  for (const preferred of preferredMetrics) {
    if (availableMetrics.includes(preferred)) {
      selectedMetric.value = preferred;
      break;
    }
  }

  // If no preferred metric found, use the first available one
  if (
    !selectedMetric.value ||
    !availableMetrics.includes(selectedMetric.value)
  ) {
    selectedMetric.value =
      availableMetrics.length > 0 ? availableMetrics[0] : "ndcg_at_10";
  }
};

// Available metrics based on the loaded data, filtered and sorted
const availableMetrics = computed(() => {
  const metrics = Object.keys(tableData.value);

  // Group metrics by type for better organization
  const metricGroups = {
    ndcg: metrics.filter((m) => m.startsWith("ndcg_at_")),
    map: metrics.filter((m) => m.startsWith("map_at_")),
    recall: metrics.filter((m) => m.startsWith("recall_at_")),
    precision: metrics.filter((m) => m.startsWith("precision_at_")),
    mrr: metrics.filter((m) => m.startsWith("mrr_at_")),
    naucs: metrics.filter((m) => m.startsWith("naucs_at_")),
    other: metrics.filter(
      (m) =>
        ![
          "ndcg_at_",
          "map_at_",
          "recall_at_",
          "precision_at_",
          "mrr_at_",
          "naucs_at_",
        ].some((prefix) => m.startsWith(prefix))
    ),
  };

  // Sort each group by K value
  const sortByKValue = (metricList: string[]) => {
    return metricList.sort((a, b) => {
      const kA = parseInt(a.split("_at_")[1]) || 0;
      const kB = parseInt(b.split("_at_")[1]) || 0;
      return kA - kB;
    });
  };

  // Create final sorted list prioritizing common evaluation metrics
  const sorted = [
    ...sortByKValue(metricGroups.ndcg),
    ...sortByKValue(metricGroups.recall),
    ...sortByKValue(metricGroups.map),
    ...sortByKValue(metricGroups.mrr),
    ...sortByKValue(metricGroups.precision),
    ...sortByKValue(metricGroups.naucs),
    ...metricGroups.other.sort(),
  ];

  return sorted;
});

// Available datasets come directly from configuration source of truth
const availableDatasets = computed(() => {
  return [...AVAILABLE_DATASETS];
});

// Datasets actually displayed: exactly the user-selected ids (no auto fallback beyond initial default)
const datasets = computed(() => {
  // Only show selected; hide when none
  if (activeSelectedDatasets.value.length === 0) return [];
  // Preserve order of AVAILABLE_DATASETS while filtering by selection
  return AVAILABLE_DATASETS.filter((d) =>
    activeSelectedDatasets.value.includes(d)
  );
});

// Helper function to calculate average score for dataset id via mapping
const getDatasetAverageScore = (datasetId: string, metricData: any): number => {
  const dsKey = datasetKeyMap.value[datasetId];
  if (!dsKey) return 0;
  const scores: number[] = [];
  Object.values(metricData).forEach((ragSystemData: any) => {
    const val = (ragSystemData as any)[dsKey];
    if (typeof val === "number" && !isNaN(val) && isFinite(val))
      scores.push(val);
  });
  if (scores.length === 0) return 0;
  return scores.reduce((a, b) => a + b, 0) / scores.length;
};

// RAG systems, filtered to only show those with data, sorted by average performance
const ragSystems = computed(() => {
  const metricData = tableData.value[metricKey.value as keyof TableData];
  if (!metricData) {
    console.log("No metric data for:", metricKey.value);
    return [];
  }

  // Filter out systems that have no data for any of the selected datasets
  const systemsWithData = Object.keys(metricData).filter((system) => {
    if (props.ragSystemsGroup && props.ragSystemsGroup.length > 0) {
      if (!props.ragSystemsGroup.includes(system)) return false;
    }
    return datasets.value.some((dataset) => {
      const score = metricData[system]?.[dataset];
      return typeof score === "number" && !isNaN(score) && isFinite(score);
    });
  });

  // Sort by average performance (descending)
  return systemsWithData.sort((a, b) => {
    const avgA = getAverageScore(a);
    const avgB = getAverageScore(b);

    // Handle cases where averages might be 0 or NaN
    if (avgA === 0 && avgB === 0) return 0;
    if (avgA === 0) return 1;
    if (avgB === 0) return -1;

    return avgB - avgA;
  });
});

// Helper functions
const getScore = (ragSystem: string, datasetId: string): number | string => {
  const metricData = tableData.value[metricKey.value as keyof TableData];
  if (!metricData || !metricData[ragSystem]) return "N/A";
  const dsKey = datasetKeyMap.value[datasetId];
  if (!dsKey) return "N/A";
  const score = metricData[ragSystem][dsKey];
  if (score === undefined || score === null || isNaN(score) || !isFinite(score))
    return "N/A";
  return score;
};

const getAverageScore = (ragSystem: string): number => {
  const scores = datasets.value
    .map((dataset) => getScore(ragSystem, dataset))
    .filter(
      (score): score is number =>
        typeof score === "number" && !isNaN(score) && isFinite(score)
    );

  if (scores.length === 0) return 0;
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
};

// Standard deviation across currently selected datasets (population SD). Requires at least 2 scores, else NaN -> shown as N/A.
const getStdDevScore = (ragSystem: string): number | string => {
  const scores = datasets.value
    .map((dataset) => getScore(ragSystem, dataset))
    .filter(
      (score): score is number =>
        typeof score === "number" && !isNaN(score) && isFinite(score)
    );
  if (scores.length < 2) return "N/A"; // not enough data
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance =
    scores.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) /
    scores.length; // population variance
  const sd = Math.sqrt(variance);
  return sd;
};

const formatScore = (score: number | string): string => {
  if (typeof score === "string") return score; // Return "N/A" as is
  if (score === 0 || isNaN(score) || !isFinite(score)) return "N/A";

  // Format based on the typical range of the metric
  if (score < 0.1) {
    // For very small values, show more decimal places
    return (score * 100).toFixed(2);
  } else if (score < 1) {
    // For values between 0.1 and 1, show as percentages with 1 decimal
    return (score * 100).toFixed(1);
  } else {
    // For values >= 1, they're likely already in percentage or other units
    return score.toFixed(2);
  }
};

const formatMetricName = (metric: string): string => {
  // Enhanced metric name formatting
  if (metric.includes("_at_")) {
    const parts = metric.split("_at_");
    const metricType = parts[0].toUpperCase();
    const kValue = parts[1];
    return `${metricType}@${kValue}`;
  }

  // Handle other metric patterns
  return metric
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const formatDatasetName = (datasetId: string): string => {
  return datasetId
    .replace(/^vidore\//, "")
    .replace(/_/g, " ")
    .replace(/\b(v\d+)$/i, (m) => m.toUpperCase())
    .replace(/\b([a-z])/g, (c) => c.toUpperCase());
};

const formatRagSystemName = (system: string): string => {
  // Enhanced RAG system name formatting
  const systemMap: { [key: string]: string } = {
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

  return (
    systemMap[system] ||
    system.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
  );
};

const updateTable = () => {
  // This function is called when metric or dataset selection changes
  // The computed properties will automatically update the table
  console.log("Table updated", {
    metric: metricKey.value,
    datasets: activeSelectedDatasets.value.length,
  });
};

// Dataset multiselect functions (internal only)
const toggleDatasetDropdown = () => {
  if (externalControls.value) return; // ignore when external
  isDatasetDropdownOpen.value = !isDatasetDropdownOpen.value;
};

const toggleDatasetSelection = (dataset: string) => {
  if (externalControls.value) return; // external control cannot modify
  const index = selectedDatasetsList.value.indexOf(dataset);
  if (index > -1) selectedDatasetsList.value.splice(index, 1);
  else selectedDatasetsList.value.push(dataset);
  updateTable();
};

onMounted(() => {
  loadTableData();
  if (!externalControls.value && props.selectedDatasets.length > 0) {
    selectedDatasetsList.value = [...props.selectedDatasets];
  }
  const handleClickOutside = (event: MouseEvent) => {
    const target = event.target as Element;
    if (!target.closest(".dataset-multiselect")) {
      isDatasetDropdownOpen.value = false;
    }
  };
  document.addEventListener("click", handleClickOutside);
  onUnmounted(() => {
    document.removeEventListener("click", handleClickOutside);
  });
});
</script>

<style scoped>
.evaluation-table-container {
  margin: var(--spacing-lg) 0;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--color-surface);
  border-radius: 12px;
  border: 1px solid var(--color-border);
  transition: background-color 0.3s ease, border-color 0.3s ease;
}

.dataset-selector,
.metric-selector {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.dataset-selector label,
.metric-selector label {
  font-weight: 500;
  color: var(--color-text-secondary);
}

.dataset-dropdown,
.metric-dropdown {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-surface);
  font-size: 0.875rem;
  color: var(--color-text-primary);
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.3s ease;
  min-width: 150px;
}

/* Custom multiselect styles */
.dataset-multiselect {
  position: relative;
  min-width: 200px;
}

.multiselect-display {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-surface);
  font-size: 0.875rem;
  color: var(--color-text-primary);
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.3s ease;
  display: flex;
  justify-content: space-between;
  align-items: center;
  user-select: none;
}

.multiselect-display:hover {
  border-color: var(--color-primary);
}

.placeholder {
  color: var(--color-text-muted);
}

.selected-count {
  color: var(--color-text-primary);
  font-weight: 500;
}

.dropdown-arrow {
  transition: transform 0.2s ease;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.dropdown-arrow.open {
  transform: rotate(180deg);
}

.multiselect-options {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  max-height: 200px;
  overflow-y: auto;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  box-shadow: var(--shadow-lg);
  z-index: 1000;
  margin-top: 4px;
}

.multiselect-option {
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.multiselect-option:hover {
  background: var(--color-background-secondary);
}

.multiselect-option input[type="checkbox"] {
  margin: 0;
  cursor: pointer;
}

.multiselect-option span {
  font-size: 0.875rem;
  color: var(--color-text-primary);
}

.multiselect-display:focus,
.metric-dropdown:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-light);
}

.table-wrapper {
  overflow-x: auto;
  background: var(--color-surface);
  border-radius: 16px;
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-md);
  transition: background-color 0.3s ease, border-color 0.3s ease;
}

.leaderboard-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.leaderboard-table th {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px 12px;
  text-align: center;
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  position: sticky;
  top: 0;
  z-index: 10;
}

.leaderboard-table th:first-child {
  text-align: left;
  border-top-left-radius: 16px;
}

.leaderboard-table th:last-child {
  border-top-right-radius: 16px;
}

/* Sticky first column (RAG System) */
.leaderboard-table th:first-child,
.leaderboard-table td.model-cell {
  position: sticky;
  left: 0;
  z-index: 25; /* Above other cells but below potential overlays */
  background: var(--color-surface); /* ensure opaque coverage */
  background-clip: padding-box;
}

/* Solid background only for body cells so header gradient remains */
/* (header keeps gradient, we re-apply gradient after this generic rule) */

/* Elevate header cell above body cells */
.leaderboard-table th.model-header,
.leaderboard-table th:first-child {
  z-index: 25;
}

/* Add subtle shadow edge when scrolling to indicate stickiness */
.leaderboard-table td.model-cell::after,
.leaderboard-table th:first-child::after {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  width: 8px;
  height: 100%;
  pointer-events: none;
  background: linear-gradient(to right, rgba(0, 0, 0, 0.08), rgba(0, 0, 0, 0));
  opacity: 0;
}

/* Show the gradient edge only when the container is scrollable & scrolled (JS hook optional) */
.table-wrapper.scrolled-left .leaderboard-table td.model-cell::after,
.table-wrapper.scrolled-left .leaderboard-table th:first-child::after {
  opacity: 1;
}

/* Improve hover contrast while keeping sticky behavior */
/* Keep sticky cell opaque even on hover (row hover still applies behind) */
.leaderboard-table tr:hover td.model-cell {
  background: var(--color-surface);
}

/* Re-apply gradient for sticky header cell */
.leaderboard-table th:first-child {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  background-clip: padding-box;
}

.model-header {
  min-width: 200px;
  max-width: 200px;
}

.dataset-header {
  min-width: 100px;
  max-width: 120px;
  text-align: center;
  word-wrap: break-word;
  white-space: normal;
  line-height: 1.2;
  vertical-align: middle;
}

.average-header {
  min-width: 80px;
  font-weight: 700;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
}

.stddev-header {
  min-width: 80px;
  font-weight: 700;
  background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
}

.leaderboard-table td {
  padding: 12px;
  text-align: center;
  border-bottom: 1px solid var(--color-border);
}

.leaderboard-table tr:hover {
  background: var(--color-primary-light);
}

.model-cell {
  text-align: left !important;
  padding: 16px 12px;
}

.model-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: var(--color-surface-hover);
  color: var(--color-text-secondary);
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 700;
  transition: background-color 0.3s ease;
}

.model-name {
  font-weight: 600;
  color: var(--color-text-primary);
}

.score-cell {
  font-family: "SF Mono", "Monaco", "Consolas", monospace;
}

.score-value {
  font-weight: 600;
  color: var(--color-text-primary);
}

.average-cell {
  background: var(--color-primary-light);
  font-weight: 700;
  border-left: 2px solid var(--color-primary);
}

.stddev-cell {
  background: var(--color-primary-light);
  font-weight: 600;
  border-left: 1px solid var(--color-primary);
}

.stddev-value {
  font-family: "SF Mono", "Monaco", "Consolas", monospace;
  color: var(--color-text-primary);
  font-weight: 600;
}

.average-value {
  font-family: "SF Mono", "Monaco", "Consolas", monospace;
  color: var(--color-primary);
  font-weight: 700;
}

.table-legend {
  display: flex;
  gap: var(--spacing-md);
  margin-top: var(--spacing-md);
  padding: var(--spacing-sm);
  justify-content: center;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

/* Loading styles */
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

/* No datasets selected message */
.no-datasets-message {
  text-align: center;
  padding: 60px 20px;
  background: var(--color-surface);
  border-radius: 16px;
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-md);
  transition: background-color 0.3s ease, border-color 0.3s ease;
}

.no-datasets-content p:first-child {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 8px 0;
}

.no-datasets-subtitle {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  margin: 0;
  line-height: 1.5;
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
@media (max-width: 768px) {
  .table-header {
    flex-direction: column;
    gap: var(--spacing-sm);
    text-align: center;
  }

  .dataset-selector,
  .metric-selector {
    width: 100%;
    justify-content: center;
  }

  .dataset-multiselect,
  .metric-dropdown {
    min-width: 200px;
  }

  .dataset-header {
    min-width: 80px;
    font-size: 0.7rem;
  }

  .leaderboard-table th,
  .leaderboard-table td {
    padding: 8px 6px;
  }

  .model-name {
    font-size: 0.8rem;
  }
}
</style>
