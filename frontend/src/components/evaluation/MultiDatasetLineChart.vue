<template>
  <div ref="root" class="chart"></div>
</template>

<script setup>
import * as d3 from "d3";
import { ref, onMounted, onBeforeUnmount, watch } from "vue";
import { themeService } from "../../services/ThemeService";

const props = defineProps({
  data: { type: Object, required: true }, // Profile data with all datasets
  metricBase: { type: String, required: true }, // e.g., 'ndcg', 'recall'
  title: { type: String, required: true },
  ks: { type: Array, default: () => [1, 3, 5, 10, 20, 50, 100] },
});

const root = ref(null);
let resizeObserver;
let themeUnsubscribe;

function render() {
  if (!root.value || !props.data) return;

  const el = root.value;
  el.innerHTML = "";

  // Increased right margin to ensure legend is not cropped; extra top/bottom padding
  const margin = { top: 50, right: 180, bottom: 70, left: 65 };
  const rect = el.getBoundingClientRect();
  const W = rect.width || 900;
  const H = 400;
  const width = W - margin.left - margin.right;
  const height = H - margin.top - margin.bottom;

  const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Extract datasets and their series
  const datasets = Object.keys(props.data);

  const series = datasets
    .map((dataset) => {
      const values = props.ks
        .map((k) => {
          const metricKey = `${props.metricBase}_at_${k}`;
          const value = props.data[dataset][metricKey] || null;
          return {
            k,
            value,
          };
        })
        .filter((d) => d.value !== null);

      return {
        dataset: dataset
          .replace("vidore/", "")
          .replace("_test_subsampled_beir", "")
          .replace("_test_beir", ""),
        fullName: dataset,
        values,
      };
    })
    .filter((s) => s.values.length > 0);

  if (series.length === 0) return;

  // Scales
  const x = d3
    .scalePoint()
    .domain(props.ks.map(String))
    .range([0, width])
    .padding(0.1);

  const allValues = series.flatMap((s) => s.values.map((v) => v.value));
  if (allValues.length === 0) return;
  const yMinRaw = d3.min(allValues);
  const yMaxRaw = d3.max(allValues);
  let yMin = yMinRaw == null ? 0 : yMinRaw;
  let yMax = yMaxRaw == null ? 1 : yMaxRaw;
  // If all values identical, expand slightly for visibility
  if (yMin === yMax) {
    const pad = yMin === 0 ? 0.05 : Math.abs(yMin) * 0.05;
    yMin = Math.max(0, yMin - pad);
    yMax = yMax + pad;
  }
  const y = d3
    .scaleLinear()
    .domain([yMin, yMax]) // Exact min / max domain (no .nice() to keep bounds tight)
    .range([height, 0]);

  // Modern color scale with better differentiation
  const themeColors = [
    "#3b82f6", // Blue
    "#8b5cf6", // Purple
    "#06b6d4", // Cyan
    "#10b981", // Emerald
    "#f59e0b", // Amber
    "#ef4444", // Red
    "#8b5a2b", // Brown
    "#6b7280", // Gray
  ];

  const colors = series.map((_, i) => themeColors[i % themeColors.length]);

  // Line generator
  const line = d3
    .line()
    .x((d) => x(String(d.k)))
    .y((d) => y(d.value))
    .curve(d3.curveMonotoneX);

  // Modern axes styling - get theme-aware colors
  const textColor =
    getComputedStyle(el).getPropertyValue("--color-text-secondary").trim() ||
    "#64748b";
  const borderColor =
    getComputedStyle(el).getPropertyValue("--color-border").trim() || "#e2e8f0";
  const surfaceColor =
    getComputedStyle(el).getPropertyValue("--color-surface").trim() ||
    "#f1f5f9";

  const xAxis = d3.axisBottom(x);
  const yAxis = d3.axisLeft(y).ticks(8).tickFormat(d3.format(".2f"));

  g.append("g")
    .attr("transform", `translate(0,${height})`)
    .call(xAxis)
    .selectAll("text")
    .style("font-size", "11px")
    .style("font-weight", "500")
    .style("fill", textColor);

  g.append("g")
    .call(yAxis)
    .selectAll("text")
    .style("font-size", "11px")
    .style("font-weight", "500")
    .style("fill", textColor);

  // Style axis lines
  g.selectAll(".domain")
    .style("stroke", borderColor)
    .style("stroke-width", "1px");

  g.selectAll(".tick line")
    .style("stroke", surfaceColor)
    .style("stroke-width", "1px");

  // Draw modern lines with gradients
  series.forEach((s, i) => {
    const lineColor = colors[i];

    // Line with shadow effect
    g.append("path")
      .datum(s.values)
      .attr("fill", "none")
      .attr("stroke", lineColor)
      .attr("stroke-width", 3)
      .attr("d", line)
      .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))");

    // Modern points with better styling
    g.selectAll(`.dot-${i}`)
      .data(s.values)
      .enter()
      .append("circle")
      .attr("class", `dot-${i}`)
      .attr("cx", (d) => x(String(d.k)))
      .attr("cy", (d) => y(d.value))
      .attr("r", 4)
      .attr("fill", "white")
      .attr("stroke", lineColor)
      .attr("stroke-width", 2)
      .style("filter", "drop-shadow(0 1px 2px rgba(0,0,0,0.1))")
      .append("title")
      .text(
        (d) =>
          `${s.dataset} ${props.metricBase}@${d.k}: ${d3.format(".3f")(
            d.value
          )}`
      );
  });

  // Modern legend with better styling - get theme-aware colors
  const primaryTextColor =
    getComputedStyle(el).getPropertyValue("--color-text-primary").trim() ||
    "#374151";

  const legend = g
    .append("g")
    .attr("transform", `translate(${width + 15}, 20)`);

  series.forEach((s, i) => {
    const legendRow = legend
      .append("g")
      .attr("transform", `translate(0, ${i * 24})`);

    legendRow
      .append("circle")
      .attr("cx", 6)
      .attr("cy", 0)
      .attr("r", 4)
      .attr("fill", "white")
      .attr("stroke", colors[i])
      .attr("stroke-width", 2);

    legendRow
      .append("text")
      .attr("x", 16)
      .attr("y", 0)
      .attr("dy", "0.35em")
      .style("font-size", "12px")
      .style("font-weight", "500")
      .style("fill", primaryTextColor)
      .text(s.dataset);
  });

  // Modern title styling
  g.append("text")
    .attr("x", width / 2)
    .attr("y", -20)
    .attr("text-anchor", "middle")
    .style("font-size", "16px")
    .style("font-weight", "600")
    .style("fill", primaryTextColor)
    .text(props.title);

  // Modern axis labels
  g.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left)
    .attr("x", 0 - height / 2)
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .style("font-size", "12px")
    .style("font-weight", "500")
    .style("fill", textColor)
    .text(props.metricBase.toUpperCase() + " Score");

  g.append("text")
    .attr(
      "transform",
      `translate(${width / 2}, ${height + margin.bottom - 10})`
    )
    .style("text-anchor", "middle")
    .style("font-size", "12px")
    .style("font-weight", "500")
    .style("fill", textColor)
    .text("k");
}

function observeResize() {
  resizeObserver = new ResizeObserver(() => render());
  resizeObserver.observe(root.value);
}

onMounted(() => {
  render();
  observeResize();

  // Listen for theme changes and re-render
  themeUnsubscribe = themeService.onThemeChange(() => {
    // Add a small delay to ensure CSS variables have been updated
    setTimeout(() => {
      render();
    }, 50);
  });
});

onBeforeUnmount(() => {
  if (resizeObserver) resizeObserver.disconnect();
  if (themeUnsubscribe) themeUnsubscribe();
});
watch(() => [props.data, props.metricBase, props.ks], render, { deep: true });
</script>

<style scoped>
.chart {
  width: 100%;
}
</style>
