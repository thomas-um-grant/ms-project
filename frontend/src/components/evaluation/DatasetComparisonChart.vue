<template>
  <div ref="root" class="chart"></div>
</template>

<script setup>
import * as d3 from "d3";
import { ref, onMounted, onBeforeUnmount, watch } from "vue";
import { themeService } from "../../services/ThemeService";

const props = defineProps({
  data: { type: Object, required: true }, // Profile data with all datasets
  metric: { type: String, required: true }, // e.g., 'ndcg_at_10'
  title: { type: String, required: true },
});

const root = ref(null);
let resizeObserver;
let themeUnsubscribe;

function render() {
  if (!root.value || !props.data) return;
  const el = root.value;
  el.innerHTML = "";

  // Increased margins (bottom for rotated labels, top for title spacing)
  const margin = { top: 50, right: 20, bottom: 110, left: 70 };
  const rect = el.getBoundingClientRect();
  const W = rect.width || 800;
  const H = 400;
  const width = W - margin.left - margin.right;
  const height = H - margin.top - margin.bottom;

  const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);
  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Extract datasets and their metric values
  const datasets = Object.keys(props.data);
  let datasetValues = datasets
    .map((dataset) => ({
      dataset: dataset
        .replace("vidore/", "")
        .replace("_test_subsampled_beir", "")
        .replace("_test_beir", ""),
      value: props.data[dataset][props.metric] || 0,
      fullName: dataset,
    }))
    .filter((d) => d.value > 0);

  // Sort ascending by metric value so bars are always ranked
  datasetValues.sort((a, b) => a.value - b.value);

  if (datasetValues.length === 0) return;

  // Scales
  const x = d3
    .scaleBand()
    .domain(datasetValues.map((d) => d.dataset))
    .range([0, width])
    .padding(0.1);

  const yMax = d3.max(datasetValues, (d) => d.value) || 1;
  const y = d3
    .scaleLinear()
    .domain([0, Math.max(1, yMax)])
    .nice()
    .range([height, 0]);

  // Color scale - modern gradient colors
  const colorScale = d3
    .scaleSequential()
    .domain([0, datasetValues.length - 1])
    .interpolator(d3.interpolateRgb("#3b82f6", "#8b5cf6"));

  const colors = datasetValues.map((_, i) => colorScale(i));

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
    .style("text-anchor", "end")
    .attr("dx", "-.8em")
    .attr("dy", ".15em")
    .attr("transform", "rotate(-45)")
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

  // Bars with modern styling
  g.selectAll(".bar")
    .data(datasetValues)
    .enter()
    .append("rect")
    .attr("class", "bar")
    .attr("x", (d) => x(d.dataset))
    .attr("width", x.bandwidth())
    .attr("y", (d) => y(d.value))
    .attr("height", (d) => height - y(d.value))
    .attr("fill", (d, i) => colors[i])
    .attr("rx", 4)
    .attr("ry", 4)
    .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))")
    .append("title")
    .text((d) => `${d.dataset}: ${d3.format(".3f")(d.value)}`);

  // Value labels with modern styling - get theme-aware colors
  const primaryTextColor =
    getComputedStyle(el).getPropertyValue("--color-text-primary").trim() ||
    "#374151";

  g.selectAll(".label")
    .data(datasetValues)
    .enter()
    .append("text")
    .attr("class", "label")
    .attr("x", (d) => x(d.dataset) + x.bandwidth() / 2)
    .attr("y", (d) => y(d.value) - 8)
    .attr("text-anchor", "middle")
    .style("font-size", "11px")
    .style("font-weight", "600")
    .style("fill", primaryTextColor)
    .text((d) => d3.format(".3f")(d.value));

  // Modern title styling
  g.append("text")
    .attr("x", width / 2)
    .attr("y", -20)
    .attr("text-anchor", "middle")
    .style("font-size", "16px")
    .style("font-weight", "600")
    .style("fill", primaryTextColor)
    .text(props.title);

  // Modern Y-axis label
  g.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left)
    .attr("x", 0 - height / 2)
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .style("font-size", "12px")
    .style("font-weight", "500")
    .style("fill", textColor)
    .text(props.metric.replace(/_/g, " ").toUpperCase());
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
watch(() => [props.data, props.metric], render, { deep: true });
</script>

<style scoped>
.chart {
  width: 100%;
}
</style>
