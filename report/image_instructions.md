# Image Instructions for `sensor-selection/report/images/`

This document explains exactly which file from `sensor-selection/report_figures/` maps to each
image path referenced in `report.tex`, what each image shows, and how to copy or export it.

---

## Quick Copy Commands

Run these commands from the **repository root** (the folder that contains `sensor-selection/`):

```bash
cp sensor-selection/report_figures/01_building_map.png      sensor-selection/report/images/building_map.png
cp sensor-selection/report_figures/02_coverage_ga.png       sensor-selection/report/images/coverage_ga.png
cp sensor-selection/report_figures/02_coverage_sa.png       sensor-selection/report/images/coverage_sa.png
cp sensor-selection/report_figures/02_coverage_pso.png      sensor-selection/report/images/coverage_pso.png
cp sensor-selection/report_figures/02_coverage_saga.png     sensor-selection/report/images/coverage_saga.png
cp sensor-selection/report_figures/03_connectivity_ga.png   sensor-selection/report/images/connectivity_ga.png
cp sensor-selection/report_figures/04_convergence.png       sensor-selection/report/images/convergence.png
cp sensor-selection/report_figures/05_comparison_summary.png sensor-selection/report/images/comparison.png
cp sensor-selection/report_figures/06_pareto_front.png      sensor-selection/report/images/pareto.png
cp sensor-selection/report_figures/07_sensitivity.png       sensor-selection/report/images/sensitivity.png
```

You also need to supply the university logo manually (see the last section).

---

## File-by-File Reference

### `images/building_map.png`
- **Source file:** `sensor-selection/report_figures/01_building_map.png`
- **Content:** The 50×50 building floor plan showing all eight priority zones (colour-coded by
  weight), five rectangular obstacle walls (solid grey), and all 150 candidate sensor positions
  (colour-coded by type: motion = blue, thermal = orange, acoustic = green).
- **Used in:** Figure 1 (Section 4.2).
- **Recommended display width in LaTeX:** `0.85\textwidth`
- **Minimum resolution:** 150 DPI at the printed size (≈ 12 cm wide). The existing PNG from
  `report_figures/` is sufficient.

---

### `images/coverage_ga.png`
- **Source file:** `sensor-selection/report_figures/02_coverage_ga.png`
- **Content:** Weighted coverage heatmap for the Genetic Algorithm solution. Brighter cells
  indicate higher coverage weight. The 21 selected sensors are marked with crosses or dots.
  A colour bar on the right shows the coverage weight scale.
- **Used in:** Figure 2 (Section 4.3).
- **Recommended display width:** `0.80\textwidth`

---

### `images/coverage_sa.png`
- **Source file:** `sensor-selection/report_figures/02_coverage_sa.png`
- **Content:** Weighted coverage heatmap for the Simulated Annealing solution (24 sensors).
  Same format as `coverage_ga.png`.
- **Used in:** Figure 3 (Section 4.3).
- **Recommended display width:** `0.80\textwidth`

---

### `images/coverage_pso.png`
- **Source file:** `sensor-selection/report_figures/02_coverage_pso.png`
- **Content:** Weighted coverage heatmap for the Binary PSO solution (44 sensors).
  Same format as `coverage_ga.png`.
- **Used in:** Figure 4 (Section 4.3).
- **Recommended display width:** `0.80\textwidth`

---

### `images/coverage_saga.png`
- **Source file:** `sensor-selection/report_figures/02_coverage_saga.png`
- **Content:** Weighted coverage heatmap for the Hybrid SAGA solution (23 sensors).
  Same format as `coverage_ga.png`.
- **Used in:** Figure 5 (Section 4.3).
- **Recommended display width:** `0.80\textwidth`

---

### `images/connectivity_ga.png`
- **Source file:** `sensor-selection/report_figures/03_connectivity_ga.png`
- **Content:** Sensor connectivity graph for the GA solution overlaid on the building floor plan.
  Nodes represent the 21 selected sensors (colour-coded by type); edges connect pairs of sensors
  within the 15-unit communication range. The graph should be visually connected (single component).
- **Used in:** Figure 6 (Section 4.4).
- **Recommended display width:** `0.80\textwidth`
- **Note:** The `report_figures/` directory also contains connectivity graphs for SA, PSO, and SAGA
  (`03_connectivity_sa.png`, `03_connectivity_pso.png`, `03_connectivity_saga.png`). These are not
  included in the current report but can be added as additional figures if desired.

---

### `images/convergence.png`
- **Source file:** `sensor-selection/report_figures/04_convergence.png`
- **Content:** Convergence curves for all four algorithms on a single plot. The x-axis is the
  iteration/generation number (or a normalised iteration axis); the y-axis is the best fitness
  found so far (lower is better). Each algorithm is shown as a distinct coloured line. The SAGA
  curve should ideally be split into a solid segment (GA phase) and a dashed segment (SA phase).
- **Used in:** Figure 7 (Section 4.5).
- **Recommended display width:** `0.90\textwidth`

---

### `images/comparison.png`
- **Source file:** `sensor-selection/report_figures/05_comparison_summary.png`
- **Content:** Side-by-side bar charts comparing the four algorithms across key metrics: combined
  fitness, weighted coverage fraction, normalised cost, and number of selected sensors. Each
  metric is a separate panel or grouped bar cluster.
- **Used in:** Figure 8 (Section 4.6).
- **Recommended display width:** `0.90\textwidth`

---

### `images/pareto.png`
- **Source file:** `sensor-selection/report_figures/06_pareto_front.png`
- **Content:** Pareto front in the coverage–cost objective space. Each point is a non-dominated
  solution obtained by varying the weight vector. The four algorithm solutions (GA, SA, PSO, SAGA)
  are highlighted with distinct markers. Axes: x = normalised cost (f2), y = weighted coverage (f1).
- **Used in:** Figure 9 (Section 4.7).
- **Recommended display width:** `0.80\textwidth`

---

### `images/sensitivity.png`
- **Source file:** `sensor-selection/report_figures/07_sensitivity.png`
- **Content:** Three-panel sensitivity analysis figure.
  - **Left panel:** Coverage and cost vs. budget level (10%–100% of total cost).
  - **Centre panel:** Coverage and cost vs. coverage weight α (0.2–0.8).
  - **Right panel:** Coverage vs. number of candidate sensors N (25–175).
- **Used in:** Figure 10 (Section 5.1).
- **Recommended display width:** `0.95\textwidth`

---

### `images/iiit_logo.jpg`
- **Source file:** Not in `report_figures/` — must be obtained separately.
- **Content:** The official IIITDM Kancheepuram institutional logo (colour or greyscale).
- **Used in:** Title page.
- **How to obtain:**
  1. Download from the official IIITDM Kancheepuram website: https://www.iiitdm.ac.in
  2. Save as `sensor-selection/report/images/iiit_logo.jpg`
  3. Alternatively, use any `.png` version and change the filename in the `\includegraphics`
     command on the title page to `images/iiit_logo.png`.
- **Recommended size:** The logo is displayed at `0.22\textwidth` on the title page.
  A source image of at least 300×300 pixels is sufficient.
- **If the logo is unavailable:** Comment out or remove the `\includegraphics{images/iiit_logo.jpg}`
  line on the title page. The report will compile without it.

---

## Compiling the Report

Once all images are in place, compile with:

```bash
cd sensor-selection/report
pdflatex report.tex
pdflatex report.tex   # run twice for correct cross-references and TOC
```

Or with `latexmk`:

```bash
cd sensor-selection/report
latexmk -pdf report.tex
```

The output PDF will be `sensor-selection/report/report.pdf`.

---

## Checklist

| Image file                        | Source in `report_figures/`         | Copied? |
|-----------------------------------|--------------------------------------|---------|
| `images/building_map.png`         | `01_building_map.png`                | [ ]     |
| `images/coverage_ga.png`          | `02_coverage_ga.png`                 | [ ]     |
| `images/coverage_sa.png`          | `02_coverage_sa.png`                 | [ ]     |
| `images/coverage_pso.png`         | `02_coverage_pso.png`                | [ ]     |
| `images/coverage_saga.png`        | `02_coverage_saga.png`               | [ ]     |
| `images/connectivity_ga.png`      | `03_connectivity_ga.png`             | [ ]     |
| `images/convergence.png`          | `04_convergence.png`                 | [ ]     |
| `images/comparison.png`           | `05_comparison_summary.png`          | [ ]     |
| `images/pareto.png`               | `06_pareto_front.png`                | [ ]     |
| `images/sensitivity.png`          | `07_sensitivity.png`                 | [ ]     |
| `images/iiit_logo.jpg`            | Download from IIITDM website         | [ ]     |
