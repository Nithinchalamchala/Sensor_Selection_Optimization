# Streamlit Web App — Deployment Guide

## Overview

The Streamlit app provides an **interactive web interface** for the Sensor Selection Problem, allowing you to:

- 🎛️ **Configure parameters** (sensors, budget, weights) via sliders
- 🚀 **Run algorithms** (GA, SA, PSO, SAGA) with one click
- 📊 **Visualize results** in real-time (coverage heatmaps, convergence curves)
- 📈 **Explore Pareto fronts** interactively
- 💾 **Export results** as CSV/JSON

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit pandas numpy matplotlib networkx scipy
```

### 2. Run the app locally

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## App Features

### Tab 1: 🏢 Environment
- View the building floor plan with zones, obstacles, and candidate sensors
- See zone details (priority, critical status, area)
- Sensor type distribution chart

### Tab 2: 🎯 Run Algorithms
- Select which algorithms to run (GA, SA, PSO, SAGA)
- Click "Run Selected Algorithms" to execute
- Real-time progress bar and status updates

### Tab 3: 📊 Results
- Performance summary table comparing all algorithms
- Convergence curves (fitness and coverage over iterations)
- Interactive solution visualization:
  - Coverage heatmap with selected sensors
  - Detailed metrics (coverage, cost, overlap, feasibility)
  - Constraint satisfaction status

### Tab 4: 📈 Pareto Front
- Generate Pareto-optimal solutions by varying objective weights
- Interactive scatter plot showing coverage vs cost tradeoff
- Table of all Pareto-optimal points

### Tab 5: 💾 Export
- Download summary results as CSV
- Download convergence histories for each algorithm
- Download Pareto data as JSON
- View configuration summary

---

## Configuration Options (Sidebar)

### Environment
- **Number of candidate sensors**: 50–200 (default: 150)
- **Random seed**: For reproducibility

### Objectives
- **α (Coverage weight)**: 0.0–1.0 (default: 0.60)
- **β (Cost weight)**: 0.0–1.0 (default: 0.25)
- **γ (Overlap penalty)**: 0.0–1.0 (default: 0.15)

Weights are automatically normalized to sum to 1.0.

### Budget
- **Budget (% of total cost)**: 20–100% (default: 50%)

### Algorithms
- ☑️ Genetic Algorithm (GA)
- ☑️ Simulated Annealing (SA)
- ☑️ Binary PSO
- ☐ Hybrid SAGA (optional, slower)

### Performance
- **Quick mode**: Faster execution with reduced iterations (recommended for testing)

---

## Deployment Options

### Option 1: Local Development (Already Running)

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### Option 2: Streamlit Community Cloud (Free)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Sensor selection optimizer"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `sensor-selection/app.py`
   - Click "Deploy"

Your app will be live at: `https://<your-username>-<repo-name>.streamlit.app`

### Option 3: Docker Container

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t sensor-optimizer .
docker run -p 8501:8501 sensor-optimizer
```

### Option 4: Heroku

Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

Deploy:
```bash
heroku create your-app-name
git push heroku main
```

---

## Performance Tips

### Quick Mode vs Full Mode

| Mode | GA Generations | SA Iterations | PSO Iterations | Time (N=150) |
|------|----------------|---------------|----------------|--------------|
| Quick | 100 | 10,000 | 100 | ~30 seconds |
| Full | 300 | 50,000 | 300 | ~5 minutes |

**Recommendation**: Use Quick mode for parameter exploration, Full mode for final results.

### Sensor Count Impact

| N Sensors | Environment Build | Algorithm Time | Total Time |
|-----------|-------------------|----------------|------------|
| 50 | <1s | ~10s | ~10s |
| 100 | ~1s | ~30s | ~30s |
| 150 | ~2s | ~60s | ~1min |
| 200 | ~3s | ~120s | ~2min |

### Pareto Front Generation

- Each weight configuration runs a separate GA
- Time = `n_weight_steps × GA_time`
- Recommended: 6–8 steps for good coverage

---

## Troubleshooting

### App won't start
```bash
# Check Streamlit version
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Slow performance
- Enable Quick mode
- Reduce number of sensors
- Run fewer algorithms simultaneously
- Close other browser tabs

### Plots not showing
```bash
# Ensure matplotlib backend is set
export MPLBACKEND=Agg
streamlit run app.py
```

---

## Advanced Configuration

### Custom Streamlit Config

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false
```

### Environment Variables

```bash
# Set default parameters
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## API for Programmatic Access

While the app is interactive, you can also use the core modules programmatically:

```python
from environment import BuildingEnvironment
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm

# Setup
env = BuildingEnvironment(n_sensors=150, seed=42)
ev = FitnessEvaluator(env, budget=env.budget_default())

# Run GA
ga = GeneticAlgorithm(ev, pop_size=100, n_generations=300)
result = ga.run(verbose=True)

# Access results
print(f"Best fitness: {result['best_fitness']}")
print(f"Coverage: {result['best_info']['f1_coverage']}")
```

---

## Screenshots

### Main Dashboard
![Dashboard](docs/screenshot_dashboard.png)

### Coverage Visualization
![Coverage](docs/screenshot_coverage.png)

### Pareto Front
![Pareto](docs/screenshot_pareto.png)

---

## Support

For issues or questions:
- Check the main [README.md](README.md)
- Review algorithm documentation in individual `.py` files
- Open an issue on GitHub

---

## License

This project is for educational purposes as part of an Optimization Techniques course project.
