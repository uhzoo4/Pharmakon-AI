## UI Pro Max Search Results
**Domain:** chart | **Query:** severity real-time dashboard network monitoring
**Source:** charts.csv | **Found:** 4 results

### Result 1
- **Data Type:** Real-Time Streaming
- **Keywords:** streaming, real-time, ticker, live, velocity, pulse, monitoring
- **Best Chart Type:** Streaming Area Chart
- **Secondary Options:** Ticker Tape, Moving Gauge
- **When to Use:** Live monitoring dashboards; IoT/ops data updating at ≥1 Hz; user needs current value at a glance
- **When NOT to Use:** Update frequency < 1/min (use periodic-refresh line chart); flashing content without reduced-motion support
- **Data Volume Threshold:** Canvas/WebGL required. Buffer last 60–300s of data. Downsample older data on scroll
- **Color Guidance:** Current pulse: #00FF00 (dark theme) or #0080FF (light theme). History: fading opacity. Grid: dark background
- **Accessibility Grade:** B
- **Accessibility Notes:** Pause/resume control required. Current value as large visible text KPI. Respect prefers-reduced-motion.
- **A11y Fallback:** Pause/resume button required; current value shown as large text KPI; prefers-reduced-motion: freeze animation
- **Library Recommendation:** Smoothed D3.js, CanvasJS
- **Interactive Level:** Real-time + Pause + Zoom

### Result 2
- **Data Type:** Anomaly Detection
- **Keywords:** anomaly, outlier, spike, alert, detection, monitoring, deviation
- **Best Chart Type:** Line Chart with Highlights
- **Secondary Options:** Scatter with Alert
- **When to Use:** Monitoring a time-series for outliers; alerting users to unexpected spikes or dips in operational data
- **When NOT to Use:** Anomalies are predefined categories (use bar with highlight); real-time context without a pause control
- **Data Volume Threshold:** Stream at ≤60fps with Canvas; batch: up to 10,000 pts; mark anomalies as a separate data layer
- **Color Guidance:** Normal: #0080FF solid line. Anomaly marker: #FF0000 circle + filled. Alert band: #FFF3CD background zone
- **Accessibility Grade:** AA
- **Accessibility Notes:** Use shape marker (not color only) for anomaly points. Add text annotation per anomaly event.
- **A11y Fallback:** Text alert annotation per anomaly; anomaly summary list panel alongside chart
- **Library Recommendation:** D3.js, Plotly, ApexCharts
- **Interactive Level:** Hover + Alert

### Result 3
- **Data Type:** Relationship / Connection Data
- **Keywords:** network, graph, nodes, edges, connections, relationships, force
- **Best Chart Type:** Network Graph
- **Secondary Options:** Hierarchical Tree, Adjacency Matrix
- **When to Use:** Mapping connections between entities; network topology or social graph exploration context
- **When NOT to Use:** Node count > 500 without clustering pre-applied; user needs precise connection counts; mobile context
- **Data Volume Threshold:** ≤100 nodes: SVG; 101–500: Canvas; >500: must apply clustering/LOD before rendering
- **Color Guidance:** Node types: categorical colors. Edges: #90A4AE at 60% opacity. Highlight path: #F59E0B
- **Accessibility Grade:** D
- **Accessibility Notes:** Fundamentally inaccessible without alternative. Never use as sole representation. Always provide list alternative.
- **A11y Fallback:** Adjacency list table (Node A → Node B → Weight); hierarchical tree view when structure allows
- **Library Recommendation:** D3.js (d3-force), Vis.js, Cytoscape.js
- **Interactive Level:** Drilldown + Hover + Drag

### Result 4
- **Data Type:** Performance vs Target (Compact)
- **Keywords:** bullet, compact, kpi, dashboard, target, benchmark, range
- **Best Chart Type:** Bullet Chart
- **Secondary Options:** Gauge, Progress Bar
- **When to Use:** Dashboard with multiple KPIs side by side; space-constrained contexts where a gauge is too large
- **When NOT to Use:** Single KPI with emphasis (use gauge); data has no defined target range; fewer than 3 KPIs
- **Data Volume Threshold:** Ideal for 3–10 bullet charts in a grid; scales to any count efficiently
- **Color Guidance:** Qualitative ranges: #FFCDD2 / #FFF9C4 / #C8E6C9 (bad/ok/good). Performance bar: #1976D2. Target: black 3px marker
- **Accessibility Grade:** AAA
- **Accessibility Notes:** All values always visible as text. Color ranges are labeled with text thresholds not color alone.
- **A11y Fallback:** Numerical values always visible (not hover-only); color ranges labeled with threshold text
- **Library Recommendation:** D3.js, Plotly, Custom SVG
- **Interactive Level:** Hover

