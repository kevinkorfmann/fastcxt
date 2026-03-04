Changelog
=========


v0.1.0 (2025–2026)
-------------------

Initial release of fastcxt.

**Architecture**

- Bidirectional Mamba encoder-decoder architecture
- Single-pass inference with built-in uncertainty (Gaussian NLL)
- FiLM conditioning on mutation rate
- Variable sample size support via InputProjection
- Optional tree topology integration for O(n log n) scaling

**Simulation & preprocessing**

- stdpopsim-first simulation pipeline with scenario registry
- 12 stdpopsim species + 3 custom msprime scenarios
- Accessibility mask support for missing data
- Clean PreprocessJob-based preprocessing pipeline

**Applications**

- Dedicated *Anopheles gambiae* analysis protocol (``fastcxt.mosquito``)
- TimeAtlas data structure for genome-wide TMRCA storage and queries
- End-to-end Ag1000G strategy document (``docs/ag1000g_strategy.md``)

**Visualization**

- Geographic collection site maps (Cartopy projections)
- Population connectivity arcs colored by between-population TMRCA
- Genome-wide TMRCA landscape across all chromosome arms
- Population-level TMRCA heatmap with geographic inset
- Multi-panel selective sweep analysis (Rdl locus on chr2L)
- Dense pairwise TMRCA raster heatmaps
- Composite dashboard combining all panels
- Dark-theme, publication-quality styling throughout

**Infrastructure**

- Centralized cluster path configuration (``fastcxt.paths``)
- ``uv`` / hatchling build system
- Comprehensive Sphinx documentation with Furo theme
- 110+ unit and integration tests
