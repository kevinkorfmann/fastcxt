Figure Gallery
==============

Publication-quality figures from the comprehensive Ag1000G *Anopheles gambiae*
analysis. All plots are auto-generated from inference results across 16
African populations, 5 chromosome arms, 3 karyotype groups, and both intra-
and inter-individual pair types.

.. admonition:: Live results
   :class: tip

   These figures update automatically as inference completes. Sections marked
   **PENDING** will populate as more populations finish processing. Re-run
   ``generate_all_figures.py`` after syncing new results.


Overview
--------

Cross-population summary of mean coalescence times across all karyotype
groups and pair types.

.. image:: _static/gallery/heatmap_all_vs_all.png
   :width: 100%
   :alt: Overview heatmap — mean TMRCA across populations and groups


Inversion Signals
-----------------

Intra-individual coalescence along chromosome 2L reveals the In(2L)a
inversion signature. Heterozygous individuals show deep coalescence inside
the inversion where recombination is suppressed between standard and inverted
arrangements. Homozygotes recombine freely within their arrangement.

.. image:: _static/gallery/2L_Burkina_Faso.png
   :width: 100%
   :alt: Burkina Faso 2L inversion signal by karyotype

*Burkina Faso — three karyotype groups overlaid. The red shaded region marks
In(2L)a (20.5–42.2 Mb). Accessibility track at bottom shows data quality.*

.. image:: _static/gallery/2L_Cameroon.png
   :width: 100%
   :alt: Cameroon 2L inversion signal by karyotype

*Cameroon — same three karyotype groups. Note similar inversion signature.*

.. image:: _static/gallery/2L_Central_African_Republic.png
   :width: 100%
   :alt: Central African Republic 2L inversion signal by karyotype

*Central African Republic — 2L inversion signal (2L data only, 2R in progress).*

.. raw:: html

   <details style="margin: 1rem 0; padding: 0.5rem; background: #1e293b; border: 1px solid #334155; border-radius: 8px;">
   <summary style="cursor: pointer; color: #60a5fa; font-weight: bold;">Other populations (click to expand)</summary>
   <p style="color: #64748b; padding: 1rem;">PENDING — will populate as inference completes for the remaining 13 populations.</p>
   </details>


Outlier Skylines
----------------

Candidate genomic regions with extreme coalescence times, filtered by
accessibility to avoid false positives from missing data. Gene annotations
from VectorBase AgamP4 are overlaid using ``adjustText`` for non-overlapping
labels.

.. image:: _static/gallery/Burkina_Faso_2L_2La_heterozygous.png
   :width: 100%
   :alt: Outlier skyline — 2L heterozygous intra

*Burkina Faso, 2L, 2La-heterozygous, intra-individual. Blue = young credible
outliers (accessible regions only), red = old credible, gray x = suspect
(low accessibility). Bottom panels: Z-score and accessibility fraction.*

.. image:: _static/gallery/Burkina_Faso_2L_2La_hom_inverted.png
   :width: 100%
   :alt: Outlier skyline — 2L hom inverted intra

*Same for homozygous inverted. Note the deep dip near Rdl (28.5 Mb) and
CDSB21 — potential selection signatures within the inverted arrangement.*

.. image:: _static/gallery/Cameroon_2L_2La_heterozygous.png
   :width: 100%
   :alt: Outlier skyline — Cameroon 2L heterozygous intra

*Cameroon, 2L, 2La-heterozygous, intra-individual.*

.. image:: _static/gallery/Cameroon_2R_2Rb_hom_standard.png
   :width: 100%
   :alt: Outlier skyline — Cameroon 2R hom standard intra

*Cameroon, 2R, 2Rb-hom-standard, intra-individual.*


Karyotype Comparisons
---------------------

Box plots comparing block-level coalescence distributions across karyotype
groups and pair types. Heterozygous intra pairs show elevated TMRCA on
inversion-bearing chromosomes (2L, 2R), while 3L/3R show uniform patterns.

.. image:: _static/gallery/Burkina_Faso.png
   :width: 100%
   :alt: Burkina Faso karyotype comparison boxplots


Chromosome-Wide Profiles
------------------------

TMRCA profiles across all chromosome arms with per-arm accessibility tracks.

.. image:: _static/gallery/Burkina_Faso_all_arms.png
   :width: 100%
   :alt: Burkina Faso all arms profile

*Each column is one chromosome arm (2L, 2R, 3L, 3R, X). Karyotype groups
overlaid by color. Red shading on 2L marks In(2L)a.*


Density Distributions
---------------------

KDE density overlays showing the distribution of block-level coalescence
times per karyotype group for each arm.

.. image:: _static/gallery/Burkina_Faso_density_overlay.png
   :width: 100%
   :alt: Burkina Faso density overlay

.. image:: _static/gallery/density_grid_intra.png
   :width: 100%
   :alt: Density grid — all populations x groups


Demographic Inference
---------------------

Inverse instantaneous coalescence rates (IICR), a proxy for effective
population size Ne(t), estimated from the TMRCA distribution and compared
to the stdpopsim *A. gambiae* Gabon reference model (``GabonAg1000G_1A17``).

.. image:: _static/gallery/demography_2L_Burkina_Faso.png
   :width: 100%
   :alt: Burkina Faso 2L IICR by karyotype

*Burkina Faso 2L: IICR by karyotype. The heterozygous curve (orange) is
inflated at intermediate times due to deep coalescence inside the inversion.
Dashed black = stdpopsim Gabon reference.*

.. image:: _static/gallery/demography_allarms_Burkina_Faso.png
   :width: 100%
   :alt: Burkina Faso IICR all arms

*All chromosome arms overlaid. 3L and 3R (no inversions) track the stdpopsim
reference more closely, providing a cleaner demographic signal.*

.. image:: _static/gallery/demography_allarms_Cameroon.png
   :width: 100%
   :alt: Cameroon IICR all arms

*Cameroon — all chromosome arms overlaid.*

.. image:: _static/gallery/demography_3L_all_allpops.png
   :width: 100%
   :alt: Cross-population IICR comparison 3L

*Cross-population IICR comparison on 3L (no inversions) — 9 completed
populations overlaid.*


Geographic Maps
---------------

Coalescence patterns projected onto Africa using population coordinates
from the Ag3 metadata.

.. image:: _static/gallery/map_2L_bubbles.png
   :width: 100%
   :alt: Geographic bubble map

*Bubble size = sample count, color = mean log-TMRCA. Three panels for
hom standard, heterozygous, and hom inverted karyotypes on 2L.*

.. image:: _static/gallery/map_2L_sparklines.png
   :width: 100%
   :alt: Geographic sparkline map

*TMRCA profiles embedded as sparklines at each population's location.
Orange lines show coalescence along 2L, red shading = In(2L)a region.*

.. image:: _static/gallery/map_2La_inversion_effect.png
   :width: 100%
   :alt: Inversion effect map

*How much deeper is coalescence inside vs outside In(2L)a for each
population. Positive values = inversion creates deeper coalescence.*


Progress & Pending Figures
--------------------------

**Completed populations (9/16):** Burkina Faso, Cameroon, Central African
Republic, Democratic Republic of the Congo, Equatorial Guinea, Gabon,
Gambia, Ghana, Guinea (all 5 arms each).

**Currently running:** Guinea-Bissau on poppy (7 populations remaining).

The following will be generated as inference completes:

.. raw:: html

   <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
     <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
       <span style="color: #64748b; font-style: italic;">PENDING</span><br/>
       <span style="color: #94a3b8;">Per-population inversion signals (7 remaining)</span>
     </div>
     <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
       <span style="color: #64748b; font-style: italic;">PENDING</span><br/>
       <span style="color: #94a3b8;">Full cross-population IICR comparison (all 16 overlaid)</span>
     </div>
     <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
       <span style="color: #64748b; font-style: italic;">PENDING</span><br/>
       <span style="color: #94a3b8;">Geographic maps with all populations filled in</span>
     </div>
     <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
       <span style="color: #64748b; font-style: italic;">PENDING</span><br/>
       <span style="color: #94a3b8;">Full density grid with all populations (16 rows x 9 cols)</span>
     </div>
   </div>
