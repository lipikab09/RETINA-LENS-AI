# Model Card: RETINA-LENS AI
**Intended use:** research/education on DR grading and lesion segmentation from colour fundus photographs. **Out of scope:** diagnosis, triage or any clinical decision.
**Training data:** configurable mix of IDRiD, DDR, FGADR, APTOS (see README; check each licence). **Metrics:** none reported until you run `evaluate`; cite `runs/<name>/metrics_test.json`.
**Known limitations:** dataset shift, camera/ethnicity/age coverage unknown, label noise, partial lesion taxonomy, calibration only valid in-distribution, Grad-CAM is not localisation.
**Ethics:** do not upload identifiable patient data to a public deployment; abstention flags are advisory.
