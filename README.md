\# Chaos-Modulated Spiking Neural Networks for Seizure Detection



\*\*Official implementation of the study:\*\*

\*\*“Composite EEG Biomarker Modeling and Energy-Accuracy Trade-Off Analysis for Multi-Patient Seizure Detection”\*\*



\---



\## 🧠 Overview



This repository presents the \*\*Chaos-SNN framework\*\*, which integrates controlled nonlinear dynamics into spiking neural networks (SNNs) for:



\* Interpretable EEG analysis

\* Energy-efficient neuromorphic computation

\* Cross-patient seizure detection



The approach combines \*\*chaotic modulation + reservoir-based SNNs\*\* to model complex neural dynamics in clinical EEG signals.



\---



\## 📂 Repository Structure



\### 🧠 Core Architecture



\* `src/models/chaos\_snn.py`

&#x20; → Implements:



&#x20; \* `ChaosModulator` (nonlinear coupling dynamics)

&#x20; \* `RecurrentLIFReservoir` (spiking computation core)



\* `src/utils/chaos\_utils.py`

&#x20; → Vectorized NumPy implementation of chaos modulation



\---



\### 🚄 Research Pipelines



\* `src/run\_chbmit\_full\_pipeline.py`

&#x20; → Full CHB-MIT pipeline including hypothesis testing (H1–H4)



\* `src/run\_chbmit\_lopo.py`

&#x20; → Leave-One-Patient-Out (LOPO) cross-validation



\* `src/run\_segment\_experiments.py`

&#x20; → Multi-dataset experiments:



&#x20; \* Bonn

&#x20; \* Bern

&#x20; \* Panwar

&#x20; \* Hauz Khas



\---



\### 📊 Metrics \& Interpretability



\* `src/analyze\_bonn\_full\_complexity\_heatmap.py`

&#x20; → Generates \*\*ISI entropy heatmaps (Neuron × Time Window)\*\*



\* `src/utils/diagnose\_preds\_npz.py`

&#x20; → Computes:



&#x20; \* AUC

&#x20; \* G-Mean

&#x20; \* Confusion matrices



\* `src/utils/consolidate\_all\_results\_v2.py`

&#x20; → Aggregates results into manuscript-ready tables



\---



\## 🚀 Quick Start



Run a quick sanity check using synthetic data:



```bash

python train\_dummy.py

```



\---



\## 🛠 Installation



Install required dependencies:



```bash

pip install torch numpy scipy scikit-learn tqdm pandas matplotlib seaborn

```



\---



\## 📈 Key Results



This implementation reproduces the core findings of the study:



\* \*\*ANN Peak Accuracy:\*\* 90.64% (intra-dataset)

\* \*\*SNN Efficiency:\*\* 69.15% accuracy with \*\*7.01% spike sparsity\*\*

\* \*\*Statistical Validation:\*\*



&#x20; \* ISI entropy differentiation

&#x20; \* Mann–Whitney U test (\*\*p < 0.001\*\*)



\---



\## ⚡ Research Highlights



\* Chaos-driven modulation improves \*\*temporal feature richness\*\*

\* Reservoir SNN enables \*\*low-power inference\*\*

\* Interpretable biomarkers via \*\*entropy-based analysis\*\*



\---



\## 📌 Notes



\* Large datasets are excluded from this repository

\* Use external storage for EEG datasets (e.g., CHB-MIT)

\* Ensure proper preprocessing before running pipelines



\---





