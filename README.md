# 🧬 Pokémon Generation using Variational Autoencoders (VAE) with Battle-Based Validation

This project explores the use of **Variational Autoencoders (VAEs)** to generate new Pokémon with realistic stat distributions and valid type combinations. It also introduces a novel **battle-based validation mechanism**, simulating combat between generated and real Pokémon to assess the generative quality.

---

## 🚀 Project Highlights

- **Model:** Variational Autoencoder (VAE)
- **Data:** Pokémon stats and types from Kaggle
- **Inputs:** HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Type 1, Type 2
- **Outputs:** 100 generated Pokémon with predicted stats and types
- **Validation:** Win rate from simulated battles vs. real Pokémon

---

## 📁 Dataset

- **Source:** [Kaggle - Pokémon with stats](https://www.kaggle.com/abcsds/pokemon)
- **Features Used:**
  - Numeric: `HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`
  - Categorical: `Type 1` (18 categories), `Type 2` (19 categories including `None`)

---

## 🧠 Model Architecture (VAE)

- **Input Dimension:** 6 stats + 18 Type1 + 19 Type2 = 43
- **Latent Space:** 16-dimensional multivariate Gaussian
- **Encoder:**
  - FC(43 → 64) → ReLU
  - Outputs: `μ`, `log(σ²)`
- **Reparameterization Trick:** `z = μ + σ * ε` with `ε ~ N(0, I)`
- **Decoder:**
  - FC(16 → 64) → ReLU
  - Outputs:
    - Stats head (6 outputs): Regressed with MSE
    - Type 1 head (18 classes): CrossEntropyLoss + Softmax
    - Type 2 head (19 classes): CrossEntropyLoss + Softmax
- **Loss Function:**
  - `Loss = MSE(stats) + CE(Type1) + CE(Type2) + KL Divergence`

---

##Sample Outcome

{
  "HP": 75.3,
  "Attack": 101.7,
  "Defense": 66.8,
  "Sp. Atk": 92.5,
  "Sp. Def": 78.1,
  "Speed": 88.6,
  "Type 1": "Fire",
  "Type 2": "Flying"
}

---

##Battle Simulation

- Each generated Pokémon battles a randomly chosen real Pokémon using a lightweight simulator based on:

- Type effectiveness chart

- Attack vs. Defense (physical or special, randomly selected)

- Speed-based turn priority

- Random damage variation (85%–100%)

---

## Sample Results

Validation Results (out of 100 battles):
Generated Pokémon Wins: 58 (58.00%)
Real Pokémon Wins:      39 (39.00%)
Draws:                   3 (3.00%)

---

##Challenges Faced

- Balancing KL Divergence: Too high led to stat collapse, too low caused overfitting.

- Ensuring diversity in Type 1/Type 2 outputs.

- Preventing generation of weak stats (e.g., HP < 5) or single-type bias (e.g., mostly Water/None).

- Proper stat scaling for decoding meaningful Pokémon stats.
  
---

##Key Contributions

- Custom deep generative model for structured Pokémon data

- One-hot encoding and classification handling for dual types

- Statistical and battle-based evaluation for generated samples

- Portable Colab notebook for reproducibility

---

##Future Improvements

- 🧠 Use Conditional VAE (CVAE) to guide generation (e.g., generate only Fire-type Pokémon)

- ⚔️ Multi-turn battle engine for deeper validation

- 🃏 Generate movesets and abilities using NLP datasets

- 🧬 Introduce adversarial training (VAE-GAN hybrid) for sharper stat realism



