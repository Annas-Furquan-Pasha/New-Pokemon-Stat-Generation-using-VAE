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

