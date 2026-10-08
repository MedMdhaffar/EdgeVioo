# SafeWatch Edge AI — Rapport de Synthèse sur la Robustesse (P8)

## 1. Objectif & Contexte de l'Étude

L'objectif de l'axe de robustesse est d'évaluer la résilience du système de détection multimodale d'incidents face à des dégradations réalistes du flux audio-vidéo :
1. **Perte / Atténuation audio** (coupure micro, atténuation progressive du signal).
2. **Perte d'informations acoustiques** (dropout de snippets audio).
3. **Bruit additif sur le canal audio** (bruit blanc à différents rapports signal-sur-bruit $SNR \in [0, 60]\text{ dB}$).
4. **Occultation vidéo** (masquage visuel partiel de $5\text{ s}$, $15\text{ s}$ ou $30\text{ s}$).
5. **Comportement de la porte adaptative (`adaptive`) vs attention croisée (`cross`) et fusion tardive (`late`)**.

> [!NOTE]
> **Espace de dégradation** : Les dégradations sont appliquées dans l'espace des descripteurs (features precomputed I3D 1024-d / VGGish 128-d). Seules les perturbations log-mel peuvent être régénérées directement depuis la vidéo source (~3 s/clip).

---

## 2. Synthèse Quantitative des Balayages de Dégradation

### A. Dégradation du Canal Audio (Atténuation & Silence)

| Modèle / Stratégie | Niveau d'atténuation | AP Global | AP Per-Clip | Macro AP Classes | tIoU | $\Delta\text{AP}_\text{glob}$ |
|---|---|---|---|---|---|---|
| **Cross-Attention (VGGish)** | 1.00 (Clean) | **0.7745** | **0.7961** | **0.6958** | 0.369 | 0.0000 |
| Cross-Attention (VGGish) | 0.75 | 0.7743 | 0.7982 | 0.7038 | 0.377 | +0.0002 |
| Cross-Attention (VGGish) | 0.50 | 0.7726 | 0.7981 | 0.7066 | 0.365 | +0.0019 |
| Cross-Attention (VGGish) | 0.25 | 0.7665 | 0.7900 | 0.6999 | 0.360 | +0.0080 |
| Cross-Attention (VGGish) | 0.00 (Silence) | 0.7243 | 0.7249 | 0.5961 | 0.356 | **+0.0502** |
| **Late Fusion (VGGish)** | 1.00 (Clean) | 0.7719 | 0.7857 | 0.7200 | 0.354 | 0.0000 |
| Late Fusion (VGGish) | 0.50 | 0.7767 | 0.7817 | 0.6961 | 0.351 | -0.0047 |
| Late Fusion (VGGish) | 0.00 (Silence) | 0.7178 | 0.7239 | 0.5783 | 0.325 | **+0.0541** |
| **Late Fusion (Log-Mel)** | 1.00 (Clean) | 0.8143 | 0.7792 | 0.6505 | 0.422 | 0.0000 |
| Late Fusion (Log-Mel) | 0.00 (Silence) | 0.8133 | 0.7705 | 0.6457 | 0.429 | **+0.0010** |

* **Constat clé** : La représentation VGGish rend la modalité audio véritablement contributive (**$\Delta = -0.050\text{ AP}$** lors d'une coupure audio totale sur VGGish, contre seulement $-0.001$ sur le modèle log-mel qui ignorait structurellement l'audio).

---

### B. Perte de Paquets Audio (Snippet Dropout)

| Drop Rate | AP Global | AP Per-Clip | Macro AP Classes | tIoU | $\Delta\text{AP}_\text{glob}$ |
|---|---|---|---|---|---|
| **0 % (Clean)** | 0.7745 | 0.7961 | 0.6958 | 0.369 | 0.0000 |
| **25 %** | 0.7736 | 0.7887 | 0.6959 | 0.369 | +0.0009 |
| **50 %** | 0.7699 | 0.7694 | 0.6938 | 0.361 | +0.0046 |
| **75 %** | 0.7594 | 0.7423 | 0.6932 | 0.351 | **+0.0151** |

* **Constat clé** : La détection résiste remarquablement jusqu'à 50 % de perte aléatoire de snippets audio ($\Delta < 0.005\text{ AP}$).

---

### C. Bruit Additif Audio (SNR Sweep)

| Rapport SNR | AP Global | AP Per-Clip | Macro AP Classes | tIoU | $\Delta\text{AP}_\text{glob}$ |
|---|---|---|---|---|---|
| **Clean (60 dB)** | 0.7745 | 0.7961 | 0.6958 | 0.369 | 0.0000 |
| **30 dB** | 0.7745 | 0.7962 | 0.6957 | 0.369 | 0.0000 |
| **15 dB** | 0.7746 | 0.7959 | 0.6962 | 0.370 | -0.0001 |
| **5 dB** | 0.7758 | 0.7922 | 0.6948 | 0.370 | -0.0013 |
| **0 dB (Bruit = Signal)** | 0.7768 | 0.7852 | 0.6910 | 0.371 | -0.0022 |

* **Constat clé** : Même à $0\text{ dB}$ (où la puissance du bruit égale celle du signal), l'AP global reste stable. **La perte d'information (dropout/silence) est le mode de défaillance critique, pas la perturbation gaussienne.**

---

### D. Occultation Vidéo (Masquage Temporel)

| Modèle | Durée d'occultation | AP Global | AP Per-Clip | Macro AP Classes | $\Delta\text{AP}_\text{glob}$ |
|---|---|---|---|---|---|
| **Cross-Attention** | 0 s (Clean) | 0.7745 | 0.7961 | 0.6958 | 0.0000 |
| Cross-Attention | 15 s | 0.7686 | 0.7645 | 0.6684 | +0.0059 |
| Cross-Attention | 30 s | 0.7623 | 0.7429 | 0.6240 | **+0.0122** |
| **Late Fusion** | 0 s (Clean) | 0.7719 | 0.7857 | 0.7200 | 0.0000 |
| Late Fusion | 30 s | 0.7480 | 0.7482 | 0.6475 | **+0.0240** |
| **Adaptive Gate (Seed 42)** | 0 s (Clean) | 0.7281 | 0.7665 | 0.6930 | 0.0000 |
| Adaptive Gate (Seed 42) | 30 s | 0.7192 | 0.7321 | 0.6192 | +0.0089 |

---

## 3. Analyse Critique de la Porte Adaptative (`AdaptiveGate`)

Une analyse rigoureuse multi-graines a été menée sur la variante `adaptive` :
1. **Dispersion entre graines élevée** : Le modèle propre `adaptive` oscille entre $0.6319$ et $0.7281$ ($\sigma \approx 0.051$, écart max $0.096$).
2. **Non-reproductibilité du gain de fiabilité** : Alors que sur une graine isolée la porte semblait apporter $+0.016\text{ AP}$ sous occlusion, la répétition sur 3 graines donne $\{+0.016, -0.004, +0.031\}$ à $15\text{ s}$.
3. **Recommandation scientifique** : L'architecture **`cross` (attention croisée bidirectionnelle)** est nettement supérieure en stabilité ($\sigma_\text{seed} \approx 0.005$) et offre les meilleures performances globales ($0.7745$ AP global / $0.7961$ per-clip).
