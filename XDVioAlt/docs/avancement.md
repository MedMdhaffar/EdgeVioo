# SafeWatch Edge AI - brief d'avancement (jury / réunion)

> One page. What is built, what it measures, what is missing. Every number here comes from a file in
> `runs/` (no number is quoted from memory). Deeper detail: `docs/journal.md` (chronology),
> `docs/protocol.md` (rules and comparability), `docs/theory_notes.md` (the theory used).

## 1. Le sujet, en une phrase

Détecter, catégoriser, localiser et **expliquer** des incidents dans un flux audio-vidéo, en
apprentissage faiblement supervisé (annotations au niveau clip), et tenir sur une plateforme Edge.
Benchmark : **XD-Violence** (4 754 vidéos, 217 h, 6 catégories).

## 2. Chaîne technique actuelle (ce qui tourne)

```
vidéo ──► features visuelles gelées (VideoSwin 768-d / I3D officiel 1024-d, grille 64 ou 16 frames)
      └► audio ──► ffmpeg mono 16 kHz ──► log-mel patch stats (132-d) + qualité du signal (5-d)
                                             │
      encodeur temporel ──► attention temporelle (MIL top-k) ──► score par snippet + score clip
                                             │                     + 6 logits de catégorie
      fusion : early | late | cross-attention | adaptive (portail α(t) piloté par 7 canaux
                                                       de fiabilité : 5 audio + 2 visuels)
                                             │
      évaluation : les deux protocoles AP × les deux agrégations (global / per-clip),
                   F1 segmentaire @tIoU 0.1-0.5, meilleur IoU moyen, délai de détection,
                   AP par catégorie, audit hors-source, ablation de modalité
```

## 3. Résultats mesurés (split de test officiel, 800 clips)

### 3.1 Référence visuelle seule (même liste `data/lists`)

| modèle | features | AP global | AP per-clip | meilleur IoU moy. | F1@IoU0.5 |
|---|---|---|---|---|---|
| visuel seul | VideoSwin 768-d, grille 64 | **0.810** | 0.771 | 0.433 | 0.322 |
| visuel seul (30 epochs) | I3D officiel 1024-d, grille 16 | 0.726 | 0.715 | 0.335 | 0.225 |

Le backbone officiel est **nettement plus faible** que le VideoSwin du miroir (0.726 contre 0.810) : c'est
ce qui plafonne tous les chiffres de la grille officielle ci-dessous, et il faut le dire en même temps
que le gain multimodal. Règle de localisation exprimée **en secondes** (min 5.33 s / trous 2.67 s) :
la même règle en snippets vaut 4× plus fin sur la grille 16 frames et fabrique une fausse chute de
l'IoU (piège documenté, `docs/journal.md` session 10).

### 3.2 Audio seul

| modèle | features | AP global | AP per-clip | Règle segmentaire |
|---|---|---|---|---|
| audio seul (miroir) | log-mel patch stats 132-d | 0.446 | 0.625 | min 5.3 s / trous 2.7 s |
| audio seul (officiel) | **VGGish officiel 128-d** | **0.640** | **0.699** | min 5.33 s / trous 2.67 s |

Lecture honnête : sur le mel 132-d, l'audio discrimine **à l'intérieur** d'un clip mais ses scores
comparables **entre** clips le sont mal (0.45 global contre 0.625 per-clip). Le VGGish officiel, lui,
réduit fortement cet écart (0.640 contre 0.699) : **une partie du défaut était bien la représentation
audio**, et non la calibration par clip (dont l'échec est mesuré en §3.3).

### 3.3 Fusion — échelle d'ablation complète (30 epochs par étage)

**Grille officielle (I3D 1024-d + VGGish 128-d), jeu de test identique (800 clips) :**

| run | AP global | AP per-clip | IoU moy. | F1@0.5 | −audio | −vidéo |
|---|---|---|---|---|---|---|
| `p2vggish30ep_visual` (**référence visuelle**) | 0.7259 | 0.7148 | 0.335 | 0.225 | — | — |
| `p2vggish30ep_audio` (audio seul) | 0.6403 | 0.6986 | 0.337 | 0.201 | — | — |
| **`p3vggish30ep_cross`** | **0.7745** | **0.7961** | **0.369** | 0.243 | 0.7243 | 0.6563 |
| `p3vggish30ep_late` | 0.7719 | 0.7857 | 0.354 | **0.265** | 0.7178 | 0.6445 |
| `p3vggish30ep_early` | 0.7643 | 0.7830 | 0.351 | 0.260 | 0.7141 | 0.6514 |
| `p3vggish30ep_adaptive` (portail) | 0.7281 | 0.7665 | 0.228 | 0.180 | 0.7087 | 0.5891 |

**Le gain multimodal est enfin franc et audité** : la fusion dépasse **sa propre** référence visuelle de
**+0.0486 en AP global et +0.0813 en AP per-clip** (≈20× et 7× les marges obtenues sur le miroir avec le
mel 132-d), et les deux modalités deviennent **nécessaires** (retirer l'audio coûte −0.050, retirer la
vidéo −0.118). L'audit hors-source confirme (voir §3.4). Réserve : l'AP global absolu reste sous le
miroir (0.7745 contre 0.8125) parce que le backbone visuel officiel est plus faible — comparaison à
**backbone égal**, pas inter-grilles.

**La revendication principale est répliquée (3 seeds).** Deux seeds supplémentaires pour `cross` et
pour le visuel seul, appariés par seed : marges **+0.0486 / +0.0396 / +0.0417** (moyenne **+0.0433**),
avec une dispersion de seed de **0.0050** (`cross`) et **0.0065** (visuel) ⇒ la marge vaut **≈6.7×** la
dispersion la plus large. L'AP per-clip de `cross` est à citer comme **0.792 ± 0.006** (plage
0.7847-0.7964), pas comme le meilleur point 0.7961. Outil : `scripts/seed_margin.py`.

**Grille miroir (VideoSwin 768-d + log-mel 132-d), pour mémoire :**

| run | AP global | AP per-clip | IoU moy. | F1@0.5 | −audio | −vidéo |
|---|---|---|---|---|---|---|
| `p1_visual_swin` (référence visuelle) | 0.8100 | 0.7709 | **0.433** | **0.322** | — | — |
| `p3full_early` | 0.8015 | 0.7818 | 0.416 | 0.310 | 0.8138 | 0.4621 |
| **`p3full_late`** | **0.8125** | **0.7825** | 0.423 | 0.306 | 0.8133 | 0.3769 |
| `p3full_cross` | 0.7981 | 0.7613 | 0.402 | 0.208 | 0.7946 | 0.4281 |
| `p3full_adaptive` | 0.7553 | 0.7462 | 0.382 | 0.211 | 0.7731 | 0.4185 |
| `p3full_late_clipnorm` | 0.8135 | 0.7703 | 0.426 | 0.314 | 0.8172 | 0.2637 |
| `p3full_adaptive_clipnorm` | 0.7668 | 0.7520 | 0.422 | 0.210 | 0.7788 | 0.3062 |
| `p2_audio_full` (audio seul) | 0.4464 | 0.6252 | 0.302 | 0.183 | 0.2463 | — |

**Trois conclusions à assumer devant le jury :**

1. **La promesse est tenue** : sur la grille officielle (VGGish), la fusion dépasse la référence visuelle
   de +0.049 global / +0.081 per-clip, et **retirer une modalité coûte** (−0.050 / −0.118). Sur le miroir
   (mel 132-d) la marge n'était que de +0.0025 / +0.0116 : l'écart entre les deux grilles est le résultat
   le plus intéressant du projet.
2. **La piste « sophistiquée » ne paie pas** : le portail de fiabilité (`adaptive`) est **dernier** sur les
   deux grilles et une troisième fois sur la grille officielle ; l'attention croisée est correcte mais
   n'apporte rien de décisif. À ne pas présenter comme la contribution du projet.
3. **La correction de calibration par clip ne marche pas** (résultat négatif honnête) : elle coûte de l'AP
   per-clip et détruit l'audio seul (0.446 → 0.329), car le **niveau sonore absolu est informatif** (une
   explosion est forte).

⇒ Le verrou n'était pas l'architecture de fusion mais la **représentation audio** : passer des statistiques
log-mel 132-d au **VGGish 128-d officiel** a multiplié le gain audio par ~20. En outre le calibrage par
température sur la grille officielle est un **résultat négatif** (NLL s'améliore, ECE non : 0.0348 →
0.0369 en val, 0.0181 → 0.0244 en test pour `late`) : les scores bruts sont déjà bien calibrés.

Table complète et à jour : `.venv/bin/python scripts/fusion_table.py`.

### 3.4 Audit de fuite (hors-source), localisation par catégorie et robustesse

- **Hors-source** : **66.6 %** des 800 clips de test proviennent d'une source vue à l'entraînement. Sur
  le sous-ensemble *jamais vu* (267 clips), l'AP est **meilleure** pour tous les runs
  (`p3vggish30ep_cross` : 0.8452 contre 0.7435 en global, 0.8258 contre 0.7817 en per-clip ; visuel seul
  0.7980 / 0.8008) ⇒ le chiffre n'est pas gonflé par la mémorisation de sources, et le gain multimodal y
  survit (+0.047 sur le visuel seul). L'audio seul est le seul run insensible au partage de source
  (0.6528 vs 0.6541) — le son ne dépend pas de l'identité du film.
- **Localisation par catégorie** (`scripts/per_class_localization.py`, règle en secondes) : deux défauts
  systématiques chiffrés — `riot` est détecté **tard** (+38 s à +62 s selon le run) et `car_accident` est
  **sous-segmenté** (513 événements annotés pour 151-234 segments prédits, F1@0.5 0.054-0.174).
- **Classe faible `abuse`** : ce n'est pas un défaut de features mais une **famine de données** — 50 clips
  positifs sur 3 804 à l'entraînement (1.3 % contre 9.6-12 % pour les autres), la tête de catégorie ne se
  déclenche **jamais** (0 TP / 0 FP au seuil 0.5) **alors que ses incidents sont bien localisés**
  (tIoU 0.460). Diagnostic : `scripts/class_diagnostic.py`. La **pondération inverse** (`pos_weight`)
  confirme la cause — la tête se réveille (0 → 7/11 vrais positifs) — mais **échoue comme remède** sur
  une métrique de classement : l'AP `abuse` de `late` passe de 0.138 à **0.096** (le rappel monte,
  l'ordre se dégrade). Sur `cross`, elle aide au contraire (+0.169 d'AP `abuse`) ⇒ l'effet **dépend de
  l'architecture**, à présenter comme une interaction. Détail : la val officielle ne contient
  **aucun** clip `abuse` ⇒ la sélection de modèle en était aveugle ; `ckpt_last.pt` est désormais
  conservé et s'avère le meilleur checkpoint en catégorisation (abuse 0.259 contre 0.250).
- **Robustesse (P8) — quatre familles, en espace de features** (`scripts/robustness_sweep.py`) :

| dégradation | modèle | propre | pire | Δ APglob | Δ macro |
|---|---|---|---|---|---|
| audio → silence | `cross` (VGGish) | 0.7745 | 0.7243 | **−0.050** | −0.100 |
| audio → silence | `p3full_late` (mel) | 0.8143 | 0.8133 | **−0.001** | −0.005 |
| bruit audio 0 dB (bruit = signal) | `cross` | 0.7745 | 0.7768 | **+0.002** | −0.005 |
| 75 % des snippets audio perdus | `cross` | 0.7745 | 0.7594 | −0.015 | −0.003 |
| occlusion visuelle 30 s | `cross` / `late` | — | — | −0.012 / −0.024 | −0.072 |
| occlusion 30 s, portail **aveugle** vs **prévenu** | `adaptive` | 0.7281 | 0.7192 / **0.7389** | −0.009 / **+0.011** | — |

  Le mode de défaillance est la **suppression d'information**, pas la perturbation (le bruit à 0 dB est
  inoffensif). Le modèle mel ne perd rien sans audio (−0.001) alors que le VGGish en perd 0.050 ⇒ c'est
  bien la représentation qui rend l'audio porteur. Contrôle de cohérence : l'atténuation niveau 0
  reproduit la ligne `--drop-modality` au 3e-9 d'AP près sur les trois modèles.
  **Portail de fiabilité : la réplication ne confirme pas.** Sur 3 seeds, le gain « portail prévenu vs
  aveugle » vaut **+0.016 / −0.004 / +0.031** (2 positifs sur 3, moyenne +0.015) — alors que le modèle
  `adaptive` **propre** varie de 0.632 à 0.728 selon le seed (dispersion **0.096**, σ 0.051), soit
  **six fois l'effet**. L'effet n'est donc pas distinguable du bruit de seed ⇒ à présenter comme « non
  réfuté, non démontré », pas comme un gain. **Réserve générale** : perturbations de la représentation
  (features précalculées), pas des médias.
- **Modalité retirée** : contrat unique (`reliability.drop_modality`) partagé par l'entraînement
  (dropout 0.15) et l'évaluation (`--drop-modality`).

## 4. Ce qui reste (par ordre de valeur)

| Priorité | Chantier | État / pourquoi |
|---|---|---|
| 1 | **P6 : interface de supervision** | ✅ console Streamlit (`app.py`, `safewatch/ui/`) : état, figures, revue d'incidents, décision opérateur append-only ; 11 tests UI. Reste : la vidéo de démonstration |
| 2 | **P4 : mode en ligne causal** | ✅ localisation par catégorie + **scoring causal/streaming** (`--causal-prefix-stride`, §4.5 du rapport). **Contrôlé sur 3 seeds** : le coût en AP est robuste (AP −0.072/−0.090, même signe partout) mais l'effet sur la **localisation s'inverse selon la seed** ⇒ non revendicable, et c'est une correction d'une conclusion initiale |
| 3 | **Rapport final + démonstration vidéo** | 🔄 `docs/report/rapport.md` complet (chiffres audités, §4.5 causal, figures reproductibles) ; reste la relecture + la vidéo |
| 4 | P5 : calibration par classe | ✅ **fait — résultat négatif/neutre** : la T globale *dégrade* la tête catégorielle (cross 0.0738→0.1100), la calibration par classe ne la sauve pas (cross 0.0810 ; late 0.0425 = égalité), `abuse` incalibrable (0/150 en val). Décision : tête non calibrée, catégorie affichée comme un rang |
| 5 | P7 : Edge — **extracteur léger** (ONNX) | tête exportée et mesurée (4.08 MB, 6.7 ms / 100 snippets, parité 3.8e-06) ; int8 refusé (3e fois) ; le goulot est l'extraction |
| 6 | `abuse` : données, pas d'algorithme | la famine est prouvée (50/3 804) et la pondération a échoué ⇒ il faut relabelliser/sur-échantillonner ou fusionner la classe |
| 7 | Répliquer le gain du portail (2 seeds supplémentaires) | ✅ **fait, et le résultat est négatif** : 2 seeds positifs sur 3, moyenne +0.015 — mais la dispersion de seed du `adaptive` propre vaut 0.096, soit 6× l'effet ⇒ **le gain du portail n'est pas démontré**. À écrire comme tel, et à retirer des arguments du rapport |
| 8 | Marge de seed des autres variantes | ✅ **fait pour la revendication principale** : marge +0.0433 contre une dispersion de 0.0065 ⇒ **tenue** (3 seeds). Restent non répliqués les rangs intermédiaires (`late`, `early`) |

### 3.5 Budget Edge (P7, premier jet mesuré)

| modèle | fp32 ONNX | parité vs PyTorch | p50 / 100 snippets | int8 | int8 p50 | verdict int8 |
|---|---|---|---|---|---|---|
| fusion `p3full_early` | 2.331 MB | 7.6e-06 | 5.70 ms | 0.597 MB | 12.52 ms | refusé (logits décalés de 0.157) |
| visuel seul `p1_visual_swin` | 2.073 MB | 9.5e-06 | 3.42 ms | 0.533 MB | 11.69 ms | refusé (décalage 0.146) |
| **fusion `p3vggish30ep_cross`** (modèle de référence) | **4.083 MB** | **3.8e-06** | **6.68 ms** | 1.404 MB | 12.77 ms | **refusé (décalage 0.132, ×0.41 en vitesse)** |

La tête du modèle de référence s'exporte proprement (parité 3.8e-06, 6.7 ms pour 100 snippets ≈ 4.4 min
de vidéo). L'int8 est **refusé une troisième fois** sur un troisième graphe : le décalage de logits
(0.13) dépasse la tolérance (0.05) et la quantification **ralentit** (×0.41) — résultat négatif
reproduit, pas un accident. Le goulot reste l'**extraction de features**, pas la tête de décision.

Mesures prises avec `loadavg₁ ≈ 16` (ladder en cours) ⇒ **majorées** ; à re-mesurer au repos.
Conclusion utile : 100 snippets ≈ 4.4 min de vidéo et coûtent ~6 ms ⇒ **la tête de décision n'est pas le
goulot d'étranglement**, c'est l'extraction de features. Antisèche : l'int8 rétrécit 4× mais ralentit
2-3× sur ce graphe (résultat négatif, assumé).

## 5. Limites assumées (à dire au jury, pas à cacher)

1. Benchmarks **re-baselinés** sur nos features (le miroir fournit 768-d VideoSwin / 2048-d I3D, l'officiel
   1024-d I3D) : aucune valeur n'est directement comparable à la littérature, et chaque tableau le dit.
2. La fusion **bat désormais** sa référence visuelle sur la grille officielle (+0.049 global /
   +0.081 per-clip) ; sur le miroir la marge reste faible (+0.0025 / +0.0116) car le backbone visuel y
   est bien plus fort et le mel 132-d plafonne l'audio. Toute comparaison inter-grilles doit donc se
   faire **à backbone égal** (le dire explicitement dans le rapport).
3. Les canaux visuels de fiabilité sont des **proxies** (énergie du descripteur, mouvement temporel) :
   luminosité/flou réels exigent les pixels décodés (piste Edge).
4. Machine **sans GPU** (CPU-only) : donc modèles sur features gelées (~0.5-1 M paramètres) plutôt
   qu'un entraînement de bout en bout ; les variantes lourdes (AVadCLIP, transformers vidéo) restent
   citées comme références, pas reproduites.
5. La **reconnaissance d'événements acoustiques** (session 15) est assurée par un tagger **gelé et
   hors domaine** (YAMNet, AudioSet-YouTube 521 classes, ONNX, ~8 ms/image de 0,96 s sur le CPU) :
   l'alerte *nomme* les événements sonores (Explosion, Gunshot, Skidding…) avec leur probabilité et
   leur heure, mais c'est de la **corroboration** — le tagger n'a jamais vu XD-Violence, la carte
   le dit dans ses avertissements et le verdict final reste le modèle de fusion + l'opérateur.
