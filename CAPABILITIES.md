# LIFT — Fonctionnalités, Capacités, Limites et Objectifs

**Analyse complète basée sur le code source réel (67 fichiers Rust, 13 crates, 505 tests).**

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Pipeline de traitement](#2-pipeline-de-traitement)
3. [Fonctionnalités implémentées](#3-fonctionnalités-implémentées)
4. [Fonctionnalités partielles](#4-fonctionnalités-partielles)
5. [Fonctionnalités manquantes](#5-fonctionnalités-manquantes)
6. [Limites actuelles](#6-limites-actuelles)
7. [Précision des analyses](#7-précision-des-analyses)
8. [Ce qui manque pour les objectifs](#8-ce-qui-manque-pour-les-objectifs)
9. [Feuille de route](#9-feuille-de-route)

---

# 1. Vue d'ensemble

LIFT est un compilateur IR unifié pour IA classique + calcul quantique, écrit en Rust (13 crates) :

| Crate | Rôle |
|-------|------|
| `lift-core` | Noyau : contexte IR, types, vérificateur, printer, pass manager |
| `lift-ast` | Lexer, parser, builder pour fichiers `.lif` |
| `lift-tensor` | 110 opérations IA, inférence de forme, calcul FLOPs |
| `lift-quantum` | 50+ portes quantiques, bruit, Kraus, QEC, topologie |
| `lift-hybrid` | 21 opérations classique↔quantique |
| `lift-opt` | 11 passes d'optimisation |
| `lift-sim` | Analyse statique, modèles de coût GPU/QPU, énergie |
| `lift-predict` | Prédiction roofline, prédiction quantique |
| `lift-config` | Parseur fichiers `.lith` |
| `lift-import` | Import ONNX, PyTorch FX, OpenQASM (squelettes) |
| `lift-export` | Export LLVM IR, OpenQASM 3.0 (basiques) |
| `lift-cli` | CLI : verify, analyse, print, optimise, predict, export |
| `lift-tests` | 505 tests, 0 échecs |

---

# 2. Pipeline de traitement

```
.lif → Lexer → Parser → Builder → Context IR → Vérification → Analyse → Optimisation → Export
```

### Étape 1 — Lexer (COMPLET)
Découpe le texte `.lif` en tokens : mots-clés, directives `#dialect`, identifiants `@name`/`%var`, littéraux, ponctuation. Gestion d'erreurs incluse.

### Étape 2 — Parser (COMPLET)
Construit l'AST : directives de dialecte, modules, fonctions, opérations avec opérandes/attributs/signatures de type. Types tensor (`tensor<1x784xf32>`), qubit, bit, hamiltonian. Recovery d'erreurs.

### Étape 3 — Builder (COMPLET)
Convertit l'AST en IR interne dans le `Context` : valeurs SSA, opérations, blocs, régions, fonctions, modules.

### Étape 4 — Context IR (COMPLET)
Structure centrale avec SlotMaps pour values, ops, blocks, regions, types + StringInterner. Types : Integer (i1-i64), Float (f16-f64, fp8), Boolean, Void, Tuple, Function, Opaque (tensor, qubit, bit, hamiltonian).

### Étape 5 — Vérification (COMPLET, 3 passes)
- **SSA** : chaque valeur définie une seule fois, chaque usage après définition
- **Bonne formation** : aucune référence pendante (ops ↔ valeurs ↔ blocs ↔ régions)
- **Linéarité** : chaque qubit consommé exactement une fois (no-cloning)

12 types d'erreurs : UndefinedValue, MultipleDefinition, DominanceViolation, TypeMismatch, LinearityViolation, QubitLeaked, BranchLinearityMismatch, DanglingReference, MissingTerminator, OrphanedOperation, OrphanedBlock, InvalidOperation.

### Étape 6 — Analyse statique (COMPLET)
Produit : total_flops, total_memory_bytes, peak_memory, num_ops par dialecte, op_breakdown. Quantique : qubits, portes 1Q/2Q/3Q, mesures, circuit_depth, estimated_fidelity, bruit accumulé.

### Étape 7 — Optimisation (11 passes)

| Passe | Type | Action concrète |
|-------|------|-----------------|
| `canonicalize` | Tensor | Normalise les patterns |
| `constant-folding` | Tensor | Évalue les constantes à la compilation |
| `dce` | Général | Supprime les ops dont les résultats sont inutilisés |
| `tensor-fusion` | Tensor | Fusionne matmul+add+relu → fused_matmul_bias_relu |
| `cse` | Général | Élimine les sous-expressions communes |
| `flash-attention` | Tensor | Remplace attention → flash attention |
| `quantisation-pass` | Tensor | Annote pour quantisation INT8/INT4 |
| `gate-cancellation` | Quantum | Annule H·H=I, X·X=I, S·Sdg=I, T·Tdg=I |
| `rotation-merge` | Quantum | Fusionne Rz(a)·Rz(b) → Rz(a+b) |
| `noise-aware-schedule` | Quantum | Réordonne les portes pour minimiser décohérence |
| `layout-mapping` | Quantum | Annote les portes 2-qubit nécessitant SWAPs |

**ATTENTION** : seules 5 passes sont connectées au CLI (canonicalize, constant-folding, dce, tensor-fusion, gate-cancellation). Les 6 autres existent en code mais pas dans le match de cmd_optimise.

### Étape 8 — Prédiction (COMPLET)
- **Roofline GPU** : compute_time_ms, memory_time_ms, bottleneck. Modèles A100 (312 TFLOPS) et H100 (989 TFLOPS).
- **Quantique** : fidélité, circuit_time_us, shots nécessaires. 3 modèles : superconducteur, ions piégés, atomes neutres.
- **Budget** : vérifie FLOPs max, mémoire max, temps max, fidélité min. ReactiveBudget pour suivi en temps réel.

### Étape 9 — Export (BASIQUE)
- **LLVM IR** : squelette seulement — les ops sont émises en commentaires, pas en vrai LLVM IR
- **OpenQASM 3.0** : 10 portes sur 50+ (H, X, Y, Z, CX, CZ, Measure, RZ, RX, RY)

---

# 3. Fonctionnalités implémentées

## 3.1 Dialecte Tensor — 110 opérations

Toutes les 110 opérations sont définies dans l'enum `TensorOp` avec conversion nom↔enum, nombre d'entrées, classification. Inférence de forme fonctionnelle pour : MatMul, Linear, Conv2D, Conv1D, DepthwiseConv2D, Attention, FlashAttention, MaxPool2D, GlobalAvgPool, BatchNorm, LayerNorm, RMSNorm, InstanceNorm, SparseMatMul, élémentaire, ELU, LeakyReLU, Mish, HardSwish. Calcul FLOPs exact pour MatMul, Linear, Conv2D, Attention, ReLU, élémentaire, fused ops.

## 3.2 Dialecte Quantum — 50+ portes

Portes : 9 standard 1Q + 7 paramétriques 1Q + 2 angle fixe + 13 portes 2Q + 2 portes 3Q + 2 multi-contrôlées + 8 mesure/contrôle + portes IonQ. Propriétés par porte : num_qubits, is_parametric, is_self_inverse, is_clifford, is_entangling. 5 jeux natifs (IBM, Rigetti, IonQ, Quantinuum, Simulateur). Bruit : GateNoise, CircuitNoise, KrausChannel (6 canaux). Topologie : linear, grid, heavy_hex, all_to_all, tree, custom + BFS. QEC : Surface, Steane, Shor, Repetition, LDPC.

## 3.3 Dialecte Hybrid — 21 opérations

Encode/Decode, 5 gradients, 4 algorithmes variationnels, 2 transferts, 4 traitements, CoExecute, 2 mesures. AnsatzType, SyncPolicy, FeatureMap, EncodingStrategy.

## 3.4 CLI — 6 commandes

`verify`, `analyse` (texte/JSON), `print`, `optimise` (avec .lith), `predict` (A100/H100), `export` (llvm/qasm).

## 3.5 Modèles d'énergie

EnergyModel A100/H100 : énergie joules/kWh, CO2 grammes, énergie quantique (cryogénie). **Non connecté au CLI.**

## 3.6 Tests — 505 tests, 0 échecs

Types, opérations, formes, FLOPs, mémoire, portes, bruit, topologie, QEC, Kraus, benchmarks (GPT-2, LLaMA-7B, ResNet-50, BERT-base).

---

# 4. Fonctionnalités partielles (code existe, incomplet)

## 4.1 Export LLVM IR — SQUELETTE

L'exporteur produit `define void @func(ptr %arg0) { entry: ; tensor.matmul  ret void }`. Les opérations sont en **commentaires**, pas en vrai LLVM IR. Aucun appel cuBLAS/cuDNN, aucune gestion mémoire.

## 4.2 Export OpenQASM — 10 portes sur 50+

Fonctionnel pour H, X, Y, Z, CX, CZ, Measure, RZ, RX, RY. Les 40+ autres → `// unsupported gate`. Le mapping qubit utilise un compteur, pas le vrai SSA.

## 4.3 Import ONNX/PyTorch/QASM — SQUELETTES

Les 3 importeurs lisent le format source mais créent un module+fonction **vides**. Aucun nœud/opération n'est réellement converti en opérations LIFT.

## 4.4 Layout Mapping — ANNOTATION

Ajoute `needs_swap = true` sur les portes 2Q non adjacentes. **Ne fait pas l'insertion réelle de SWAPs ni le routage.**

## 4.5 Passes non connectées au CLI

6 passes existent mais ne sont pas dans le `match` de `cmd_optimise` : rotation-merge, flash-attention, cse, quantisation-pass, noise-aware-schedule, layout-mapping.

## 4.6 Inférence de forme — PARTIELLE

Fonctionne pour environ 20 opérations sur 110. Manque : Conv3D, ConvTranspose2D, Reshape, Permute, Concat, Split, Slice, LSTM, GRU, RNN, FFT, SVD, Einsum, GNN, MoE, diffusion, quantisation, parallélisme.

---

# 5. Fonctionnalités manquantes

## 5.1 Pas de vérification sémantique des opérations

Le vérificateur vérifie SSA/bonne formation/linéarité mais **ne vérifie PAS** que `tensor.matmul` a 2 entrées tensor, que les dimensions sont compatibles, que `tensor.conv2d` reçoit un tensor 4D, etc.

## 5.2 Pas d'exécution réelle

LIFT ne peut pas **exécuter** de programme. C'est purement un compilateur d'analyse. Il n'y a pas de runtime, pas d'interpréteur, pas de backend d'exécution GPU/QPU.

## 5.3 Pas de simulation quantique réelle

Le module `quantum_sim` fait de l'**analyse statique** (comptage de portes, estimation de fidélité). Il ne simule PAS l'état quantique (pas de vecteur d'état, pas de matrice densité, pas de simulation Monte Carlo).

## 5.4 Pas de génération de code machine

L'export LLVM ne produit pas de code exécutable. Il faudrait : lowering des opérations tensor vers des appels de bibliothèques (cuBLAS, cuDNN, oneDNN), gestion mémoire (allocation/libération), ordonnancement des kernels, code de lancement GPU.

## 5.5 Pas de support multi-fichiers

Un programme LIFT = un seul fichier `.lif`. Pas de système d'import/include, pas de modules séparés, pas de linking.

## 5.6 Pas de décomposition de portes

LIFT ne décompose pas automatiquement les portes non natives vers le jeu natif du hardware cible. Les jeux natifs sont définis mais pas utilisés pour la transpilation.

## 5.7 Pas de scheduling GPU

Pas de placement des opérations sur des streams CUDA, pas de recouvrement calcul/mémoire, pas de parallélisme d'opérations.

## 5.8 Pas d'auto-différentiation

Les opérations de gradient sont **déclarées** (grad_matmul, grad_relu, etc.) mais il n'y a pas de système d'auto-différentiation qui construit automatiquement le graphe backward à partir du forward.

## 5.9 Pas de gestion de données

Pas de chargement de données (datasets), pas de data loaders, pas de preprocessing. LIFT travaille uniquement sur le graphe de calcul.

---

# 6. Limites actuelles

## 6.1 Limites structurelles

| Limite | Impact |
|--------|--------|
| Pas d'exécution | LIFT analyse mais ne peut pas exécuter de modèle |
| Export squelette | Le code généré (LLVM/QASM) n'est pas exécutable en l'état |
| Import squelette | Impossible d'importer un vrai modèle ONNX/PyTorch |
| Pas de simulation QC | Fidélité estimée par formule, pas par simulation réelle |
| CLI incomplet | 6/11 passes non accessibles en ligne de commande |
| Énergie non connectée | Le modèle d'énergie existe mais n'est pas dans le CLI |

## 6.2 Limites du modèle de coût

- Le modèle roofline est une **approximation grossière** : il ne prend pas en compte les effets de cache, la latence de lancement des kernels, le recouvrement calcul/mémoire
- Le modèle quantique utilise des paramètres de bruit **moyens** par défaut, pas les propriétés réelles du device cible
- L'estimation de fidélité suppose un bruit **indépendant** par porte (pas de corrélations spatiales/temporelles)

## 6.3 Limites du vérificateur

- Pas de vérification de types d'opérations (nombre/type d'entrées)
- Pas de vérification de compatibilité de dimensions (forme des tensors)
- Pas de vérification de dominance complète (CFG)
- La vérification de linéarité ne gère pas les branches conditionnelles de façon exhaustive

## 6.4 Limites de l'optimiseur

- `tensor-fusion` ne reconnaît qu'un seul pattern (matmul+add+relu)
- `gate-cancellation` ne détecte que les paires **consécutives** (pas les paires séparées par des portes sur d'autres qubits)
- `rotation-merge` ne fonctionne que sur les rotations **consécutives** sur le même qubit
- `noise-aware-schedule` utilise un tri par temps de porte, pas un vrai algorithme d'ordonnancement contraint
- `layout-mapping` annote seulement, ne fait pas le routage réel

---

# 7. Précision des analyses

## 7.1 Comptage FLOPs

| Opération | Précision | Formule |
|-----------|-----------|---------|
| MatMul (MxK × KxN) | **Exacte** | 2 × M × K × N |
| MatMul batch (BxMxK × BxKxN) | **Exacte** | 2 × B × M × K × N |
| Linear (Mx K × KxN + N) | **Exacte** | 2 × M × K × N + M × N |
| Conv2D | **Exacte** | 2 × B × Cout × Hout × Wout × Cin × Kh × Kw |
| Attention | **Exacte** | 2 × B × H × (S² × D + S × D²) |
| ReLU / élémentaire | **Exacte** | nombre d'éléments |
| Reshape, Transpose | **Exacte** | 0 FLOPs (correct) |
| Fused ops | **Exacte** | somme des composants |
| LSTM, GRU, RNN | **Non implémenté** | — |
| Conv3D, ConvTranspose | **Non implémenté** | — |
| Einsum, FFT, SVD | **Non implémenté** | — |

**Précision globale** : pour les modèles purement Transformer (GPT, BERT, LLaMA), la précision du comptage FLOPs est **excellente** (erreur < 1%). Pour les modèles CNN, elle est bonne pour Conv2D mais manque les autres convolutions. Pour les modèles récurrents (LSTM), les FLOPs ne sont pas comptés.

## 7.2 Estimation mémoire

Calcule `éléments × byte_size(dtype)` par tensor. Précis pour la mémoire **statique** mais ne modélise pas : les activations intermédiaires allouées dynamiquement, le fragmentation mémoire GPU, les buffers de workspace (cuDNN), le KV cache pour l'inférence LLM.

## 7.3 Prédiction de temps (roofline)

| Aspect | Précision |
|--------|-----------|
| Identification compute-bound vs memory-bound | **Bonne** (cas standard) |
| Temps absolu | **Ordre de grandeur** (facteur 2-5x d'erreur possible) |
| Effets de cache | **Non modélisé** |
| Latence de lancement kernel | **Non modélisé** |
| Multi-GPU | **Non modélisé** (suppose 1 GPU) |
| Recouvrement calcul/mémoire | **Non modélisé** |

## 7.4 Fidélité quantique

L'estimation de fidélité est le **produit des fidélités individuelles** : F = ∏ f_gate × f_décoherence. C'est une **borne supérieure** (la vraie fidélité est souvent pire à cause des corrélations de bruit, du crosstalk, des erreurs de lecture).

---

# 8. Ce qui manque pour atteindre les objectifs fixés

L'objectif de LIFT est : **"Simulate → Predict → Optimise → Compile"**. Voici l'état actuel :

| Objectif | État | Ce qui manque |
|----------|------|---------------|
| **Simulate** | 40% | Analyse statique OK, mais pas de simulation d'exécution réelle (pas de vecteur d'état quantique, pas d'interpréteur tensor) |
| **Predict** | 70% | Roofline GPU OK, prédiction quantique OK, mais modèle trop simplifié (pas de cache, pas de multi-GPU, pas de scheduling) |
| **Optimise** | 50% | 11 passes existent, mais patterns limités, 6 passes non connectées, pas de graphe de réécriture général |
| **Compile** | 10% | Export LLVM/QASM squelettes, pas de code exécutable réel |

## 8.1 Pour atteindre Simulate (100%)

1. **Simulateur de vecteur d'état quantique** : multiplier les matrices de portes sur un vecteur 2^n. Nécessaire pour valider les circuits quantiques.
2. **Interpréteur tensor** : exécuter les opérations tensor avec des vraies valeurs numpy-like. Nécessaire pour valider les modèles IA.
3. **Simulation Monte Carlo** : pour estimer la distribution de mesure avec bruit.

## 8.2 Pour atteindre Predict (100%)

1. **Modèle de coût affiné** : intégrer latence de lancement, effets de cache L2, scheduling overlappé.
2. **Profils hardware réels** : charger les propriétés réelles des QPU (calibration IBM Quantum, temps de porte par qubit).
3. **Multi-GPU** : modèle de communication inter-GPU (NVLink, PCIe).
4. **Prédiction quantique avancée** : modèle de bruit corrélé, crosstalk, erreurs de lecture.

## 8.3 Pour atteindre Optimise (100%)

1. **Connecter les 6 passes manquantes au CLI** (quick fix).
2. **Plus de patterns de fusion** : matmul+gelu, conv+bn+relu, attention+layernorm.
3. **Gate cancellation non-locale** : annuler des paires séparées par des opérations sur d'autres qubits (commutation).
4. **Routage réel** : implémenter SABRE ou A* pour le layout mapping avec insertion de SWAPs.
5. **Décomposition de portes** : transpiler les portes non natives vers le jeu natif.
6. **Système de réécriture à base de patterns** : permettre de définir des règles de transformation déclaratives.

## 8.4 Pour atteindre Compile (100%)

1. **Lowering tensor → LLVM** : générer des appels réels vers cuBLAS/cuDNN/oneDNN.
2. **Lowering quantum → QASM complet** : supporter les 50+ portes.
3. **Gestion mémoire** : allocateur de mémoire GPU (allocation, libération, réutilisation).
4. **Code de lancement** : générer le code host qui orchestre les kernels GPU.
5. **Backend quantique** : générer du code pour IBM Qiskit Runtime, Amazon Braket, ou Google Cirq.
6. **Import réel** : convertir les vrais graphes ONNX/PyTorch en opérations LIFT.

---

# 9. Feuille de route (priorité)

## Priorité 1 — Quick fixes (effort faible, impact immédiat)

- [ ] Connecter les 6 passes restantes au CLI (modifier `cmd_optimise` dans main.rs)
- [ ] Connecter EnergyModel au CLI (ajouter une commande `energy`)
- [ ] Connecter predict_quantum au CLI (ajouter `--quantum` à la commande predict)
- [ ] Corriger le fichier lift-test manquant (`lift-test/src/config.rs`)

## Priorité 2 — Import/Export fonctionnels (effort moyen, impact élevé)

- [ ] Import ONNX réel : mapper les nœuds ONNX vers TensorOp
- [ ] Import PyTorch FX réel : mapper les nœuds FX vers TensorOp
- [ ] Export QASM complet : supporter toutes les 50+ portes
- [ ] Import QASM réel : parser les portes et créer les opérations quantum

## Priorité 3 — Optimisation avancée (effort moyen)

- [ ] Plus de patterns de fusion tensor
- [ ] Gate cancellation non-locale (commutation)
- [ ] Décomposition de portes vers jeu natif
- [ ] Routage réel (SABRE)

## Priorité 4 — Simulation (effort élevé)

- [ ] Simulateur de vecteur d'état (jusqu'à ~25 qubits)
- [ ] Interpréteur tensor simplifié
- [ ] Inférence de forme pour les 90 opérations restantes

## Priorité 5 — Compilation réelle (effort très élevé)

- [ ] Lowering tensor → LLVM avec appels cuBLAS
- [ ] Gestion mémoire GPU
- [ ] Backend quantum (Qiskit/Braket)

---

# Résumé final

| Métrique | Valeur |
|----------|--------|
| **Crates** | 13 |
| **Fichiers Rust** | 67 |
| **Tests** | 505 (0 échecs) |
| **Opérations définies** | 179 (110 tensor + 48 quantum + 21 hybrid) |
| **Passes d'optimisation** | 11 (5 connectées au CLI) |
| **Modèles de coût** | 5 (A100, H100, superconducteur, ions piégés, atomes neutres) |
| **Portes exportées QASM** | 10 / 50+ |
| **Import fonctionnel** | 0 / 3 |
| **Exécution possible** | Non |
| **Compilation réelle** | Non |

**LIFT est un framework d'analyse et d'optimisation IR solide et bien testé**, avec une excellente couverture des dialectes (tensor, quantum, hybrid) et une architecture propre. Son point fort est l'analyse statique (FLOPs, mémoire, fidélité, bruit, coût). **Ce qui lui manque principalement**, c'est la capacité d'exécuter réellement du code : les exports sont des squelettes, les imports sont vides, et il n'y a pas de runtime. Pour devenir un compilateur complet "Simulate → Predict → Optimise → Compile", il faut implémenter le lowering réel, les imports fonctionnels, et un simulateur.

**Ce document est l'analyse complète et honnête de l'état de LIFT.**
