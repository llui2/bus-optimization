# Optimització de línies de bus amb Genetic Algorithm

Aquest repositori conté la implementació del Treball de Fi de Grau **“Optimització de la configuració de línies d’autobús”**. L’objectiu principal del projecte és optimitzar línies de bus mitjançant un **algorisme genètic** aplicat sobre una xarxa viària representada com un graf i una matriu Origen-Destí que representa la demanda de passatgers.

El projecte inclou tres models principals:

1. **GA per a una línia**
2. **GA multilínia seqüencial**
3. **GA multilínia conjunt**

Cada model permet estudiar el comportament de l’algorisme en una situació diferent: primer optimitzant una sola línia, després afegint línies una darrere l’altra, i finalment optimitzant totes les línies simultàniament.

---

## 1. Requisits

El projecte està pensat per executar-se amb Python.

Versió recomanada:

```bash
Python 3.9
```

Llibreries principals utilitzades:

```bash
numpy
pandas
matplotlib
networkx
```

Si el projecte inclou un fitxer `requirements.txt`, es poden instal·lar les dependències amb:

```bash
pip install -r requirements.txt
```

Si no hi ha `requirements.txt`, es poden instal·lar manualment amb:

```bash
pip install numpy pandas matplotlib networkx
```

---

## 2. Estructura general del projecte

L’estructura principal del projecte és aproximadament la següent:

```text
bus-optimization/
├── data/
│   ├── road_network/
│   │   └── variants/
│   │       ├── base/
│   │       ├── light_fill/
│   │       └── dense_fill/
│   └── bus_network/
│       ├── nodes.csv
│       └── od_scenarios/
│           ├── od_base.csv
│           ├── od_one_center.csv
│           ├── od_two_centers.csv
│           ├── od_center_periphery.csv
│           └── od_dense_hotspots.csv
│
└── script/
    └── GA_2026/
        ├── busline_ga/
        │   ├── config/
        │   ├── core/
        │   ├── experiments/
        │   ├── generators/
        │   └── visualization/
        ├── results/
        ├── run_single_line_guided_experiments.py
        ├── run_seq_od_comparison_only.py
        └── run_seq_full_grid_only.py
```

La carpeta més important per executar el codi és:

```text
script/GA_2026/
```

Per tant, abans d’executar qualsevol model, cal situar-se en aquesta carpeta.

En Windows PowerShell:

```powershell
cd "C:\Users\neuso\OneDrive - URV\uni\4t\TFG\bus-optimization-main\bus-optimization-main\bus-optimization\script\GA_2026"
```

---

## 3. Dades d’entrada

El projecte utilitza dues fonts principals de dades.

### 3.1 Xarxa viària

La xarxa viària es representa com un graf. Els nodes representen punts de la xarxa i les arestes representen trams de carretera.

Les variants principals utilitzades són:

```text
base
dense_fill
```

La variant `base` correspon a la xarxa estàndard. La variant `dense_fill` afegeix connexions addicionals entre nodes propers, augmentant la connectivitat del mapa.

També pot existir la variant:

```text
light_fill
```

però en les proves principals del TFG s’han utilitzat sobretot `base` i `dense_fill`.

---

### 3.2 Matrius Origen-Destí

Les matrius OD representen la demanda de passatgers entre parades.

Els escenaris principals són:

```text
base
one_center
two_centers
center_periphery
dense_hotspots
```

Cada escenari representa una distribució diferent de la demanda:

| OD case            | Descripció                                           |
| ------------------ | ---------------------------------------------------- |
| `base`             | Demanda general distribuïda per la xarxa.            |
| `one_center`       | Demanda concentrada en una zona central.             |
| `two_centers`      | Demanda concentrada en dos nuclis principals.        |
| `center_periphery` | Demanda entre una zona central i zones perifèriques. |
| `dense_hotspots`   | Demanda concentrada en punts de demanda elevada.     |

---

## 4. Paràmetres principals

Els models utilitzen una configuració comuna:

| Paràmetre         | Valor habitual | Descripció                                |
| ----------------- | -------------: | ----------------------------------------- |
| `line_length`     |              6 | Nombre de parades per línia.              |
| `population_size` |            150 | Nombre d’individus de cada població.      |
| `generations`     |            100 | Nombre de generacions del GA.             |
| `lambda`          |            0.5 | Pes entre servei i cost.                  |
| `seed`            |             42 | Llavor aleatòria per reproduir resultats. |
| `n_lines`         |       2, 3 o 4 | Nombre de línies del sistema multilínia.  |

El paràmetre `lambda` controla el compromís entre servei i cost:

| Valor de λ | Interpretació                            |
| ---------: | ---------------------------------------- |
|        `0` | Prioritat total al servei.               |
|      `0.5` | Compromís intermedi entre servei i cost. |
|        `1` | Prioritat total al cost.                 |

---

# 5. Model 1: GA per a una línia

El primer model optimitza una única línia de bus. Cada individu de la població representa una possible seqüència de parades.

El fitxer principal és:

```text
busline_ga/experiments/main_ga.py
```

---

## 5.1 Executar una prova simple

Exemple amb mapa `base`, OD `base` i `lambda = 0.5`:

```powershell
python -B -m busline_ga.experiments.main_ga --map-case base --od-case base --lambda 0.5 --seed 42 --od-min-to-plot 3 --output-dir results\single_line_guided\example\od_base\map_base\lambda_0p5
```

Aquesta execució genera una línia optimitzada i guarda els resultats dins de la carpeta indicada a `--output-dir`.

---

## 5.2 Comparació OD × mapes

Per comparar com canvia la línia segons la demanda i el mapa, es poden executar proves amb totes les OD i els dos mapes principals.

Exemples:

```powershell
python -B -m busline_ga.experiments.main_ga --map-case base --od-case base --lambda 0.5 --seed 42 --od-min-to-plot 3 --output-dir results\single_line_guided\od_comparison\od_base\map_base
```

```powershell
python -B -m busline_ga.experiments.main_ga --map-case dense_fill --od-case base --lambda 0.5 --seed 42 --od-min-to-plot 3 --output-dir results\single_line_guided\od_comparison\od_base\map_dense_fill
```

```powershell
python -B -m busline_ga.experiments.main_ga --map-case base --od-case one_center --lambda 0.5 --seed 42 --od-min-to-plot 1 --output-dir results\single_line_guided\od_comparison\od_one_center\map_base
```

---

## 5.3 Full grid per a una línia

També es pot executar una graella completa amb:

```text
OD cases = base, one_center, two_centers, center_periphery, dense_hotspots
map cases = base, dense_fill
lambda = 0, 0.25, 0.5, 0.75, 1
```

Això permet estudiar:

* l’efecte de la matriu OD;
* l’efecte del mapa;
* l’efecte del paràmetre `lambda`.

Si existeix l’script:

```text
run_single_line_guided_experiments.py
```

es pot executar directament amb:

```powershell
python run_single_line_guided_experiments.py
```

Els resultats es guarden habitualment a:

```text
results/single_line_guided/
```

---

## 5.4 Resultats generats pel model d’una línia

Segons la configuració del codi, cada execució pot generar:

```text
summary.txt
config.txt
final_line_with_od.pdf
final_line_with_od.png
metrics.csv
```

Els fitxers més importants són:

| Fitxer                   | Descripció                                     |
| ------------------------ | ---------------------------------------------- |
| `final_line_with_od.pdf` | Visualització final de la línia sobre el mapa. |
| `summary.txt`            | Resum textual de la millor línia obtinguda.    |
| `metrics.csv`            | Mètriques de servei, cost i fitness.           |
| `config.txt`             | Configuració utilitzada en l’execució.         |

---

# 6. Model 2: GA multilínia seqüencial

El model multilínia seqüencial construeix diverses línies una darrere l’altra. Primer optimitza la línia 1, la fixa, després optimitza la línia 2 tenint en compte la primera, i així successivament fins arribar al nombre total de línies.

El fitxer principal és:

```text
busline_ga/experiments/main_multiline_ga.py
```

---

## 6.1 Executar una prova simple

Exemple amb dues línies, mapa `base`, OD `base` i `lambda = 0.5`:

```powershell
python -B -m busline_ga.experiments.main_multiline_ga --map-case base --od-case base --n-lines 2 --line-length 6 --population-size 150 --generations 100 --lambda 0.5 --seed 42 --shared-stop-penalty 0.0 --output-dir results\sequential_multiline_guided\example\od_base\map_base\n_lines_2\lambda_0p5
```

Important: en algunes versions del projecte, `main_multiline_ga.py` no reconeix l’argument `--od-min-to-plot`. Per això, en el model multilínia seqüencial és millor no utilitzar aquest argument al CLI si dona error.

---

## 6.2 Prova OD comparison

Aquesta prova compara totes les OD amb els dos mapes principals, mantenint `lambda = 0.5`.

La idea és generar resultats amb aquesta estructura:

```text
results/sequential_multiline_guided/od_comparison/
├── od_base/
│   ├── map_base/
│   │   ├── n_lines_2/lambda_0p5/
│   │   ├── n_lines_3/lambda_0p5/
│   │   └── n_lines_4/lambda_0p5/
│   └── map_dense_fill/
│       ├── n_lines_2/lambda_0p5/
│       ├── n_lines_3/lambda_0p5/
│       └── n_lines_4/lambda_0p5/
├── od_one_center/
│   ├── map_base/n_lines_2/lambda_0p5/
│   └── map_dense_fill/n_lines_2/lambda_0p5/
├── od_two_centers/
├── od_center_periphery/
└── od_dense_hotspots/
```

En aquesta configuració:

* per `od_base`, es poden provar `K = 2`, `K = 3` i `K = 4`;
* per la resta d’OD, es pot provar només `K = 2` per reduir el temps d’execució.

Si existeix l’script:

```text
run_seq_od_comparison_only.py
```

es pot executar amb:

```powershell
python run_seq_od_comparison_only.py
```

Aquest script hauria de:

1. executar el GA seqüencial;
2. guardar les mètriques;
3. generar la imatge final;
4. comprovar que s’ha creat un `.pdf` o `.png`.

---

## 6.3 Full grid seqüencial

El full grid serveix per estudiar l’efecte de `lambda`.

Per reduir el temps de càlcul, es pot executar només amb:

```text
od_case = base
map_case = base, dense_fill
n_lines = 2
lambda = 0, 0.25, 0.5, 0.75, 1
```

L’estructura de sortida esperada és:

```text
results/sequential_multiline_guided/full_grid/
└── od_base/
    ├── map_base/
    │   └── n_lines_2/
    │       ├── lambda_0/
    │       ├── lambda_0p25/
    │       ├── lambda_0p5/
    │       ├── lambda_0p75/
    │       └── lambda_1/
    └── map_dense_fill/
        └── n_lines_2/
            ├── lambda_0/
            ├── lambda_0p25/
            ├── lambda_0p5/
            ├── lambda_0p75/
            └── lambda_1/
```

Si existeix l’script:

```text
run_seq_full_grid_only.py
```

es pot executar amb:

```powershell
python run_seq_full_grid_only.py
```

---

## 6.4 Resultats generats pel model seqüencial

Cada execució del model seqüencial pot generar:

```text
multiline_ga_summary.txt
multiline_config.txt
multiline_system_metrics.csv
multiline_lines.csv
multiline_adjusted_line_metrics.csv
multiline_pairwise_overlap.csv
multiline_map.pdf
multiline_map.png
```

Els fitxers més importants són:

| Fitxer                           | Descripció                                  |
| -------------------------------- | ------------------------------------------- |
| `multiline_map.pdf`              | Visualització final del sistema multilínia. |
| `multiline_ga_summary.txt`       | Resum textual de l’execució.                |
| `multiline_system_metrics.csv`   | Mètriques globals del sistema.              |
| `multiline_lines.csv`            | Informació de cada línia generada.          |
| `multiline_pairwise_overlap.csv` | Solapament entre parelles de línies.        |
| `multiline_config.txt`           | Configuració de l’experiment.               |

Important: una execució no s’ha de considerar completa només perquè s’hagin creat els CSV/TXT. Per a les proves visuals del TFG, cal comprovar que existeix també:

```text
multiline_map.pdf
```

o bé:

```text
multiline_map.png
```

---

# 7. Model 3: GA multilínia conjunt

El model multilínia conjunt optimitza totes les línies simultàniament. En aquest cas, cada individu de la població representa un sistema complet de `K` línies.

El fitxer principal és:

```text
busline_ga/experiments/main_joint_multiline_ga.py
```

També pot existir un fitxer de visualització directa:

```text
busline_ga/visualization/visualize_joint_multiline_ga.py
```

Aquest últim és útil perquè executa el model i genera directament la visualització final.

---

## 7.1 Executar una prova simple del model conjunt

Exemple amb dues línies, mapa `base`, OD `base` i `lambda = 0.5`:

```powershell
python -B -m busline_ga.experiments.main_joint_multiline_ga --map-case base --od-case base --n-lines 2 --line-length 6 --population-size 150 --generations 100 --lambda 0.5 --seed 42 --mutation-prob 0.20 --crossover-prob 0.80 --compactness-penalty 0.0 --n-elite 2 --tournament-size 3 --od-min-demand 3
```

---

## 7.2 Executar el model conjunt amb visualització

Per generar directament la imatge final del sistema conjunt, es pot utilitzar:

```powershell
python -B -m busline_ga.visualization.visualize_joint_multiline_ga --map-case base --od-case base --n-lines 2 --line-length 6 --population-size 150 --generations 100 --lambda 0.5 --seed 42 --mutation-prob 0.20 --crossover-prob 0.80 --compactness-penalty 0.0 --n-elite 2 --tournament-size 3 --od-min-demand 3
```

Exemple amb `lambda = 0.6`:

```powershell
python -B -m busline_ga.visualization.visualize_joint_multiline_ga --map-case base --od-case base --n-lines 2 --line-length 6 --population-size 150 --generations 100 --lambda 0.6 --seed 42 --mutation-prob 0.20 --crossover-prob 0.80 --compactness-penalty 0.0 --n-elite 2 --tournament-size 3 --od-min-demand 3
```

---

## 7.3 Proves recomanades per al model conjunt

Com que el model conjunt és més costós computacionalment, es recomana començar amb:

```text
od_case = base
map_case = base
n_lines = 2
lambda = 0.5, 0.6
```

Després es pot ampliar a:

```text
n_lines = 3
n_lines = 4
```

i comparar el comportament amb `lambda = 0.5` i `lambda = 0.6`.

Exemples:

```powershell
python -B -m busline_ga.visualization.visualize_joint_multiline_ga --map-case base --od-case base --n-lines 3 --line-length 6 --population-size 150 --generations 100 --lambda 0.5 --seed 42 --od-min-demand 3
```

```powershell
python -B -m busline_ga.visualization.visualize_joint_multiline_ga --map-case base --od-case base --n-lines 3 --line-length 6 --population-size 150 --generations 100 --lambda 0.6 --seed 42 --od-min-demand 3
```

```powershell
python -B -m busline_ga.visualization.visualize_joint_multiline_ga --map-case base --od-case base --n-lines 4 --line-length 6 --population-size 150 --generations 100 --lambda 0.5 --seed 42 --od-min-demand 3
```

```powershell
python -B -m busline_ga.visualization.visualize_joint_multiline_ga --map-case base --od-case base --n-lines 4 --line-length 6 --population-size 150 --generations 100 --lambda 0.6 --seed 42 --od-min-demand 3
```

---

## 7.4 Resultats generats pel model conjunt

El model conjunt pot generar:

```text
joint_multiline_summary.txt
joint_multiline_config.txt
joint_multiline_system_metrics.csv
joint_multiline_lines.csv
joint_multiline_map.pdf
joint_multiline_map.png
```

Els noms exactes poden variar segons la versió del codi, però els resultats principals són:

| Fitxer                               | Descripció                               |
| ------------------------------------ | ---------------------------------------- |
| `joint_multiline_map.pdf`            | Visualització final del sistema conjunt. |
| `joint_multiline_system_metrics.csv` | Mètriques globals del sistema.           |
| `joint_multiline_lines.csv`          | Informació de les línies generades.      |
| `joint_multiline_config.txt`         | Configuració de l’experiment.            |

---

# 8. Scripts auxiliars recomanats

Per facilitar l’execució de les proves, es poden utilitzar scripts situats directament dins de `script/GA_2026/`.

## 8.1 Una línia

```powershell
python run_single_line_guided_experiments.py
```

Executa les proves finals del model d’una línia.

Resultats esperats:

```text
results/single_line_guided/
```

---

## 8.2 Multilínia seqüencial: OD comparison

```powershell
python run_seq_od_comparison_only.py
```

Executa les proves de comparació d’OD i mapes per al model seqüencial.

Resultats esperats:

```text
results/sequential_multiline_guided/od_comparison/
```

---

## 8.3 Multilínia seqüencial: full grid

```powershell
python run_seq_full_grid_only.py
```

Executa les proves de diferents valors de `lambda` per al model seqüencial.

Resultats esperats:

```text
results/sequential_multiline_guided/full_grid/
```

---

# 9. Recomanacions d’execució

## 9.1 Començar sempre amb una prova petita

Abans d’executar una graella completa, és recomanable provar un sol cas:

```powershell
python -B -m busline_ga.experiments.main_ga --map-case base --od-case base --lambda 0.5 --seed 42 --output-dir results\test_single_line
```

o bé:

```powershell
python -B -m busline_ga.experiments.main_multiline_ga --map-case base --od-case base --n-lines 2 --line-length 6 --population-size 150 --generations 100 --lambda 0.5 --seed 42 --output-dir results\test_seq_multiline
```

Això permet comprovar que:

* les dades es carreguen correctament;
* no hi ha errors de path;
* es creen els fitxers de resultats;
* es genera la imatge final.

---

## 9.2 Comprovar sempre que s’ha creat la imatge

En les proves del TFG no n’hi ha prou amb generar CSV/TXT. Cal comprovar que s’ha creat la visualització final.

Per exemple:

```text
multiline_map.pdf
```

o:

```text
multiline_map.png
```

Si només apareixen fitxers com:

```text
multiline_ga_summary.txt
multiline_config.txt
multiline_system_metrics.csv
multiline_lines.csv
```

vol dir que el GA ha acabat, però que encara falta executar o arreglar la part de visualització.

---

## 9.3 No interrompre l’execució abans d’hora

El model multilínia pot tardar força més que el model d’una línia. Si s’interromp l’execució amb `Ctrl + C`, és possible que es creïn algunes carpetes o CSV parcials, però que no es generi el PDF final.

Per tant, una execució només es considera acabada correctament quan:

1. el procés acaba sense error;
2. es guarden les mètriques;
3. es crea la imatge final.

---

# 10. Resultats utilitzats al TFG

Per al TFG, les proves principals són:

## Una línia

```text
- Evolució de H, P i C amb inicialització aleatòria.
- Comparació entre inicialització aleatòria i guiada.
- Comparació entre mapes base i dense_fill.
- Comparació entre escenaris OD.
- Comparació de diferents valors de lambda.
- Front de Pareto.
```

## Multilínia seqüencial

```text
- Comparació de K = 2, K = 3 i K = 4 amb OD base.
- Comparació OD × mapes amb K = 2.
- Comparació de lambdes amb OD base.
- Anàlisi de servei global, cost global i solapament.
```

## Multilínia conjunta

```text
- Comparació de lambda = 0.5 i lambda = 0.6.
- Execucions amb K = 2, K = 3 i K = 4.
- Anàlisi del servei global, cost global, fitness i recorregut final.
```

---

# 11. Notes finals

Aquest projecte utilitza una xarxa viària simulada i matrius OD generades artificialment. Per tant, els resultats no representen una ciutat real, sinó un entorn controlat per estudiar el comportament de l’algorisme.

L’objectiu principal és comparar diferents estratègies d’optimització:

* una sola línia;
* línies afegides de manera seqüencial;
* línies optimitzades conjuntament.

Aquesta comparació permet analitzar com canvien el servei, el cost i el solapament entre línies segons el model utilitzat.
