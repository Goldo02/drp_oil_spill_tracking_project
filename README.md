# Multi-Drone Oil Spill Tracking

Simulazione di droni cooperanti per il tracking di una macchia di olio con:

- modello del drone a **single integrator**
- **partizione Voronoi** del dominio
- **legge di controllo** verso il centroide della cella
- stima locale del raggio tramite **edge detection**
- **consensus distribuito** sulle stime del raggio

Il progetto combina due livelli:

1. controllo di movimento dei droni nello spazio
2. stima geometrica della macchia tramite sensing locale

## Obiettivo

Ogni drone osserva una finestra locale della mappa, estrae il bordo della macchia, aggiorna una stima locale del raggio e si muove verso un target Voronoi pesato sulla frontiera dell'olio.

L'idea e mantenere i droni:

- separati in modo naturale grazie alle celle Voronoi
- orientati verso regioni informative del bordo
- coordinati in modo distribuito senza un controllore centrale

## Architettura del sistema

| Modulo | Ruolo |
|---|---|
| `environment.py` | Mappa, spill circolare e spill gaussiano |
| `sensors.py` | Sensori rumorosi GPS e camera |
| `edge_detection.py` | Pre-processing e Canny edge detection |
| `drone.py` | Stato del drone e dinamica single-integrator |
| `simulation_engine.py` | Sensing, Voronoi, controllo, consensus e logging |
| `visualization.py` | Rendering di spill, droni, target Voronoi e bordo |
| `main.py` | Setup, esecuzione e salvataggio delle figure |

## Modello del drone

Il drone e modellato come un **single integrator**:

```math
\dot p_i = u_i
```

con:

```math
p_i = [x_i, y_i]^T
```

e in discreto:

```math
p_i[k+1] = p_i[k] + \Delta t \, u_i[k]
```

Il comando viene saturato con una velocita massima per evitare dinamiche troppo aggressive.

## Partizione Voronoi

La partizione e costruita sui siti dati dalle posizioni correnti dei droni.

Per ogni punto del dominio si assegna il drone piu vicino. In implementazione il dominio continuo viene approssimato su una griglia di supporto campionata per mantenere il costo computazionale basso.

La cella Voronoi del drone `i` e:

```math
V_i = \{ x \in \Omega : \|x - p_i\| \le \|x - p_j\| \ \forall j \neq i \}
```

## Legge di controllo

Il target del drone e il centroide pesato della propria cella:

```math
c_i = \frac{\sum_{x \in V_i} \rho(x)\,x}{\sum_{x \in V_i} \rho(x)}
```

con densita:

```math
\rho(x) = 4 f(x)\,[1-f(x)]
```

Questa scelta mette peso massimo vicino al bordo della macchia, dove `f(x) \approx 0.5`.

Il comando e:

```math
u_i = \mathrm{sat}_{u_{\max}}\big(k(c_i - p_i)\big)
```

Interpretazione:

- se il drone e lontano dal target, si muove verso il centroide della cella
- se e vicino al target, il comando si riduce
- la partizione Voronoi impedisce che tutti i droni collassino nello stesso punto

## Modello della macchia

Il codice usa soprattutto una macchia circolare ammorbidita:

```math
d(x,y) = \sqrt{(x-x_0)^2 + (y-y_0)^2}
```

```math
f(x,y) =
\begin{cases}
1, & d(x,y) \le r_0 \\
\exp\left(-\frac{(d(x,y)-r_0)^2}{2\sigma^2}\right), & d(x,y) > r_0
\end{cases}
```

E disponibile anche un modello gaussiano continuo.

## Sensing ed edge detection

Ogni drone acquisisce una finestra locale della mappa globale. Il patch viene:

1. normalizzato
2. sfocato con Gaussian blur
3. passato a Canny
4. convertito in punti bordo

La stima locale del raggio usa i punti bordo rilevati:

```math
\hat r_i = \frac{1}{N}\sum_{k=1}^{N}\sqrt{(x_k-x_0)^2 + (y_k-y_0)^2}
```

Questa non e una least squares circle fit classica, ma una stima geometrica semplice e coerente con il caso circolare.

## Consensus distribuito

Le stime locali del raggio vengono poi fuse con un consensus tra i droni che vedono abbastanza olio.

Per un drone `i`:

```math
r_i^{k+1} = r_i^k + \alpha \left(\frac{1}{|N_i|}\sum_{j \in N_i} r_j^k - r_i^k\right)
```

con `alpha = 0.5`.

I droni non partecipanti possono essere aggiornati in modo euristico verso i partecipanti vicini con una media pesata inversa sulla distanza.

## Workflow della simulazione

Per ogni frame:

1. si aggiorna il sensing locale quando previsto
2. si esegue edge detection
3. si stima il raggio locale
4. si selezionano i droni idonei al consensus
5. si aggiornano le stime con consensus distribuito
6. si calcola la partizione Voronoi
7. si ricava il centroide pesato di ogni cella
8. si applica il comando single-integrator
9. si aggiorna la posizione dei droni
10. si salva lo stato per grafici e debug

## Parametri principali

Valori di default piu importanti:

- `dt = 0.05`
- `measure_every = 3`
- `consensus_iters = 10`
- `consensus_gain = 0.5`
- `neighbor_gain = 0.35`
- `control_gain = 1.8`
- `max_speed = 0.6`
- `voronoi_grid_step = 8`

Il dominio di simulazione e:

```text
x in [-5, 5]
y in [-5, 5]
grid_size = 500
```

## Esecuzione

### Simulazione headless

```bash
python3 main.py --frames 5000
```

### Con visualizzazione

```bash
python3 main.py --visualize --frames 5000
```

### Consensus fully connected

```bash
python3 main.py --visualize --frames 5000 --fully-connected
```

### Raggio di comunicazione

```bash
python3 main.py --communication-radius-cells 205
```

## Output generati

La simulazione salva:

- `final_simulation_state.png`
- `consensus_convergence.png`

Nel grafico finale compaiono:

- spill di olio
- droni
- punti bordo rilevati
- target Voronoi di ogni drone

## Note di implementazione

- La partizione Voronoi e calcolata su una griglia campionata per ridurre il costo computazionale.
- Il target e il centroide pesato della cella, non il baricentro puro.
- La densita `4 f (1-f)` concentra i droni verso la fascia di frontiera.
- La parte di consensus resta attiva per la stima del raggio.
- La dinamica del drone e volutamente semplice per rendere il sistema facile da estendere.

## Limiti attuali

- Il Voronoi e approssimato numericamente su una griglia
- Il drone e un modello single-integrator idealizzato
- La stima del raggio assume una geometria quasi circolare
- Non e presente un filtro probabilistico tipo Kalman
- Non c'e stima distribuita con covarianza

## Possibili estensioni

- visualizzazione esplicita delle celle Voronoi
- robust circle fitting al posto della media delle distanze
- EKF o DKF per la stima del raggio
- supporto a spill ellittici o non circolari
- legge di controllo radiale/tangenziale piu sofisticata

