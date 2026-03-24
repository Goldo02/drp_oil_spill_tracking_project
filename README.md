# Oil Spill Tracking — Drone Simulation

## OVERVIEW DEL PROGETTO

Questo repository implementa una simulazione di droni che tracciano il bordo di una macchia di olio (oil spill tracking). L'obiettivo è che un gruppo di droni trovi il bordo della macchia, si disponga lungo il perimetro e mantenga un'orbita stabile con una distribuzione angolare uniforme (consensus). Il codice è progettato come ambiente didattico/ricerca: semplice, modulare e leggibile, per studiare controllo radiale + consenso tangenziale su un bordo morfologicamente definito.

- Problema: droni che, usando solo una telecamera locale e propriocezione, devono identificare il bordo di una macchia e mantenervi un'orbita.
- Obiettivo: orbita (orbiting) stabile alla distanza desiderata $r_0$ e distribuzione uniforme degli angoli $\phi$ dei droni attorno al centro della macchia.
- Approccio high-level:
  - Ricerca casuale (`SEARCH`) finché la percezione locale non indica la presenza del bordo.
  - Quando si vede il bordo: passaggio in `APPROACH`, applicazione di una legge radiale (Matveev-style) per convergere al raggio desiderato e controllo tangenziale con termine di consensus per distribuire i droni uniformemente.

## ARCHITETTURA DEL SISTEMA

Componenti principali e ruoli:

- `SimulationEngine` (`simulation_engine.py`)
  - Orchestratore principale della simulazione.
  - Mantiene la mappa (`SimulationMap`), il modello di macchia (`OilSpill`) e la lista di `Drone`.
  - Esegue il loop `step()` che aggiorna percezione, transizioni di stato e controlli per ciascun drone.
  - Implementa i parametri di controllo globali: `c_star`, `u_bar`, `v_base`, `k_consensus`.

- `Drone` (`drone.py`)
  - Classe che incapsula stato fisico (posizione, velocità), sensori (`GPSSensor`, `CameraSensor`) e dinamica base (integrazione, rimbalzo ai bordi).
  - Espone API: `get_camera_view()`, `get_gps_pos()`, `set_velocity()`, `update_position()`.

- `oil_spill / field` (`environment.py`)
  - Modelli di macchia: `CircleOilSpill`, `GaussianOilSpill`.
  - Fornisce `field(X,Y)` che ritorna valori continui (0..1) sulla griglia del mondo; `CircleOilSpill` implementa una "softened circle" con raggio $r_0$.

- `edge_detection` (`edge_detection.py`)
  - Funzioni semplici di elaborazione locale:
    - `identify_centroid(camera_matrix)`: centro di massa dei pixel con valore > 0.5.
    - `check_geometric_lock(camera_matrix)`: regola "perfect geometric edge": il pixel centrale è olio (>=0.5) e ha almeno un vicino che è acqua (<0.5).
  - Queste funzioni sono helper per decisioni di stato e per identificare lock geometrici.

Interazione:
- `SimulationEngine` calcola il campo (`oil_spill.field`) su una griglia (`SimulationMap.X,Y`) e lo passa ai sensori dei droni. Ad ogni `step()`:
  1. Per ogni drone: legge `camera_view` tramite `Drone.get_camera_view()`.
  2. Calcola una misura locale (es. media 5x5 al centro) e decide transizione `SEARCH`→`APPROACH`.
  3. Per i droni in `APPROACH`: applica controllo radiale + tangenziale (consensus) e aggiorna le velocità.
  4. Integrazione di posizione tramite `Drone.update_position(dt)`.

## MODELLO DEL DRONE

Stato interno (variabili principali):
- Posizione: `x`, `y`.
- Velocità: `vx`, `vy`.
- Orientamento/heading: `theta` è presente in `SimulationEngine` quando impostato per `APPROACH` (non è inizializzato in `Drone.__init__` ma viene attribuito dai chiamanti).
- Modalità: `mode` in `Drone` → stringa: `"SEARCH"`, `"APPROACH"`, `"LOCKED"`.
- Sensori: `gps` (GPSSensor) e `camera` (CameraSensor).

Differenza tra modalità:
- `SEARCH`: comportamento di ricerca (inizializzazione di velocità casuale, guardare in `SimulationEngine.add_drone` e transizione basata su soglia di percezione).
- `APPROACH`: il drone ha rilevato la macchia vicino al centro della sua camera; segue controllo composto da:
  - Componente radiale per regolare il raggio alla distanza desiderata $r_0$.
  - Componente tangenziale per orbitarvi; include un termine di consensus per distribuire gli angoli.
- `LOCKED`: stato in cui il drone ferma la dinamica (velocità = 0). Usato per blocchi geometrici (se attivato).

Sensori:
- Propriocettivi:
  - `GPSSensor.sense((x,y))`: ritorna la posizione reale con rumore gaussiano opzionale.
- Esterocettivi:
  - `CameraSensor.sense(world_field, x, y, x_coords, y_coords)`: estrae una finestra locale (default 25×25) dalla griglia `world_field` centrata nella posizione $(x,y)$ e aggiunge rumore / blur opzionale.

## PERCEZIONE

Come si costruisce la `camera_view`:
- `CameraSensor.sense()` calcola gli indici di griglia corrispondenti a $(x,y)$:
  - usa `x_coords` e `y_coords` (lineari) per mappare $x$ e $y$ a indici: `i_center = int((x - x_coords[0]) / dx)` e `j_center = int((y - y_coords[0]) / dy)`.
  - estrae la sotto-matrice da `world_field[i_min:i_max, j_min:j_max]` e la normalizza come float; pad se vicino ai bordi.
- Nota implementativa importante: la mappatura usa `x` per l'indice riga e `y` per l'indice colonna. Questo funziona coerentemente con la costruzione di `SimulationMap.X,Y` ma può risultare confusa (di solito si usa $i$ per y e $j$ per x). Tenere conto quando estendi il codice.

Rilevamento del bordo:
- Nel `SimulationEngine.step()` la misura principale è:
  - Estrae `camera_view` (25×25).
  - Calcola `center_val` come media della finestra centrale 5×5:
    - win = 2 → slice `h//2-win : h//2+win+1`
  - Se `center_val > c_star` (con `c_star = 0.5` di default), si considera che il drone "vede" la macchia al centro e passa in `APPROACH`.
- Funzioni di `edge_detection`:
  - `identify_centroid(camera_matrix)`: centro di massa dei pixel con valore > 0.5 — utile per stimare localmente la forma.
  - `check_geometric_lock(camera_matrix)`: controlla se il pixel centrale è olio e ha vicini acqua — condizione semplice per rilevare un "bordo netto".

Limiti della percezione binaria/soglia:
- Sensore locale non fornisce il bordo globale; solo un piccolo patch 25×25.
- Soglie fisse (`0.5`, `c_star`) sensibili a rumore e a scale del campo.
- Se il campo è sfumato (es. Gaussian), la decisione binaria può essere imprecisa.
- Camera centratissima: se il drone è appena fuori dal bordo, la macchia può essere interamente fuori dalla finestra → falso negativo.
- Raccomandazione: usare misure continue (gradiente locale, centroide) e filtri temporali per robustezza.

## CONTROLLO

L'architettura del controllo separa le direzioni radiale e tangenziale — questa separazione è alla base della stabilità e chiarezza del comportamento.

Terminologia e variabili chiave:
- Centro della macchia: $(x_0, y_0)$.
- Posizione drone: $(x,y)$.
- Distanza corrente: $d = \sqrt{(x-x_0)^2 + (y-y_0)^2}$.
- Angolo polare: $\phi = \operatorname{atan2}(y - y_0, x - x_0)$.
- Vettori locali:
  - radiale unitario: $\mathbf{r} = (\cos\phi, \sin\phi)$.
  - tangenziale unitario (CCW): $\mathbf{t} = (-\sin\phi, \cos\phi)$.
- Parametri principali:
  - $r_0$: raggio desiderato (dal modello `CircleOilSpill`).
  - $k_{radial}$: guadagno del controllo radiale (in codice: `k_radial`, = 1.5).
  - $v_{base}$: velocità tangenziale base (`v_base`, = 0.3).
  - $k_{consensus}$: guadagno consenso (`k_consensus`, = 3.0).
  - saturazioni: limite massimo su $u_\phi$ (angular correction), e clip su $v_{tangential}$.

### 5.1 Legge di Matveev (controllo radiale)
- Implementazione nel codice:
  - Radial error: $e_r = d - r_0$.
  - Comando radiale: $v_{radial} = -k_{radial} \cdot e_r$.
- Interpretazione fisica:
  - Se $d > r_0$ (siamo fuori), $e_r > 0$ → $v_{radial} < 0$ → componente radiale punta verso il centro, portandoci verso $r_0$.
  - Se $d < r_0$, la componente radiale spinge fuori.
- Perché funziona per orbiting:
  - Mantenendo la componente tangenziale non zero e regolando solo la componente radiale per portare $d \to r_0$, il drone converge su una circonferenza di raggio $r_0$. Con un opportuna tangenziale, la traiettoria diventa una rotazione attorno al centro.

### 5.2 Consensus (distribuzione uniforme)
- Concetto:
  - Per N droni sull'anello, osservare gli angoli $\phi_i$. L'obiettivo è che gli spazi angolari tra droni siano uniformi: idealmente ogni gap = $2\pi/N$.
- Implementazione:
  - Calcolo $\phi_i$ con `np.arctan2`.
  - Ordinamento per $\phi$ per definire l'ordine CCW.
  - gap_to_next = $(\phi_{i+1} - \phi_i) \bmod 2\pi$.
  - gap_from_prev = $(\phi_i - \phi_{i-1}) \bmod 2\pi$.
  - errori: $e_{next} = gap\_to\_next - ideal\_gap$, $e_{prev} = gap\_from\_prev - ideal\_gap$.
  - delta_gap = $e_{next} - e_{prev}$.
  - correzione angolare: $u_\phi = k_{consensus} \cdot delta\_gap$ (clipped).
  - Applicazione: tangential speed modificata come $v_{tan} = v_{base} + u_\phi d$ (poi clamp).
- Perché convergerà:
  - Quando i gap sono uguali, $gap\_to\_next = gap\_from\_prev = ideal\_gap$ → $e_{next}=e_{prev}=0$ → $delta\_gap = 0$ → $u_\phi=0$ (equilibrio).
  - Analogia: ogni drone usa solo informazione locale (angoli dei vicini ordinati) per smussare gli scarti. Il termine delta_gap agisce come un controllo proporzionale sugli scarti di spaziatura.
- Caso N=2:
  - ideal_gap = $\pi$. Se i due droni sono opposti, gap_to_next = gap_from_prev = $\pi$, delta_gap = 0 → equilibrio stabile (180°).
  - Se non opposti, delta_gap tende a generare velocità tangenziale che aumenti/decremente l'angolo relativo fino a $\pi$.

### 5.3 Separazione radiale / tangenziale
- Nel codice le componenti sono calcolate separatamente e poi sommate:
  - $ \mathbf{v} = v_{tan}\, \mathbf{t} + v_{radial}\, \mathbf{r}$.
- Perché fondamentale:
  - Mantiene la stabilità; il controllo radiale non interferisce direttamente con quello angolare.
  - Se non separate: un singolo vettore velocità con un controller combinato può indurre oscillazioni o oscillazioni accoppiate (allosync) e perdita di monotonicità di convergenza.
- Cosa succede se non si separano:
  - I comandi possono scambiarsi tra radiale e tangenziale causando scatti, overshoot del raggio e difficoltà di convergenza angolare.

### 5.4 Saturazione (clip)
- Nella pratica `u_\phi` viene limitato (`max_u_phi = 0.8`) e `v_tangential` è clamped tra 0.05 e 1.5.
- Perché serve:
  - Evita comandi eccessivi che potrebbero provocare instabilità numerica e oscillazioni molto grandi.
  - Riflette i limiti fisici reali di motori/attuatori.
- Effetto sulla stabilità:
  - Introduce non-linearità ma previene scatti. Con guadagno troppo alto senza saturazione la risposta potrebbe divergere o oscillare.

## DINAMICA DEL SISTEMA

Evoluzione temporale (riassunto di `SimulationEngine.step()`):
1. Aggiornamento frame counter.
2. Calcolo delle phi per droni in `APPROACH` e computazione del consensus quando N>1.
3. Per ogni drone:
   - Ottenimento `camera_view` = `Drone.get_camera_view(world_field, x_coords, y_coords)`.
   - Calcolo `center_val` (media 5×5 centrale).
   - Transizione `SEARCH -> APPROACH` se `center_val > c_star`. All'ingresso in APPROACH, `theta` inizializzato alla direzione tangenziale CCW.
   - Se in `APPROACH`: calcolo `v_radial` e `v_tangential` (con correzione consensus) e conversione in componenti (vx, vy).
   - Se `camera_view` perde traccia (max < 0.15) → ritorno a `SEARCH`.
   - Integrazione di posizione tramite `Drone.update_position(dt)` (clamp e rimbalzo ai bordi della mappa).

Transizione SEARCH → APPROACH:
- Basata su soglia `c_star` applicata alla media centrale 5×5. Questo vuol dire che il drone avverte "abbastanza olio" nel centro della sua vista e assume di essere sopra il bordo.

Comportamento emergente:
- Droni in SEARCH vagano; quando incontrano il bordo passano in APPROACH e convergono radialmente verso $r_0$. Con consensus attivo e con più droni sull'anello, si ottiene una ridistribuzione angolare verso spazi uniformi. Possono anche perdere il bordo e tornare a SEARCH se la macchia non è più visibile nella finestra.

## CONVERGENZA

Quando il sistema è stabile:
- Stato stabile locale: per ogni drone in APPROACH, $d \approx r_0$ e i gap angolari sono vicini all'ideal gap $2\pi / N$; quindi $v_{radial} \approx 0$ e $u_\phi \approx 0$.
- La convergenza del consensus è possibile perché il controllo è proporzionale sui gap angolari locali; se i guadagni e saturazioni sono ragionevoli, gli scarti vengono smorzati nel tempo.

Configurazione ideale:
- N droni uniformemente distribuiti su una circonferenza di raggio $r_0$.
- Per N=2 l'assetto ideale è $\phi_2 - \phi_1 = \pi$.

Perché il consensus si annulla:
- Per uniformità: gap_to_next = gap_from_prev = ideal_gap ⇒ errori nulli ⇒ delta_gap = 0 ⇒ $u_\phi = 0$. Quindi nessuna correzione tangenziale aggiuntiva è richiesta.

## LIMITAZIONI

- Percezione locale e soglie:
  - Soglie fisse (`c_star`, 0.5 threshold in `identify_centroid`) sensibili a scale del campo e rumore.
- Rumore:
  - `CameraSensor` e `GPSSensor` possono introdurre rumore; non c'è filtro temporale nel controllo (es. Kalman). Questo può portare a oscillazioni.
- Mapping coordinate-index:
  - `CameraSensor` usa `x` per riga e `y` per colonna; potenzialmente fonte di confusione e errori se si cambiano convenzioni.
- Dipendenza dai parametri:
  - Guadagni (`k_radial`, `k_consensus`), `v_base` e saturazioni determinano stabilità e velocità di convergenza; non adaptive.
- Bordi della mappa:
  - `Drone.update_position` rimbalza invertendo la componente di velocità; se i droni partono vicino al bordo potrebbero compiere rimbalzi indesiderati. (Si raccomanda inizializzare con margine interno.)

## POSSIBILI MIGLIORAMENTI

Percezione e stima del bordo:
- Usare deriva del campo (gradiente) o fit di cerchio locale (RANSAC / least squares) per stimare raggio e centro localmente invece di soglie.
- Usare `identify_centroid()` come iniziale ma poi applicare un fitting della curva locale per stimare $\phi$ e la normale.

Filtri e stima:
- Filtrare le misure GPS/camera con filtro di Kalman o exponential smoothing per rendere il controllo meno sensibile al rumore.
- Stima distribuita dello stato del gruppo (ogni drone condivide la sua $\phi$) per un consensus più efficiente.

Comunicazione e cooperazione:
- Implementare scambio esplicito di messaggi (posizione o phi) per migliorare il consensus e renderlo robusto a occlusioni/sfalsamenti locali.
- Protocollo per gestione ingressi/uscite (join/leave) dell'anello senza perturbazioni.

Controller più sofisticati:
- Aggiungere controllo PD sulla componente radiale per ridurre overshoot.
- Regolazione adattiva del `k_consensus` in funzione della densità.

Robustezza ai bordi della mappa:
- Inizializzare droni con margine (es. 10% interno) oppure generare posizioni distribuite gaussiane attorno al centro.

## COME ESEGUIRE IL CODICE

Prerequisiti:
- Python 3.8+ (testato con Python 3.x)
- Dipendenze: `numpy`, `matplotlib`
- Installa con pip:
```bash
python3 -m pip install numpy matplotlib
```

Esempi di esecuzione:
- Eseguire simulazione headless (salva immagine finale):
```bash
python3 main.py --frames 5000
```
- Eseguire con visualizzazione:
```bash
python3 main.py --visualize --frames 5000
```
- Debug rapido (script di test già presente):
```bash
python3 debug_consensus.py
```

Parametri utili (nel codice):
- Modificare in `simulation_engine.py`:
  - `c_star` (soglia per passare in APPROACH)
  - `v_base` (velocità tangenziale base)
  - `k_consensus` (guadagno consenso)
- Modificare in `environment.py`:
  - `CircleOilSpill.__init__(r0=..., sigma=...)` per cambiare raggio e morbidezza del bordo.
- Posizioni iniziali:
  - `main.py` ora inizializza posizioni uniformi nella mappa; per margine interno: usare `np.random.uniform(xmin+margin, xmax-margin)`.

## ESEMPI E DIAGRAMMI INTUITIVI

Vettori di controllo (diagramma ASCII, drone sopra il bordo, CCW tangential):
  
  Drone (x,y)
     |
     |  r (radiale)
     v
    o----> tangential (t, CCW)
     \\
      \\  radial towards/away from center

Rappresentazione controllo:
- calcolo vx,vy:
$$
\mathbf{v} = v_{tan}\,\mathbf{t} + v_{radial}\,\mathbf{r}
$$
dove
$$
v_{radial} = -k_{radial}(d - r_0), \quad v_{tan} = v_{base} + u_\phi d.
$$

## NOTE IMPLEMENTATIVE E "PERCHÉ" DELLE SCELTE

- Separazione radiale/tangenziale: scelta strutturale semplice e robusta che rende il comportamento prevedibile — il radiale regola il raggio, il tangenziale gestisce la distribuzione.
- Uso di una finestra centrale 5×5 e soglia `c_star`: trade-off tra semplicità e robustezza; la media centrale attenua rumore locale e l'uso di 5×5 è una via di mezzo tra sensibilità e stabilità.
- Consensus locale con ordinamento per $\phi$: evita necessità di comunicazione esplicita; i droni determinano un ordine topologico e regolano gap usando solo questo ordinamento.
- Saturazione: prevenire comandi troppo grandi dovuti a errori iniziali o rumore.
- Sensori modulari (`sensors.py`): facilita l'estensione (aggiunta di blur, diversa dimensione, variabilità di rumore).

## CONCLUSIONE RAPIDA

Questo progetto illustra un'architettura semplice ed efficace per il problema oil-spill edge tracking: percezione locale + controllo separato radiale/tangenziale + semplice consenso angolare. È un buon punto di partenza per sperimentazioni e miglioramenti: migliorare la percezione (fit/gradiente), introdurre filtraggio e comunicazione, e testare la sensibilità ai parametri per rendere il sistema più robusto nelle condizioni realistiche.

--- 

Se vuoi, posso:
- Aggiungere esempi di parametri per esperimenti (grid di valori per `k_consensus`, `v_base`).
- Implementare una versione con filtro (Kalman) e un semplice protocollo di comunicazione per il consensus.
# Oil Spill Multi-Drone Simulation

A modular and scalable Python simulation for autonomous drones detecting and tracking oil spills using noisy sensors.

## Architecture

- **`main.py`**: Entry point. Initializes the environment and multi-drone simulation.
- **`simulation_engine.py`**: The core simulation loop. Manages agent states and sensor polling.
- **`drone.py`**: Modular drone class with proprioceptive (GPS) and exteroceptive (Camera) support.
- **`sensors.py`**: Implementation of noisy sensors (Gaussian noise on GPS and Camera data).
- **`environment.py`**: Defines the simulation map and oil spill mathematical models.
- **`edge_detection.py`**: Geometric algorithms for identifying oil spill boundaries from local camera data.
- **`visualization.py`**: Decoupled visualization module using Matplotlib.

## Features

- **Multi-Drone Support**: Simulates multiple agents independently searching the map.
- **Sensor Noise**: Real-world-like noise modeled for GPS and local camera perception.
- **Geometric Locking**: Drones identify the exact "straddling" edge of the oil spill.
- **Modern Viz**: Clear feedback for drone states (Searching: RED, Locked: GREEN).

## How to Run

Install dependencies:
```bash
pip install numpy matplotlib
```

Run simulation:
```bash
python3 main.py
```
