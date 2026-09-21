# Controllo Distribuito di Copertura 1D su Bordo Mappato

## 1. Panoramica e Obiettivo del Sistema

Questo progetto implementa una simulazione multi-drone per il rilevamento, la mappatura e la copertura distribuita del bordo di una regione di interesse, modellata come una fuoriuscita di petrolio in un dominio bidimensionale. Il codice combina tre componenti principali:

- una fase di **mapping distribuito**, in cui ogni drone osserva localmente il campo ambientale, estrae punti di bordo e aggiorna una propria griglia di occupazione;
- una fase di **consenso su griglia**, in cui le mappe locali vengono fuse tramite scambio locale di messaggi;
- una fase di **controllo di copertura 1D**, in cui il bordo chiuso ricostruito viene trattato come una varieta' monodimensionale parametrizzata per lunghezza d'arco, e i droni si distribuiscono lungo il perimetro mediante partizionamento Voronoi 1D e target di Lloyd.

Il problema affrontato non e' un classico problema di coverage planare 2D. La regione da coprire e' il **confine** dell'area inquinata, cioe' una curva chiusa o aperta immersa in `R^2`. Dal punto di vista geometrico, la quantita' rilevante non e' la distanza euclidea nel piano, ma la distanza geodetica lungo il bordo ordinato. Il bordo viene quindi convertito in una catena discreta:

```text
B = {p_0, p_1, ..., p_{N-1}},    p_i in R^2
```

dove gli indici consecutivi rappresentano punti adiacenti sul perimetro. Se il bordo e' chiuso, anche `p_{N-1}` e `p_0` sono considerati adiacenti.

L'obiettivo finale della legge di controllo e' ottenere una distribuzione regolare dei robot lungo il bordo noto o mappato. Ogni drone:

- conosce la propria posizione;
- riceve, eventualmente in modalita' multi-hop, le posizioni note degli altri droni;
- proietta tali posizioni sul bordo;
- costruisce una partizione Voronoi 1D lungo la curva;
- calcola il centro della propria cella Voronoi lungo la coordinata d'arco;
- si muove verso tale centro rispettando una velocita' massima.

In termini concettuali, il sistema realizza una dinamica di Lloyd discreta su un grafo di bordo, con comunicazione distribuita e saturazione cinematica.

## 2. Modello di Sistema e Architettura

### 2.1 Moduli principali

La codebase e' organizzata nei seguenti moduli:

- `main.py`: entry point della simulazione, parsing degli argomenti, costruzione di ambiente, engine, droni e visualizzatore.
- `environment.py`: definizione del dominio spaziale e dei modelli di oil spill.
- `simulation_engine.py`: ciclo di simulazione, sensing, consenso, transizione di stato e applicazione delle azioni.
- `drone.py`: modello dell'agente drone, stato fisico, sensori, griglia locale, consenso locale e integrazione del moto.
- `controller.py`: controller distribuito, estrazione/ordinamento del bordo, Voronoi 1D, Multi-Source Shortest Path, target di Lloyd e legge di controllo.
- `sensors.py`: sensori GPS e camera.
- `edge_detection.py`: estrazione dei pixel di bordo da una misura locale.
- `visualization.py`: visualizzazione della mappa, dei droni, delle frecce di controllo, delle celle Voronoi 1D e dei target.
- `tests/`: test unitari per consenso, transizione mapping-Lloyd, bordo chiuso, multi-hop e Voronoi 1D.

### 2.2 Dominio fisico e campo ambientale

La classe `SimulationMap` in `environment.py` descrive il dominio fisico:

```python
SimulationMap(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), grid_size=500)
```

Essa costruisce:

- `x_coords`: campioni lungo l'asse `x`;
- `y_coords`: campioni lungo l'asse `y`;
- `X, Y`: meshgrid con convenzione `indexing="ij"`;
- `dx, dy`: passo spaziale della griglia fisica.

Il campo ambientale `world_field` e' una matrice 2D contenente valori reali, tipicamente in `[0, 1]`, interpretati come concentrazione o occupazione associata alla macchia di petrolio. Due modelli sono disponibili:

- `CircleOilSpill`: regione circolare con bordo ammorbidito;
- `SmoothedPolygonOilSpill`: poligono irregolare chiuso, smussato e campionato, usato per generare forme piu' realistiche.

La soglia principale di occupazione e':

```python
occupancy_threshold = 0.5
```

Essa separa la regione considerata occupata/inquinata dalla regione libera:

```text
occupied(x, y) = world_field(x, y) >= occupancy_threshold
```

### 2.3 Architettura della classe `Drone`

La classe `Drone` modella un agente mobile autonomo. Le variabili di stato fondamentali sono:

- `drone_id`: identificativo del robot;
- `x, y`: posizione nel sistema di riferimento globale della simulazione;
- `max_speed`: velocita' massima applicabile;
- `grid`: griglia di occupazione locale del drone;
- `gps`: istanza di `GPSSensor`;
- `camera`: istanza di `CameraSensor`;
- `known_positions`: dizionario delle posizioni note degli altri droni;
- `known_boundary_arcs`: dizionario delle coordinate d'arco note;
- `known_boundary_points`: punti del bordo noto o mappato;
- `boundary_s`: coordinata d'arco corrente del drone sul bordo;
- `boundary_index`: indice discreto piu' vicino sul bordo;
- `target_centroid`: target di Lloyd corrente in coordinate globali;
- `control_state`: stato del comportamento, inizialmente `"mapping"` e poi `"lloyd"`.

Il drone non contiene direttamente tutta la logica di controllo. Esso delega a un controller onboard:

```python
self.controller = DroneController(...)
```

Il metodo:

```python
Drone.compute_action(world_field=None, x_coords=None, y_coords=None)
```

chiama `self.controller.compute_action(...)` e restituisce il comando di velocita' desiderato.

### 2.4 Architettura della classe `Controller`

La classe `Controller` contiene le primitive geometriche e algoritmiche comuni:

- estrazione del bordo da griglia o punti;
- ordinamento del bordo;
- calcolo delle lunghezze d'arco;
- proiezione di punti sul bordo;
- Voronoi 1D via Multi-Source Shortest Path;
- target di Lloyd lungo la curva;
- azione verso il target con saturazione.

La sottoclasse `DroneController` aggiunge la logica specifica dell'agente:

- esplorazione casuale durante la fase di mapping;
- tracking locale del bordo quando il drone ha gia' celle occupate nella griglia;
- azione Lloyd durante la fase di copertura.

La distinzione e' importante:

- `Controller` e' una libreria geometrica e di coverage;
- `DroneController` e' il controller onboard che decide quale comportamento usare in base allo stato del drone.

### 2.5 Stati operativi del sistema

La simulazione opera con una macchina a stati implicita:

```text
mapping  ->  lloyd
```

Durante `mapping`:

- i droni osservano localmente il campo;
- aggiornano la griglia locale;
- eseguono consenso sulla griglia;
- cercano di rilevare la chiusura del bordo mappato.

Durante `lloyd`:

- il bordo chiuso rilevato viene distribuito a tutti i droni;
- ogni drone viene proiettato sul bordo;
- le posizioni vengono scambiate via multi-hop;
- ogni drone calcola la partizione Voronoi 1D e si muove verso il proprio target.

La transizione avviene in `SimulationEngine._transition_to_lloyd_state(boundary_points)`, chiamata quando `is_mapped_polygon_closed(...)` riconosce un loop chiuso nella mappa media.

### 2.6 Dinamica dei robot e limitazioni cinematiche

Il modello cinematico implementato e' un integratore semplice in piano:

```text
q_i(k+1) = q_i(k) + u_i(k) dt
```

dove:

- `q_i = [x_i, y_i]^T` e' la posizione del drone `i`;
- `u_i = [v_{x,i}, v_{y,i}]^T` e' il comando di velocita';
- `dt` e' il passo di integrazione.

Nel codice, l'integrazione avviene in:

```python
Drone.action(command, dt=1.0, bounds=None)
```

Il comando viene prima validato e saturato tramite:

```python
Drone._clip_command(command, max_speed)
```

La norma del comando e' limitata:

```text
u_clipped = u                         se ||u|| <= v_max
u_clipped = v_max * u / ||u||          se ||u|| > v_max
```

Il controller applica inoltre una saturazione preventiva con:

```python
Controller._clip_action(action, max_speed=0.12)
```

Quindi la limitazione cinematica e' presente su due livelli:

- nel controller, che genera azioni gia' compatibili con `max_speed`;
- nel modello del drone, che garantisce comunque il vincolo fisico prima di aggiornare `x, y`.

Dopo l'integrazione, la posizione viene limitata ai bounds del dominio:

```python
self.x = np.clip(self.x, x_bounds[0], x_bounds[1])
self.y = np.clip(self.y, y_bounds[0], y_bounds[1])
```

Nella fase Lloyd, dopo il movimento nel piano, il drone viene riproiettato sul bordo tramite:

```python
drone.project_to_boundary()
```

Questo rende il moto effettivo coerente con il modello 1D: il drone e' rappresentato da una posizione planare, ma il suo stato di controllo rilevante e' la coordinata lungo il bordo.

## 3. Modello dei Sensori e Processo di Misura

### 3.1 Sensori exteroceptive: percezione del campo ambientale

Il sensore exteroceptive principale e' la camera:

```python
CameraSensor(size=..., noise_std=..., occupancy_threshold=0.5)
```

Essa misura una finestra locale del campo `world_field` centrata sulla posizione corrente del drone. Il processo e' implementato in:

```python
CameraSensor.sense(world_field, x, y, x_coords, y_coords, occupancy_threshold=None)
```

La pipeline e':

1. validazione degli input;
2. estrazione di una finestra locale quadrata;
3. aggiunta di rumore gaussiano;
4. clipping dei valori in `[0, 1]`;
5. eventuale smoothing gaussiano;
6. sogliatura per stimare la frazione occupata;
7. edge detection;
8. conversione dei punti di bordo da coordinate immagine a coordinate globali.

#### Estrazione della finestra locale

La funzione:

```python
CameraSensor._extract_local_window(...)
```

calcola gli indici centrali:

```text
i_center = round((x - x_coords[0]) / dx)
j_center = round((y - y_coords[0]) / dy)
```

La finestra ha dimensione fissa `(size, size)`. Se il drone si trova vicino al bordo del dominio, la finestra viene completata con padding a zero. Questo significa che la camera restituisce sempre una misura con la stessa forma, anche quando il campo visivo esce parzialmente dal dominio simulato.

#### Rumore di misura

Il rumore e' additivo, gaussiano e a media nulla:

```text
z = h(q) + eta,    eta ~ N(0, sigma^2)
```

Nel codice:

```python
Sensor.add_noise(value)
```

Se `noise_std <= 0`, il valore viene restituito senza perturbazione. Per la camera, il valore rumoroso viene poi saturato:

```python
noisy_matrix = np.clip(noisy_matrix, 0.0, 1.0)
```

#### Soglia di occupazione

La soglia `occupancy_threshold` viene usata per classificare i pixel locali:

```python
binary_window = noisy_matrix >= threshold
oil_fraction = mean(binary_window)
```

`oil_fraction` rappresenta la frazione di pixel del campo visivo classificati come occupati. Non entra direttamente nella legge Lloyd, ma viene memorizzata come informazione diagnostica nello stato del drone.

#### Edge detection

Il bordo locale viene estratto in `edge_detection.detect_edges(...)`.

Il metodo non usa Canny classico. Opera invece come estrazione di frontiera da una maschera sogliata:

1. se il contrasto dell'immagine e' troppo basso (`val_max - val_min < 0.15`), restituisce una maschera vuota;
2. applica, se disponibile, un filtro gaussiano;
3. costruisce la maschera `mask = smoothed >= threshold`;
4. erode la maschera;
5. definisce il bordo come differenza tra maschera originale ed erosa:

```text
edges = mask XOR erosion(mask)
```

o, nel fallback senza `scipy`, come:

```text
edges = mask AND NOT erosion(mask)
```

`extract_edge_points(edges)` converte la maschera binaria in punti immagine:

```text
[(row_1, col_1), ..., (row_M, col_M)]
```

#### Conversione da coordinate locali a globali

La camera produce punti in coordinate immagine. La funzione:

```python
CameraSensor._local_to_world_coordinates(...)
```

li converte in coordinate globali:

```text
world_x = center_world_x + (row - center_row) * dx
world_y = center_world_y + (col - center_col) * dy
```

Il risultato e' un array `(M, 2)` di punti `[x, y]` nel sistema di riferimento globale della simulazione.

### 3.2 Aggiornamento dello stato sensoriale del drone

Il metodo:

```python
Drone.sense(world_field, x_coords, y_coords)
```

riceve la misura della camera e aggiorna variabili locali:

- `edge_detected`: vero se sono presenti punti di bordo;
- `last_edge_points`: tutti i punti di bordo globali rilevati;
- `last_nls_points`: copia dei punti di bordo, usata per visualizzazione;
- `last_edge_count`: numero di punti rilevati;
- `last_oil_fraction`: frazione occupata nella finestra;
- `last_boundary_anchor_point`: media filtrata dei punti di bordo;
- `last_edge_point`: punto di bordo piu' vicino alla posizione del drone.

Il filtro sull'anchor point e':

```text
a_k = 0.8 a_{k-1} + 0.2 mean(edge_points)
```

Questo non e' un filtro di stato completo, ma una semplice media esponenziale per stabilizzare un riferimento locale del bordo osservato.

### 3.3 Aggiornamento della griglia locale

I punti di bordo vengono fusi nella griglia locale del drone con:

```python
Drone.update_grid(edge_points, x_min, y_min, resolution, point_radius_cells=...)
```

La conversione punto-cella e':

```text
ix = int((x - x_min) / resolution)
iy = int((y - y_min) / resolution)
```

Ogni punto valido marca una cella occupata. Se `point_radius_cells > 0`, viene marcata anche un'impronta circolare discreta attorno alla cella:

```text
dx^2 + dy^2 <= radius^2
```

La fusione non usa media temporale, nonostante esista un parametro storico `alpha`. Nel codice attuale:

```python
self.grid = (np.maximum(self.grid, measurement_grid) > 0.0).astype(float)
```

Quindi la mappa locale e' binaria e monotona: una cella gia' osservata come occupata rimane occupata.

### 3.4 Estrazione di un bordo noto da campo o griglia

La funzione:

```python
Controller.initialize_known_boundary(world_field_or_points, x_coords=None, y_coords=None, force_closed=True)
```

accetta due tipi di input:

- un array `(N, 2)` di punti di bordo gia' noti;
- una matrice 2D, interpretata come campo o griglia di occupazione.

Nel primo caso, i punti vengono copiati direttamente:

```python
self.known_boundary_points = pts
```

Nel secondo caso:

1. il campo viene convertito a `float`;
2. viene calcolata la maschera:

```python
occupied = field >= self.occupancy_threshold
```

3. se la maschera e' vuota, il bordo noto diventa vuoto;
4. si prova a estrarre un contorno ordinato tramite `contourpy`;
5. se `contourpy` non e' disponibile o non produce contorni, si ricade su una ricerca di celle di bordo.

#### Estrazione con `contourpy`

`_extract_ordered_contour(...)` usa:

```python
contourpy.contour_generator(...).lines(self.occupancy_threshold)
```

Tra tutte le isolinee, sceglie quella di lunghezza massima. Questo e' coerente con l'assunzione che il bordo principale della macchia sia la componente chiusa dominante.

Se il primo e l'ultimo punto coincidono numericamente, l'ultimo viene rimosso:

```python
if ||contour[0] - contour[-1]|| < 1e-9:
    contour = contour[:-1]
```

La chiusura non e' rappresentata duplicando il primo punto alla fine, ma con il flag logico `known_boundary_closed`.

#### Fallback tramite maschera

`_extract_boundary_mask_points(occupied, x_coords, y_coords)` scansiona tutte le celle occupate. Una cella occupata viene considerata di bordo se almeno uno degli 8 vicini e' libero:

```text
occupied[ix, iy] = True
exists neighbor free  =>  boundary_mask[ix, iy] = True
```

I punti risultanti sono coordinate mondo associate ai campioni `x_coords[ix], y_coords[iy]`. Questo fallback puo' non garantire un ordinamento geometrico perfetto, motivo per cui esiste `_ensure_ordered_closed_boundary`.

### 3.5 Sensori proprioceptive e posizionamento

Il sensore proprioceptive e':

```python
GPSSensor(noise_std=gps_noise)
```

Il metodo:

```python
Drone.get_gps_pos()
```

restituisce:

```python
self.gps.sense(self.position)
```

dove:

```python
Drone.position = np.array([self.x, self.y])
```

Anche il GPS usa rumore gaussiano additivo tramite `Sensor.add_noise`. La posizione misurata e' usata nello scambio multi-hop della fase Lloyd:

```python
SimulationEngine._sensed_position(drone)
```

Se il drone possiede `get_gps_pos`, viene usato il GPS. In caso contrario viene usata la posizione reale.

### 3.6 Coordinate locali, globali e coordinate d'arco

Il sistema usa tre rappresentazioni spaziali:

1. **coordinate immagine locali** della camera, cioe' righe e colonne nella finestra sensoriale;
2. **coordinate globali 2D** `[x, y]` nel dominio della simulazione;
3. **coordinate d'arco 1D** `s` lungo il bordo ordinato.

La conversione locale-globale e' gestita dalla camera. La conversione globale-arco e' gestita dal controller.

#### Lunghezze d'arco discrete

Dato un bordo ordinato:

```text
B = {p_0, p_1, ..., p_{N-1}}
```

`Controller._boundary_arc_lengths(boundary_points, is_closed)` calcola:

```text
s_0 = 0
s_i = sum_{k=0}^{i-1} ||p_{k+1} - p_k||,    i = 1,...,N-1
```

La lunghezza totale e':

```text
L = s_{N-1}                              se bordo aperto
L = s_{N-1} + ||p_0 - p_{N-1}||          se bordo chiuso
```

Il vettore `arc_lengths` contiene le coordinate d'arco dei campioni discreti. `total_length` contiene la lunghezza totale.

#### Proiezione di una posizione sul bordo

La funzione:

```python
Controller._arc_length_at_position(boundary_points, arc_lengths, position, total_length, is_closed)
```

proietta una posizione planare `q` sul segmento di bordo piu' vicino. Per ogni segmento:

```text
a = p_i
b = p_{i+1}
v = b - a
t = clip(((q - a) dot v) / (v dot v), 0, 1)
projection = a + t v
```

Si seleziona il segmento con distanza euclidea minima:

```text
argmin_i ||q - projection_i||^2
```

La coordinata d'arco corrispondente e':

```text
s = s_i + t ||p_{i+1} - p_i||
```

Per bordo chiuso, `s` viene riportata modulo `L`.

Il metodo restituisce:

- `best_s`: coordinata d'arco;
- `best_point`: punto proiettato in `R^2`;
- `best_index`: indice discreto piu' vicino.

#### Interpolazione inversa: da arco a punto

La funzione:

```python
Controller._point_at_arc_length(boundary_points, arc_lengths, arc_length, total_length, is_closed)
```

fa l'operazione inversa: dato `s`, trova il segmento che contiene quella coordinata d'arco e interpola linearmente tra i due campioni:

```text
p(s) = (1 - t) p_i + t p_{i+1}
```

Per bordi chiusi, `s` e' considerato modulo `L`; per bordi aperti, `s` viene clippato in `[0, L]`.

### 3.7 Rumore, discretizzazione e stato locale del sensore

Il sistema gestisce rumore e discretizzazione in modo pragmatico:

- il rumore sensoriale e' additivo gaussiano sia per GPS sia per camera;
- il campo viene campionato su griglie discrete;
- i punti di bordo rilevati dalla camera sono quantizzati dalla risoluzione spaziale della mappa;
- la mappa locale e' binaria, non probabilistica;
- il consenso e' una fusione per massimo, quindi non riduce probabilisticamente incertezze o falsi positivi;
- la proiezione su bordo riduce l'errore laterale rispetto alla curva durante Lloyd;
- l'ordinamento e l'interpolazione per arco mitigano la natura discreta del bordo trasformando una sequenza di punti in una curva polilineare.

In particolare, durante Lloyd, lo stato locale usato per il controllo non e' la misura raw della camera, ma:

```text
(known_boundary_points, known_positions, known_boundary_arcs)
```

cioe':

- una rappresentazione discreta condivisa del bordo;
- un insieme di posizioni note dei robot;
- eventuali coordinate d'arco gia' note, piu' stabili della sola proiezione euclidea.

## 4. Protocollo di Comunicazione Multi-Hop

### 4.1 Comunicazione per il consenso sulla mappa

Durante la fase `mapping`, i droni scambiano messaggi di griglia:

```python
Drone.create_consensus_message()
```

Il messaggio contiene:

```python
{
    "sender_id": drone_id,
    "grid": copy_of_local_grid,
}
```

`SimulationEngine._exchange_consensus_messages()` consegna a ogni drone i messaggi dei vicini entro raggio di comunicazione. Il vicino `j` e' raggiungibile da `i` se:

```text
||q_i - q_j|| <= communication_radius
```

oppure se `fully_connected=True`.

La fusione locale e':

```python
self.grid = (np.maximum.reduce(grids) > 0.0).astype(float)
```

Quindi il consenso sulla mappa e' una propagazione distribuita di occupazione binaria. Dopo un numero sufficiente di round e con grafo connesso, le celle osservate da un drone possono diffondersi agli altri.

### 4.2 Comunicazione multi-hop delle posizioni

Nella richiesta viene citata la funzione `_update_multihop_positions`. Nel codice attuale il ruolo corrispondente e' implementato da:

```python
SimulationEngine._exchange_positions_multihop()
```

Questa funzione viene chiamata all'inizio di:

```python
SimulationEngine._apply_lloyd_actions()
```

prima che i droni calcolino le azioni Lloyd.

### 4.3 Inizializzazione delle informazioni locali

All'inizio dello scambio multi-hop, l'engine misura le posizioni:

```python
sensed_positions = {
    drone.drone_id: self._sensed_position(drone)
}
```

e raccoglie le coordinate d'arco correnti:

```python
sensed_arcs = {
    drone.drone_id: drone.boundary_s
}
```

Ogni drone resetta poi la propria conoscenza locale alla sola informazione su se stesso:

```python
drone.known_positions = {
    drone.drone_id: sensed_positions[drone.drone_id].copy()
}
```

Se esiste una coordinata d'arco, viene memorizzata:

```python
drone.known_boundary_arcs[drone.drone_id] = boundary_s
```

Questo reset evita che posizioni obsolete rimangano indefinitamente nella memoria locale.

### 4.4 Raggio di comunicazione e distanza euclidea

Il raggio fisico di comunicazione e' derivato da un parametro espresso in celle:

```python
communication_radius = communication_radius_cells * 0.5 * (abs(dx) + abs(dy))
```

Nel caso standard, `dx` e `dy` provengono dalla `SimulationMap`. Due droni sono connessi se:

```python
np.linalg.norm(drone_i.position - drone_j.position) <= communication_radius
```

Questa e' una distanza euclidea nel piano, non una distanza lungo il bordo. La comunicazione resta quindi modellata come radio range isotropo in `R^2`.

### 4.5 Propagazione a cascata delle informazioni

La funzione esegue:

```python
hop_count = max(1, len(self.drones))
for _ in range(hop_count):
    ...
```

Ad ogni round:

1. per ogni coppia di droni `i, j`, se sono in range, `i` riceve il dizionario `known_positions` di `j`;
2. riceve anche `known_boundary_arcs` di `j`;
3. gli aggiornamenti vengono accumulati in buffer temporanei;
4. alla fine del round vengono applicati.

Il punto chiave e' che non viene scambiata solo la posizione del vicino diretto, ma l'intero insieme di posizioni note dal vicino:

```python
pending_updates[i].update(drone_j.known_positions)
```

Questo realizza una propagazione multi-hop. Se il grafo di comunicazione e' connesso, dopo un numero di iterazioni pari al diametro del grafo ogni drone puo' conoscere tutti gli altri. Il codice usa `len(drones)` round, un limite superiore semplice sul diametro massimo di un grafo connesso con `N` nodi.

La funzione protegge inoltre l'informazione del drone stesso:

```python
if known_id != drone_id:
    drone.known_positions[known_id] = position
drone.known_positions[drone_id] = sensed_positions[drone_id].copy()
```

Quindi un drone non puo' sovrascrivere la propria posizione corrente con una copia obsoleta ricevuta da altri.

## 5. Estrazione e Ordinamento del Bordo

### 5.1 Rappresentazione discreta del bordo

Il bordo noto e' memorizzato come:

```python
self.known_boundary_points: np.ndarray shape (N, 2)
self.known_boundary_closed: bool
self.known_boundary_ordered: bool
```

La chiusura non richiede che il primo punto sia ripetuto alla fine. Il bordo chiuso e' rappresentato dalla connessione implicita:

```text
p_{N-1} <-> p_0
```

Questa scelta evita duplicazioni nella parametrizzazione d'arco e semplifica il trattamento modulo `L`.

### 5.2 Rilevamento di chiusura durante il mapping

La chiusura del bordo mappato viene verificata in:

```python
SimulationEngine._dfs_polygon_closure_check(grid)
```

Il metodo:

1. estrae le celle occupate dalla griglia media;
2. costruisce un grafo di adiacenza 8-connesso;
3. mantiene solo celle con almeno due vicini, candidate a formare un loop;
4. verifica che la componente sia connessa;
5. rifiuta endpoint aperti;
6. calcola quante celle libere risultano racchiuse dal loop;
7. richiede un'area interna minima `closure_min_enclosed_false_cells`;
8. estrae una sequenza ordinata di punti del contorno.

L'area interna e' calcolata tramite flood fill delle celle libere connesse al bordo esterno della griglia. Le celle libere non raggiungibili dal bordo esterno sono considerate racchiuse.

Questa logica evita di passare a Lloyd quando i droni hanno osservato solo una catena aperta di bordo.

### 5.3 Ordinamento tramite contorno racchiuso

Quando disponibile, `_ordered_enclosed_contour_points_from_grid(...)` usa `contourpy` sulla maschera delle celle libere racchiuse. Questo produce una isolinea ordinata e continua del bordo interno del loop.

Il metodo verifica la continuita' con:

```python
_is_boundary_trace_continuous(points)
```

Il criterio e':

```text
max_i ||p_{i+1} - p_i|| <= 3 * resolution
```

Per il contorno estratto dalla regione racchiusa, nei test il vincolo e' anche piu' stretto, circa una cella.

### 5.4 Ordinamento di celle e fallback centerline

Se il contorno racchiuso non e' disponibile, il codice usa:

- `_order_loop_cells(cells, adjacency)`: DFS con scelta del prossimo vicino che minimizza il cambio di direzione;
- `_ordered_centerline_points_from_grid(grid)`: ordinamento angolare e media per bin per collassare un bordo spesso in una singola centerline.

Il metodo centerline e' rilevante per misure rasterizzate con spessore maggiore di una cella. Invece di mantenere un anello spesso, aggrega punti in bin angolari e ne calcola la media, ottenendo una sequenza monodimensionale piu' adatta al controllo 1D.

### 5.5 Ordinamento geometrico in `Controller._ensure_ordered_closed_boundary`

Quando il bordo non e' gia' marcato come ordinato, il controller applica:

```python
Controller._ensure_ordered_closed_boundary()
```

L'algoritmo e':

1. calcola per ogni punto la distanza al nearest neighbor;
2. usa la mediana di tali distanze come scala tipica di campionamento;
3. costruisce un ordinamento greedy nearest-neighbor partendo dal punto `0`;
4. controlla se l'ultimo punto ordinato e' vicino al primo;
5. se la distanza `last_to_first` e' minore di `3 * median_nn`, marca il bordo come chiuso;
6. altrimenti conserva i punti originali e marca il bordo come non chiuso.

In pseudocodice:

```text
order = [0]
while esistono punti non visitati:
    current = order[-1]
    next = argmin_{j non visitato} ||p_j - p_current||
    order.append(next)
```

Questo algoritmo e' semplice e adatto a bordi campionati densamente e senza forti auto-intersezioni. Non e' un solver ottimo di travelling salesman; e' un ordinatore geometrico locale coerente con il tipo di bordo prodotto dalla pipeline di mapping.

## 6. Partizionamento Voronoi 1D e Algoritmo MSSP

### 6.1 Bordo come grafo pesato

Il bordo ordinato viene interpretato come un grafo:

```text
G = (V, E)
```

dove:

```text
V = {0, 1, ..., N-1}
```

e gli archi sono:

```text
E = {(i, i+1)}                         per bordo aperto
E = {(i, i+1), (N-1, 0)}               per bordo chiuso
```

Il peso dell'arco tra due campioni consecutivi e' la distanza euclidea:

```text
w(i, j) = ||p_i - p_j||
```

La distanza usata per la partizione Voronoi non e':

```text
||p - q|| in R^2
```

ma la distanza minima lungo il grafo di bordo:

```text
d_B(i, seed) = shortest path distance on G
```

### 6.2 Multi-Source Shortest Path Voronoi

Il metodo:

```python
Controller.multi_source_shortest_path_voronoi(points, seeds, is_closed)
```

assegna ogni campione del bordo al seed piu' vicino in distanza d'arco discreta.

Ogni seed e' un dizionario con almeno:

```python
{
    "robot_id": ...,
    "index": ...
}
```

La coda di priorita' viene inizializzata con distanza zero per ogni seed:

```python
distances[index] = 0.0
owner[index] = robot_id
heapq.heappush(pq, (0.0, index, order, robot_id))
```

Poi viene eseguito un Dijkstra multi-sorgente:

```text
while pq not empty:
    pop nodo u con distanza minima
    for v in neighbors(u):
        new_dist = dist[u] + w(u, v)
        if new_dist < dist[v]:
            dist[v] = new_dist
            owner[v] = robot_id
```

Il risultato e':

- `owner`: array di lunghezza `N`, contenente l'id del drone proprietario di ogni punto del bordo;
- `distances`: distanza d'arco discreta dal seed proprietario.

Questo algoritmo costruisce una partizione Voronoi 1D sul bordo:

```text
V_i = {p_k in B : d_B(k, seed_i) <= d_B(k, seed_j), per ogni j}
```

### 6.3 Costruzione dei seed

I seed sono creati in:

```python
Controller.compute_ring_ordering(current_drone, drones)
```

Per ogni posizione nota:

- se e' disponibile una coordinata d'arco in `known_boundary_arcs`, si usa quella;
- altrimenti si proietta la posizione planare sul bordo con `_arc_length_at_position`.

Ogni seed contiene:

```python
{
    "robot_id": drone_id,
    "index": seed_idx,
    "arc_length": seed_s,
    "position_on_boundary": projected,
}
```

I seed vengono ordinati per `arc_length`, ottenendo l'ordine dei droni lungo il perimetro.

### 6.4 Ring ordering: predecessore e successore

`compute_ring_ordering(...)` restituisce una struttura che include:

- `N`: numero di droni noti;
- `occupied_points`: punti del bordo;
- `assigned_drone_indices`: proprietario Voronoi di ogni punto;
- `distances`: distanze MSSP;
- `arc_lengths`: coordinate d'arco dei campioni;
- `total_boundary_length`: lunghezza totale;
- `seeds`: seed proiettati;
- `ring`: lista ordinata di dati per drone;
- `current`, `pred`, `succ`: drone corrente, predecessore e successore sul bordo;
- `center_of_mass`: centro medio dei punti di bordo, usato solo per diagnostica/logging.

Per un bordo chiuso, predecessore e successore sono calcolati in modo circolare:

```text
pred(i) = i - 1 mod M
succ(i) = i + 1 mod M
```

Questa struttura e' usata anche dalla visualizzazione per mostrare la partizione 1D.

### 6.5 Target di Lloyd lungo il bordo

Il metodo:

```python
Controller._lloyd_targets_from_seed_arcs(seeds, total_length, is_closed)
```

calcola i centroidi 1D delle celle, rappresentati come coordinate d'arco.

#### Caso bordo aperto

I seed sono ordinati:

```text
s_1 <= s_2 <= ... <= s_M
```

I confini tra celle sono i midpoint tra seed adiacenti:

```text
b_0 = 0
b_i = (s_i + s_{i+1}) / 2
b_M = L
```

La cella del robot `i` e':

```text
[b_{i-1}, b_i]
```

Il target di Lloyd e' il punto medio della cella:

```text
c_i = (b_{i-1} + b_i) / 2
```

#### Caso bordo chiuso

Per un bordo chiuso, le distanze devono essere considerate modulo `L`. Per ogni seed:

```text
left_gap  = (s_i - s_{i-1}) mod L
right_gap = (s_{i+1} - s_i) mod L
```

La lunghezza della cella e':

```text
cell_len = 0.5 * (left_gap + right_gap)
```

Il confine sinistro e':

```text
cell_start = (s_i - 0.5 * left_gap) mod L
```

Il confine destro e':

```text
cell_end = (s_i + 0.5 * right_gap) mod L
```

Il target e':

```text
target_s = (cell_start + 0.5 * cell_len) mod L
```

Il metodo restituisce per ogni drone:

```python
{
    "target_arc_length": ...,
    "cell_arc_length": ...,
    "cell_start_arc_length": ...,
    "cell_end_arc_length": ...,
}
```

`compute_ring_ordering(...)` converte poi `target_arc_length` in coordinate globali usando:

```python
_point_at_arc_length(...)
```

e memorizza il risultato in:

```python
drone.target_centroid
```

### 6.6 Relazione tra MSSP e target analitici

Nel codice convivono due livelli:

- `multi_source_shortest_path_voronoi(...)` assegna i campioni discreti del bordo ai seed piu' vicini;
- `_lloyd_targets_from_seed_arcs(...)` calcola target continui 1D usando midpoint delle coordinate d'arco.

Questa scelta e' utile perche':

- l'MSSP produce una partizione discreta visualizzabile e testabile;
- i target analitici evitano che il centro della cella dipenda troppo dalla densita' locale dei campioni;
- la legge di controllo puo' muoversi verso una coordinata d'arco continua, poi convertita in punto interpolato sulla polilinea.

## 7. Logica di Controllo

### 7.1 Controllo durante la fase di mapping

La funzione principale e':

```python
DroneController.compute_action(drone, world_field, x_coords, y_coords)
```

Se il drone e' in stato `"mapping"` e riceve il campo:

1. cerca un target nella propria griglia locale con `_grid_target(drone)`;
2. se esistono celle occupate, entra in modalita' `"boundary_tracking"`;
3. altrimenti usa esplorazione casuale.

#### Esplorazione

`_exploration_action(drone)` usa una direzione casuale normalizzata salvata in:

```python
drone.exploration_direction
```

Se il prossimo passo uscirebbe dal dominio, la componente corrispondente viene invertita. Il comando ha norma:

```python
exploration_speed = 0.08
```

#### Tracking locale del bordo

`_boundary_tracking_action(...)` interpola il campo nella posizione del drone e stima il gradiente con differenze finite:

```text
grad_x = (f(x + dx, y) - f(x - dx, y)) / (2 dx)
grad_y = (f(x, y + dy) - f(x, y - dy)) / (2 dy)
```

Il gradiente normalizzato approssima la normale al bordo:

```text
n = grad(f) / ||grad(f)||
```

Una tangente e' costruita ruotando la normale:

```text
t = [-n_y, n_x]
```

L'errore rispetto all'isolinea desiderata e':

```text
e = concentration - occupancy_threshold
```

Il comando combina:

- avanzamento tangenziale lungo il bordo;
- correzione normale per rimanere vicino alla soglia.

Nel codice:

```python
self.k_t * tangent - normal_gain * normal
```

dove:

```python
normal_gain = self.k_n * error + self.boundary_lock_gain * error
```

Il comando viene saturato con `_clip_action`.

### 7.2 Transizione da mapping a Lloyd

Ogni passo della simulazione esegue:

```python
SimulationEngine.step()
```

La sequenza e':

1. aggiornamento del campo ambientale;
2. misura locale se il frame e' un frame di misura;
3. consenso sulla griglia durante `mapping`;
4. calcolo della mappa media e dell'errore di disaccordo;
5. verifica di chiusura del bordo;
6. eventuale transizione a `lloyd`;
7. calcolo e applicazione delle azioni.

Quando viene rilevato un bordo chiuso:

```python
_transition_to_lloyd_state(boundary_points)
```

esegue:

- memorizzazione del bordo chiuso in `engine.closed_boundary_points`;
- cambio di stato globale a `"lloyd"`;
- cambio di stato di ogni drone a `"lloyd"`;
- copia del bordo in ogni controller onboard;
- proiezione di ogni drone sul bordo;
- inizializzazione completa di `known_positions` e `known_boundary_arcs`.

### 7.3 Controllo Lloyd 1D

Durante `lloyd`, l'engine chiama:

```python
SimulationEngine._apply_lloyd_actions()
```

La sequenza e':

1. `_exchange_positions_multihop()`;
2. `drone.compute_action()` per ogni drone;
3. `drone.action(action, bounds=...)`;
4. `drone.project_to_boundary()`.

Il controller onboard usa:

```python
DroneController._lloyd_action(drone)
```

che a sua volta chiama:

```python
ring_info = self.compute_ring_ordering(drone, None)
action = self._equidistant_action(drone, ring_info, ...)
```

### 7.4 Legge di controllo lungo la coordinata d'arco

Il metodo:

```python
Controller._equidistant_action(drone, ring_info, ...)
```

calcola il comando verso il target di Lloyd.

Se sono disponibili:

- `target_arc_length`;
- `arc_lengths`;
- `total_boundary_length`;
- `occupied_points`;

allora il controllo avviene direttamente sulla coordinata d'arco.

Per bordo chiuso, l'errore orientato e':

```text
e_s = (s_target - s_current + 0.5 L) mod L - 0.5 L
```

Questa formula sceglie il verso piu' corto lungo il ciclo e restituisce un errore in:

```text
[-L/2, L/2)
```

Per bordo aperto:

```text
e_s = s_target - s_current
```

Il passo lungo arco e':

```text
Delta s = clip(k_t e_s, -v_max, v_max)
```

Il prossimo punto sulla curva e':

```text
q_next = p(s_current + Delta s)
```

Il comando planare restituito e':

```text
u = q_next - q_current
```

e viene nuovamente saturato:

```python
return self._clip_action(next_point - current_pos, max_speed=max_speed)
```

Il codice salva anche:

```python
drone.pending_boundary_s = next_arc
drone.pending_boundary_point = next_point
```

Quando `project_to_boundary` viene chiamato subito dopo l'integrazione, questi valori permettono di aggiornare coerentemente la coordinata d'arco senza riproiettare da zero una posizione gia' calcolata sulla curva.

### 7.5 Fallback planare

Se l'informazione d'arco non e' disponibile, `_equidistant_action` usa un controllo proporzionale planare:

```text
u = k_t (target - q)
```

anche questo saturato da `_clip_action`.

Nel flusso normale Lloyd, pero', la via d'arco e' disponibile e rappresenta il caso principale.

### 7.6 Modalita' di visualizzazione statica dei target

La visualizzazione non modifica la dinamica. Essa legge lo stato calcolato dai controller e lo rappresenta graficamente.

Nel piano 2D, `Visualizer.update_drone(...)` mostra:

- posizione del drone;
- raggio di comunicazione, se abilitato;
- freccia del comando `last_control_vector`;
- target di Lloyd `target_centroid` come marker a stella.

Nel pannello laterale, `Visualizer.update_ring_partition(...)` mostra la partizione 1D:

- asse orizzontale: lunghezza d'arco del bordo;
- segmenti colorati: celle Voronoi dei droni;
- cerchi: posizione/seed corrente;
- stelle: target di Lloyd.

Se il `ring_info` non e' ancora disponibile, il visualizzatore ricostruisce una rappresentazione dai droni e dal bordo noto usando le stesse primitive geometriche:

```python
Controller._boundary_arc_lengths(...)
Controller._arc_length_at_position(...)
Controller._lloyd_targets_from_seed_arcs(...)
```

Questa e' una modalita' di visualizzazione statica dei target: serve a mostrare dove i droni dovrebbero convergere, ma la legge di controllo effettiva resta nel controller.

## 8. Metriche, Diagnostica e Output

### 8.1 Errore di consenso

L'engine calcola la mappa media:

```python
mean_grid = mean(drone.grid for drone in drones)
```

e l'errore medio di disaccordo:

```text
E = mean_i ||grid_i - mean_grid||_2
```

Questo valore viene salvato in:

```python
engine.error_history
```

e usato per il plot di convergenza del consenso.

### 8.2 Output salvati

`main.py` salva output in `tmp_output/`, tra cui:

- `final_simulation_state.png`;
- `consensus_convergence.png`;
- `final_occupancy_grid.png`;
- `final_occupancy_grid_robot_D*.png`;
- `simulation_animation.gif`;
- `oil_mapping_data.npy`;
- `oil_mapping_metadata.json`.

`oil_mapping_data.npy` contiene un array `(N, 2)` di punti di bordo in coordinate mondo. Se la mappa consensuale non contiene abbastanza punti, il codice ricade sul bordo del modello di oil spill.

### 8.3 Test rilevanti

I test verificano proprieta' centrali del sistema:

- `test_mssp_assigns_open_boundary_by_arc_distance`: MSSP su bordo aperto;
- `test_mssp_assigns_closed_boundary_across_wraparound_edge`: MSSP su bordo chiuso con wrap-around;
- `test_compute_actions_updates_voronoi_and_moves_toward_lloyd_target`: azioni verso target Lloyd;
- `test_closed_lloyd_targets_split_boundary_by_arc_midpoints`: target su bordo chiuso;
- `test_multihop_does_not_overwrite_a_drone_own_current_position`: protezione della posizione propria nel multi-hop;
- `test_multihop_refreshes_stale_neighbor_positions_with_current_values`: aggiornamento di informazioni obsolete;
- `test_dfs_polygon_closure_detects_closed_loop_not_open_chain`: distinzione tra loop chiuso e catena aperta;
- `test_transition_to_lloyd_state_loads_boundary_into_each_drone`: corretta inizializzazione dello stato Lloyd.

## 9. Interpretazione Scientifica del Sistema

Il sistema puo' essere interpretato come una pipeline di stima e controllo distribuito:

```text
campo continuo
    -> misure locali rumorose
    -> punti di bordo locali
    -> griglie di occupazione locali
    -> consenso distribuito
    -> bordo chiuso ordinato
    -> grafo 1D pesato
    -> Voronoi geodetico multi-sorgente
    -> target di Lloyd
    -> controllo saturo lungo arco
```

La scelta fondamentale e' ridurre il problema di copertura del bordo a un problema 1D su una curva. Questa riduzione e' giustificata quando:

- il confine della regione e' l'oggetto da monitorare;
- la posizione normale al bordo deve essere vincolata;
- l'obiettivo e' distribuire i robot lungo il perimetro e non coprire l'area interna.

La metrica corretta diventa quindi la distanza d'arco e non la distanza euclidea planare. Per questo il codice usa Multi-Source Dijkstra sul bordo ordinato invece di un Voronoi 2D classico.

Il risultato e' un sistema distribuito in cui ogni drone puo', a partire dalla conoscenza propagata via multi-hop, ricostruire localmente una partizione coerente del perimetro e convergere verso una configurazione piu' uniforme lungo il bordo.

## 10. Esecuzione della Simulazione

Esempio headless:

```bash
python main.py --frames 500 --num-drones 5 --oil-shape smoothed_polygon --range-based
```

Esempio con visualizzazione:

```bash
python main.py --visualize --frames 500 --num-drones 5 --show-2d-voronoi
```

Parametri rilevanti:

- `--communication-radius-cells`: raggio di comunicazione espresso in celle della mappa fisica;
- `--measure-every`: intervallo tra misure camera successive;
- `--fully-connected`: abilita comunicazione globale;
- `--range-based`: forza comunicazione limitata dal raggio;
- `--closure-min-enclosed-false-cells`: area interna minima per dichiarare chiuso il bordo;
- `--mapping-point-radius-cells`: spessore discreto usato per rasterizzare punti di bordo;
- `--show-2d-voronoi`: mostra la partizione Voronoi 1D proiettata sul bordo 2D;
- `--no-save-gif`: disabilita il salvataggio della GIF.

## 11. Sintesi Concettuale

In sintesi, il progetto implementa un sistema di robotica mobile distribuita per:

- percepire localmente un bordo di oil spill tramite sensori rumorosi;
- fondere mappe locali binarie attraverso consenso distribuito;
- riconoscere un loop chiuso nel bordo mappato;
- convertire il bordo in una curva discreta parametrizzata per arco;
- scambiare posizioni e coordinate d'arco con comunicazione multi-hop;
- costruire una partizione Voronoi 1D tramite Multi-Source Dijkstra;
- calcolare target di Lloyd come centri di celle lungo il perimetro;
- muovere i droni con controllo saturo e riproiezione sul bordo.

Il contributo geometrico centrale e' la separazione tra spazio di immersione `R^2` e spazio di controllo `S^1` o intervallo 1D: i droni esistono e si muovono nel piano, ma la copertura e' definita sulla coordinata d'arco del bordo.

## 12. Valutazione Critica Rispetto alle Regole d'Esame

Questa sezione e' volutamente critica. Il progetto ha una buona base tecnica per un esame su sistemi distribuiti, ma non va presentato come se fosse gia' un lavoro scientifico completo. Per passare l'esame e difenderlo bene, bisogna essere molto chiari su cosa il sistema dimostra realmente e su quali sono i suoi limiti.

### 12.1 Giudizio sintetico

Il progetto e' **sufficiente e potenzialmente buono** per un esame universitario, soprattutto se il report spiega bene la parte distribuita e mostra risultati numerici convincenti. L'idea di trattare il bordo di una macchia d'olio come una varieta' 1D e di applicare Voronoi/Lloyd lungo la coordinata d'arco e' interessante, non banale e difendibile.

Tuttavia, il progetto non va venduto come sistema realistico di oil-spill tracking. E' principalmente un **simulatore didattico** di:

- percezione distribuita;
- consenso su mappa di occupazione;
- transizione da mapping a coverage;
- controllo distribuito su bordo noto/mappato.

La parte piu' forte e' la formulazione geometrica 1D e l'uso di Multi-Source Dijkstra sul bordo. La parte piu' debole e' la validazione sperimentale: se non vengono mostrati grafici, metriche e confronti numerici, il progetto rischia di sembrare solo una demo visiva.

### 12.2 Valutazione per criteri d'esame

#### Thoughtfulness and completeness of the project report: 0-10

Stima realistica: **6.5-8/10**, a seconda di quanto bene viene scritto il report.

Punti forti:

- il problema e' formulabile chiaramente come sistema distribuito di robotica mobile;
- ci sono robot, sensori, comunicazione, consenso, controllo e simulazione;
- il modello geometrico del bordo e' abbastanza originale;
- la pipeline mapping -> consenso -> bordo chiuso -> Lloyd e' articolata e interessante.

Punti deboli:

- molte scelte sono euristiche e non dimostrate formalmente;
- il consenso sulla griglia e' molto semplice, basato su massimo binario;
- non c'e' una vera analisi di stabilita' del controllo Lloyd;
- il rumore e' modellato in modo elementare;
- il sistema non confronta diversi algoritmi o baseline.

Per ottenere un voto decente sul report, bisogna evitare frasi vaghe come "il sistema funziona bene". Bisogna invece scrivere:

- quale problema distribuito viene risolto;
- quali informazioni sono locali e quali vengono comunicate;
- perche' la distanza corretta e' quella lungo bordo;
- quali approssimazioni sono state fatte;
- quali metriche mostrano che il comportamento migliora.

#### Quality of the results: 0-10

Stima realistica senza ulteriori esperimenti: **5-6/10**.

Stima realistica con grafici ben preparati: **6.5-8/10**.

I risultati attuali possono bastare per passare, ma solo se vengono presentati con evidenza numerica. Le regole d'esame chiedono esplicitamente almeno 2 pagine di risultati con dati e grafici. Quindi e' necessario mostrare almeno:

- andamento dell'errore di consenso nel tempo;
- numero di celle occupate note nella mappa media;
- frame o immagini prima/dopo la transizione a Lloyd;
- posizioni dei droni sul bordo finale;
- lunghezze delle celle Voronoi finali;
- errore di uniformita' lungo il bordo, ad esempio deviazione standard delle distanze d'arco tra droni;
- confronto tra comunicazione fully connected e range-based, anche minimale.

La figura finale della simulazione e la GIF sono utili, ma non sono sufficienti da sole. Una commissione puo' facilmente chiedere: "Dov'e' la prova quantitativa che il sistema converge?". Serve almeno un grafico che risponda a questa domanda.

#### Correctness of individual question/answer: 0-10

Stima realistica: **6-8/10**, se si studiano bene i concetti.

Le domande probabili riguarderanno:

- perche' questo e' un sistema distribuito;
- cosa viene comunicato tra droni;
- differenza tra consenso sulla mappa e multi-hop sulle posizioni;
- differenza tra Voronoi 2D e Voronoi 1D sul bordo;
- ruolo della soglia di occupazione;
- come si passa da coordinate `(x, y)` a coordinata d'arco `s`;
- cosa fa Dijkstra multi-sorgente;
- perche' serve saturare la velocita';
- quali limiti introduce la discretizzazione;
- cosa succede se il grafo di comunicazione non e' connesso.

Se queste risposte sono chiare, il progetto e' difendibile. Se invece si presenta il codice senza saper spiegare la geometria, il rischio e' alto.

### 12.3 Probabilita' di passare

Per puntare semplicemente al **18**, il progetto e' piu' che sufficiente, a condizione che:

- venga consegnato un report ordinato di 6-8 pagine;
- ci siano almeno 2 pagine di risultati con grafici veri;
- durante la discussione si sappia spiegare la pipeline distribuita;
- non si esageri dichiarando prestazioni o realismo non dimostrati.

Senza grafici numerici, il progetto potrebbe comunque essere interessante, ma la parte "Quality of the results" rischia di essere penalizzata pesantemente. In quel caso il voto dipenderebbe quasi tutto dalla qualita' della spiegazione orale.

Con un report scritto bene, figure leggibili e 3-4 metriche semplici, una valutazione complessiva plausibile potrebbe essere nell'intervallo:

```text
18-22: se i risultati sono basilari ma chiari
22-25: se il report e' solido e i grafici mostrano bene consenso e Lloyd
25+: solo se vengono aggiunti confronti, analisi dei limiti e discussione piu' scientifica
```

### 12.4 Cosa migliorare prima della consegna

Le priorita', in ordine, sono:

1. **Produrre grafici quantitativi.** Non basta mostrare animazioni. Servono curve e numeri.
2. **Misurare l'uniformita' finale sul bordo.** Ad esempio gap d'arco tra droni e deviazione standard rispetto al gap ideale `L/N`.
3. **Mostrare il ruolo della comunicazione.** Un confronto `fully_connected` vs `range_based` renderebbe il progetto molto piu' convincente.
4. **Spiegare chiaramente la transizione mapping -> Lloyd.** E' una parte importante e originale della pipeline.
5. **Dichiarare i limiti senza nasconderli.** Questo migliora la credibilita' del report.

### 12.5 Limiti tecnici da ammettere esplicitamente

I limiti principali sono:

- il campo ambientale e' simulato, non reale;
- la camera e' un modello semplificato;
- il GPS e' modellato con rumore gaussiano semplice;
- la mappa locale e' binaria, non probabilistica;
- il consenso usa un massimo logico, non un filtro bayesiano;
- il bordo deve diventare chiuso prima di passare a Lloyd;
- l'ordinamento del bordo puo' fallire su geometrie molto rumorose o auto-intersecanti;
- non e' dimostrata formalmente la stabilita' globale del controllo;
- non ci sono ostacoli, collision avoidance o vincoli dinamici realistici del drone.

Questi limiti non rendono il progetto insufficiente. Anzi, se discussi bene, mostrano consapevolezza tecnica. Il punto e' presentarli come assunzioni del simulatore, non come dettagli trascurabili.

### 12.6 Frase onesta da usare nel report

Una formulazione difendibile e':

```text
Il progetto non mira a realizzare un sistema operativo realistico per il monitoraggio ambientale, ma a studiare in simulazione una pipeline distribuita per la percezione e la copertura di un bordo mappato. Il contributo principale e' la riduzione del problema di coverage del confine a un problema 1D lungo la coordinata d'arco, risolto tramite partizionamento Voronoi geodetico e aggiornamento di Lloyd distribuito.
```

Questa frase e' importante perche' mette il progetto nel perimetro giusto: non promette troppo, ma valorizza cio' che il codice fa davvero.

## 13. Possibili Riferimenti Bibliografici per il Report

Questa sezione raccoglie riferimenti utili da inserire nel report. Non e' necessario citarli tutti: per un report di 6-8 pagine conviene usare pochi riferimenti, ma scelti bene e collegati direttamente alle parti del progetto.

### 13.1 Riferimenti essenziali

Questi sono i riferimenti piu' importanti per giustificare scientificamente il progetto.

#### Distributed control e reti robotiche

**Bullo, Cortes e Martinez - Distributed Control of Robotic Networks**

Riferimento consigliato per:

- modello di rete robotica distribuita;
- comunicazione locale;
- coordinamento multi-agente;
- deployment, rendezvous, coverage e boundary estimation.

Citazione:

```bibtex
@book{bullo2009distributed,
  author    = {Francesco Bullo and Jorge Cortes and Sonia Martinez},
  title     = {Distributed Control of Robotic Networks: A Mathematical Approach to Motion Coordination Algorithms},
  publisher = {Princeton University Press},
  year      = {2009},
  isbn      = {978-0-691-14195-4},
  url       = {https://fbullo.github.io/dcrn/}
}
```

Perche' e' utile nel report:

> Questo e' probabilmente il riferimento piu' adatto per collocare il progetto dentro il tema dei sistemi robotici distribuiti. Va citato nell'introduzione e nella sezione "Distributed system adopted".

#### Coverage control con reti di sensori mobili

**Cortes, Martinez, Karatas e Bullo - Coverage Control for Mobile Sensing Networks**

Riferimento consigliato per:

- coverage control;
- sensori mobili;
- partizioni Voronoi;
- legame tra controllo distribuito e funzioni di costo geometriche.

Citazione:

```bibtex
@article{cortes2004coverage,
  author  = {Jorge Cortes and Sonia Martinez and Timur Karatas and Francesco Bullo},
  title   = {Coverage Control for Mobile Sensing Networks},
  journal = {IEEE Transactions on Robotics and Automation},
  volume  = {20},
  number  = {2},
  pages   = {243--255},
  year    = {2004},
  doi     = {10.1109/TRA.2004.824698}
}
```

Perche' e' utile nel report:

> E' il riferimento principale per giustificare l'uso di Voronoi e Lloyd-like algorithms in reti di sensori mobili. Nel tuo progetto la differenza importante e' che il coverage non e' 2D sull'area, ma 1D lungo il bordo.

#### Lloyd algorithm

**Lloyd - Least Squares Quantization in PCM**

Riferimento consigliato per:

- Lloyd algorithm;
- aggiornamento verso centroidi;
- interpretazione come procedura iterativa di quantizzazione/partizionamento.

Citazione:

```bibtex
@article{lloyd1982least,
  author  = {Stuart P. Lloyd},
  title   = {Least Squares Quantization in PCM},
  journal = {IEEE Transactions on Information Theory},
  volume  = {28},
  number  = {2},
  pages   = {129--137},
  year    = {1982},
  doi     = {10.1109/TIT.1982.1056489}
}
```

Perche' e' utile nel report:

> Va citato quando descrivi il calcolo dei target di copertura come centroide della cella. Nel progetto il Lloyd update e' adattato a celle 1D lungo lunghezza d'arco.

#### Dijkstra e shortest paths

**Dijkstra - A Note on Two Problems in Connexion with Graphs**

Riferimento consigliato per:

- shortest path;
- Multi-Source Dijkstra;
- grafo del bordo;
- distanza geodetica discreta.

Citazione:

```bibtex
@article{dijkstra1959note,
  author  = {Edsger W. Dijkstra},
  title   = {A Note on Two Problems in Connexion with Graphs},
  journal = {Numerische Mathematik},
  volume  = {1},
  pages   = {269--271},
  year    = {1959},
  doi     = {10.1007/BF01386390}
}
```

Perche' e' utile nel report:

> Serve per giustificare l'algoritmo di shortest path usato nella partizione Voronoi 1D. Nel codice l'MSSP e' una versione multi-sorgente di Dijkstra sul grafo del bordo.

#### Consensus multi-agente

**Olfati-Saber, Fax e Murray - Consensus and Cooperation in Networked Multi-Agent Systems**

Riferimento consigliato per:

- consensus;
- cooperazione multi-agente;
- topologie di comunicazione;
- robustezza rispetto a cambiamenti del grafo.

Citazione:

```bibtex
@article{olfatisaber2007consensus,
  author  = {Reza Olfati-Saber and J. Alex Fax and Richard M. Murray},
  title   = {Consensus and Cooperation in Networked Multi-Agent Systems},
  journal = {Proceedings of the IEEE},
  volume  = {95},
  number  = {1},
  pages   = {215--233},
  year    = {2007},
  doi     = {10.1109/JPROC.2006.887293}
}
```

Perche' e' utile nel report:

> Va citato nella parte sulla comunicazione e sul consenso distribuito. Anche se il consenso implementato nel progetto e' semplice, questo riferimento colloca il meccanismo dentro la teoria dei networked multi-agent systems.

### 13.2 Riferimenti consigliati ma non obbligatori

Questi riferimenti servono se vuoi rendere il report piu' robusto o se hai spazio nella bibliografia.

#### Mobile robotics generale

**Siegwart, Nourbakhsh e Scaramuzza - Introduction to Autonomous Mobile Robots**

Utile per:

- modello generale di robot mobile;
- sensori;
- localizzazione;
- percezione;
- motion planning.

Citazione:

```bibtex
@book{siegwart2011mobile,
  author    = {Roland Siegwart and Illah Reza Nourbakhsh and Davide Scaramuzza},
  title     = {Introduction to Autonomous Mobile Robots},
  edition   = {2},
  publisher = {MIT Press},
  year      = {2011},
  isbn      = {978-0-262-01535-6}
}
```

Uso consigliato:

> Citalo nella sezione "System model", soprattutto per motivare robot mobili, sensori e rappresentazione cinematica.

#### Robotica probabilistica e percezione

**Thrun, Burgard e Fox - Probabilistic Robotics**

Utile per:

- sensor models;
- rumore di misura;
- occupancy grids;
- localizzazione e mapping probabilistico.

Citazione:

```bibtex
@book{thrun2005probabilistic,
  author    = {Sebastian Thrun and Wolfram Burgard and Dieter Fox},
  title     = {Probabilistic Robotics},
  publisher = {MIT Press},
  year      = {2005},
  isbn      = {978-0-262-20162-9}
}
```

Uso consigliato:

> Questo riferimento e' utile anche per ammettere un limite: il tuo progetto usa occupancy grid binarie, non una formulazione probabilistica completa.

#### Metodi grafici per multi-agent networks

**Mesbahi e Egerstedt - Graph Theoretic Methods in Multiagent Networks**

Utile per:

- teoria dei grafi;
- reti multi-agente;
- protocolli di agreement;
- connettivita' e topologie di comunicazione.

Citazione:

```bibtex
@book{mesbahi2010graph,
  author    = {Mehran Mesbahi and Magnus Egerstedt},
  title     = {Graph Theoretic Methods in Multiagent Networks},
  publisher = {Princeton University Press},
  year      = {2010},
  isbn      = {978-0-691-14061-2},
  doi       = {10.1515/9781400835355}
}
```

Uso consigliato:

> Citalo se nel report enfatizzi il grafo di comunicazione, il multi-hop o il bordo come grafo pesato.

#### Planning e shortest paths

**LaValle - Planning Algorithms**

Utile per:

- grafi;
- path planning;
- ricerca su grafi;
- pianificazione in robotica.

Citazione:

```bibtex
@book{lavalle2006planning,
  author    = {Steven M. LaValle},
  title     = {Planning Algorithms},
  publisher = {Cambridge University Press},
  year      = {2006},
  url       = {https://lavalle.pl/planning/}
}
```

Uso consigliato:

> Non e' indispensabile, ma puo' aiutare a contestualizzare Dijkstra e l'uso di grafi in robotica.

### 13.3 Riferimenti da usare con attenzione

Alcuni riferimenti sono molto forti ma rischiano di allargare troppo il report. Usali solo se servono davvero.

#### Coordinamento e ottimizzazione geometrica

**Cortes e Bullo - Coordination and Geometric Optimization via Distributed Dynamical Systems**

Citazione:

```bibtex
@article{cortes2005coordination,
  author  = {Jorge Cortes and Francesco Bullo},
  title   = {Coordination and Geometric Optimization via Distributed Dynamical Systems},
  journal = {SIAM Journal on Control and Optimization},
  volume  = {44},
  number  = {5},
  pages   = {1543--1574},
  year    = {2005},
  doi     = {10.1137/S0363012903428652}
}
```

Uso consigliato:

> Da citare solo se vuoi discutere piu' formalmente il legame tra ottimizzazione geometrica, nonsmooth dynamics e coordinamento distribuito.

#### Consensus in multivehicle cooperative control

**Ren, Beard e Atkins - Information Consensus in Multivehicle Cooperative Control**

Citazione:

```bibtex
@article{ren2007information,
  author  = {Wei Ren and Randal W. Beard and Ella M. Atkins},
  title   = {Information Consensus in Multivehicle Cooperative Control},
  journal = {IEEE Control Systems Magazine},
  volume  = {27},
  number  = {2},
  pages   = {71--82},
  year    = {2007},
  doi     = {10.1109/MCS.2007.338264}
}
```

Uso consigliato:

> Puo' essere usato come riferimento piu' applicativo sul consensus in sistemi multi-veicolo. Non e' obbligatorio se citi gia' Olfati-Saber, Fax e Murray.

### 13.4 Bibliografia minima consigliata

Se vuoi tenere il report compatto, una bibliografia minima ma seria potrebbe essere:

1. Bullo, Cortes, Martinez - `Distributed Control of Robotic Networks`.
2. Cortes, Martinez, Karatas, Bullo - `Coverage Control for Mobile Sensing Networks`.
3. Lloyd - `Least Squares Quantization in PCM`.
4. Dijkstra - `A Note on Two Problems in Connexion with Graphs`.
5. Olfati-Saber, Fax, Murray - `Consensus and Cooperation in Networked Multi-Agent Systems`.
6. Thrun, Burgard, Fox - `Probabilistic Robotics`, solo se vuoi giustificare occupancy grids e rumore.

Questa lista basta per coprire:

- sistemi distribuiti;
- robotica multi-agente;
- coverage control;
- Lloyd/Voronoi;
- shortest path;
- consensus;
- sensori e mappe di occupazione.

### 13.5 Dove inserirli nel report

Una distribuzione ragionevole dei riferimenti e':

- **Abstract/Introduction**: Bullo et al. 2009, Cortes et al. 2004.
- **Problem formulation**: Cortes et al. 2004, Lloyd 1982.
- **Distributed system adopted**: Bullo et al. 2009, Olfati-Saber et al. 2007.
- **System model**: Siegwart et al. 2011, Thrun et al. 2005.
- **Proposed solution**: Lloyd 1982, Dijkstra 1959, Cortes et al. 2004.
- **Implementation**: LaValle 2006, Mesbahi and Egerstedt 2010 se parli di grafi.
- **Results and discussion**: richiama Cortes et al. 2004 per confrontare il concetto di coverage, ma specifica che il tuo caso e' coverage 1D su bordo.

### 13.6 Frase pronta per introdurre la bibliografia

Nel report puoi scrivere una frase simile:

```text
The project is inspired by classical distributed coverage control for mobile sensing networks, where Voronoi partitions and Lloyd-like updates are used to deploy mobile sensors. Differently from standard planar coverage formulations, the proposed simulator restricts the coverage domain to the one-dimensional boundary of a mapped region. The boundary is represented as a weighted graph, and geodesic Voronoi cells are computed through a multi-source shortest path procedure.
```

Questa frase collega direttamente i riferimenti al contributo specifico del progetto.
