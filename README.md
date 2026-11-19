# Distributed Oil Spill Tracking

Questo repository implementa una simulazione 2D di una squadra di droni che individua e mappa il bordo di una macchia di petrolio con sensori locali e scambio distribuito di mappe. Dopo aver stimato un contorno chiuso, i droni si distribuiscono lungo quel contorno e poi lo percorrono in pattuglia.

Questo documento descrive il comportamento effettivamente implementato nel codice, con attenzione ai concetti utili per un report di *Distributed Robot Perception*. Il simulatore mantiene internamente il campo di verità dell'olio per generare le misure; i droni ricevono soltanto finestre locali simulate dalla camera e le proprie stime GPS. La verità dell'ambiente serve al simulatore e alla visualizzazione, non viene passata direttamente al controllore dei droni.

## Indice

- [Obiettivo e architettura](#obiettivo-e-architettura)
- [Modello del mondo e della macchia](#modello-del-mondo-e-della-macchia)
- [Droni e sensori](#droni-e-sensori)
- [Rappresentazione probabilistica della mappa](#rappresentazione-probabilistica-della-mappa)
- [Comunicazione e consenso distribuito](#comunicazione-e-consenso-distribuito)
- [Le tre fasi operative](#le-tre-fasi-operative)
- [Sequenza di un passo di simulazione](#sequenza-di-un-passo-di-simulazione)
- [Struttura dei moduli](#struttura-dei-moduli)
- [Esecuzione e parametri](#esecuzione-e-parametri)
- [Output](#output)
- [Assunzioni e limiti del modello](#assunzioni-e-limiti-del-modello)

## Obiettivo e architettura

Il sistema affronta tre problemi in sequenza:

1. **Mapping**: esplorare l'ambiente, osservare localmente la macchia e costruire una mappa probabilistica condivisa.
2. **Copertura Lloyd 1D sul bordo** (*dynamic Lloyd coverage*): una volta stimato un contorno chiuso, assegnare a ogni drone una porzione del bordo e ridurre gli squilibri tra le posizioni dei droni.
3. **Boundary patrol**: dopo la convergenza della copertura, far avanzare i droni lungo il contorno mantenendo una formazione distribuita.

I componenti principali sono:

| Modulo | Responsabilità |
| --- | --- |
| `main.py` | Parametri CLI, inizializzazione, ciclo di esecuzione e salvataggio degli artefatti. |
| `environment.py` | Dominio spaziale e modelli statici/dinamici della macchia. |
| `simulation_engine.py` | Passi temporali, acquisizione sensoriale, messaggi, consenso, diagnostica e applicazione del moto. |
| `drone.py` | Stato locale del drone, sensori, mappe e fusione delle misure. |
| `sensors.py` | Modelli GPS e camera e conversione delle osservazioni locali in coordinate mondo. |
| `edge_detection.py` | Estrazione del bordo dalla misura della camera. |
| `controller.py` | Controllo in mapping, rilevamento/consenso del contorno, partizione 1D Lloyd e pattugliamento. |
| `visualization.py` | Viste della simulazione, mappe, diagnostica e GIF. |

## Modello del mondo e della macchia

### Dominio e griglie

`SimulationMap` definisce il campo fisico usato per generare e campionare la macchia. Con i valori predefiniti del programma, il dominio è `[-7, 7] × [-7, 7]`, discretizzato su una griglia `500 × 500`. Gli array `X` e `Y` sono creati con `meshgrid(indexing="ij")`; perciò la prima dimensione degli array corrisponde all'asse x e la seconda all'asse y. In questa griglia il passo è circa `14/499 = 0.0281` unità.

Il motore crea separatamente la griglia delle mappe dei droni, che copre `[-10, 10] × [-10, 10]` con risoluzione `0.1`, cioè `200 × 200` celle. La distinzione è importante: il campo continuo/campionato della simulazione alimenta la camera, mentre i dati osservati vengono rasterizzati nella griglia di occupancy più grossolana.

I droni vengono inizializzati in posizioni casuali uniformi nel dominio fisico. Le direzioni di esplorazione iniziali hanno angoli equispaziati attorno al cerchio, con un offset casuale comune; ciò evita di inizializzare tutta la squadra con heading identici. Con `seed=42` le estrazioni pseudo-casuali sono riproducibili.

### Campo di concentrazione

Il modello espone un campo scalare nell'intervallo `[0, 1]`: valori alti rappresentano olio, valori bassi acqua libera. Il controllore usa normalmente `occupancy_threshold = 0.6` per classificare una cella/pixel come occupata.

- **Macchia circolare** (`CircleOilSpill`): concentrazione `1` entro il raggio; all'esterno la concentrazione decresce con una funzione gaussiana in funzione della distanza dal bordo. Il parametro `sigma` regola la larghezza della transizione.
- **Poligono smussato** (`SmoothedPolygonOilSpill`, scelta predefinita): genera angoli casuali e raggi perturbati intorno al centro. L'irregolarità dipende da `smoothness`; la curva chiusa viene interpolata con spline periodica quando SciPy è disponibile, altrimenti con una curva Catmull–Rom. Il campo combina appartenenza al poligono e distanza firmata dal bordo con una transizione logistica. Il componente connesso più grande del poligono campionato viene mantenuto.

Entrambe le macchie supportano dinamica opzionale. Una velocità di deriva trasla il centro e il bordo a ogni passo di `dt`; una velocità di espansione aumenta il raggio (o scala radialmente i vertici nel caso del poligono). Se non si specificano `--dynamic-move` e `--dynamic-expand`, la macchia resta immobile. La direzione di deriva, se richiesta, viene estratta una volta usando il seed della simulazione; il modulo è la velocità richiesta.

Il modello della macchia è dunque la *ground truth* controllata del problema di percezione. L'algoritmo dei droni non legge i vertici né il raggio reale per stimare la mappa: legge le misure locali prodotte dalla camera.

## Droni e sensori

Ogni `Drone` mantiene posizione fisica `(x, y)`, stima GPS, controller locale, direzione di esplorazione, stato operativo e le seguenti mappe, tutte con la stessa forma:

- `grid`: probabilità di occupazione per cella;
- `occupancy_signal_grid`: somma/valore di evidenza di occupazione;
- `information_grid`: quantità di informazione disponibile;
- `boundary_grid`: maschera binaria del bordo derivata dalla mappa.

Le celle mai viste rimangono incerte: probabilità `0.5` e informazione `0`. Una cella non osservata non viene quindi interpretata come acqua.

La velocità massima del drone è `0.12` unità mondo per passo e quella di esplorazione è impostata allo stesso valore. Il simulatore applica comandi come velocità e non modella accelerazione o dinamica d'assetto; se una posizione oltrepassa il confine fisico, la traiettoria viene riflessa sulla parete.

### GPS

`GPSSensor` aggiunge rumore gaussiano indipendente a x e y. In `main.py` il rumore standard è `0.03` unità mondo. La stima risultante viene memorizzata in `last_gps_position` e usata per posizionare la finestra della camera, stimare i vicini di comunicazione, proiettare i droni sul bordo e calcolare il controllo. La dinamica fisica resta invece nello stato `(x, y)` senza rumore.

### Camera

La `CameraSensor` estrae una finestra quadrata centrata sulla stima GPS. La dimensione predefinita è `120 × 120` pixel. I pixel provengono dal campo del mondo campionato; in prossimità del margine del dominio la finestra viene completata con zeri. Al campo locale viene aggiunto rumore gaussiano (deviazione standard predefinita `0.03`) e i valori vengono limitati a `[0,1]`.

Per ogni misura la camera calcola:

- `oil_fraction`: frazione di pixel oltre la soglia di occupancy;
- `edge_points`: punti del bordo trasformati da indici locali `(riga, colonna)` a coordinate mondo `(x, y)`;
- l'immagine locale rumorosa, conservata dal drone per aggiornare la mappa.

Il blur preventivo esplicito della camera è disattivato per impostazione predefinita. Il rilevatore di bordo applica comunque una leggera levigatura gaussiana (sigma `1.0`) prima dell'estrazione della maschera. La conversione in coordinate mondo usa la spaziatura della griglia fisica e il centro della finestra; non è una ricostruzione prospettica di una camera con orientamento o modello ottico.

### Rilevamento del bordo

`detect_edges` rifiuta prima immagini vuote o con contrasto complessivo inferiore a `0.15`, poi confronta l'immagine levigata con la soglia (`0.6` nella configurazione del motore). Il contorno è la differenza tra la maschera binaria e la sua erosione: produce una fascia di bordo di circa un pixel. Se SciPy è disponibile si usano `gaussian_filter` e `binary_erosion`; in caso contrario il codice fornisce un'erosione locale di fallback.

Questo è un estrattore di isolinee/bordo da una misura di concentrazione, non un Canny parametrizzato dalle soglie `edge_threshold1/2` (i campi esistono nella classe sensore ma non sono usati da questo algoritmo). I punti rilevati sono anche convertiti dalla camera in coordinate globali; `last_edge_point` è il più vicino alla stima GPS.

## Rappresentazione probabilistica della mappa

Le osservazioni del footprint della camera sono proiettate sulla mappa `200 × 200`. Per ciascun pixel il drone determina la cella mondo corrispondente e classifica il campione come olio (`pixel >= 0.6`) oppure acqua. Se più pixel ricadono nella stessa cella, il valore della misura è la loro frazione occupata. Per le celle viste in quella misura:

\[
S_i \leftarrow m_i, \qquad I_i \leftarrow 1, \qquad p_i = S_i/I_i
\]

Qui `S` è `occupancy_signal_grid`, `I` è `information_grid`, `m` è la misura locale aggregata e `p` è la probabilità di occupazione. Le celle non viste conservano `I=0` e quindi `p=0.5`. Il codice sostituisce il valore locale della cella quando la osserva di nuovo; non implementa un filtro Bayesiano con log-odds né un accumulo temporale indipendente delle misure dello stesso drone.

La maschera di bordo locale individua le celle occupate note (`p >= 0.6` e `I > 0`) che hanno almeno un vicino libero noto (`p < 0.6`, informazione positiva) in una finestra 8-connessa. Le celle incerte non contano né come olio né come acqua libera. I centri delle celle di bordo vengono convertiti in coordinate mondo per il controllo e la visualizzazione.

## Comunicazione e consenso distribuito

### Grafo di comunicazione

In modalità range-based, due droni sono vicini se la distanza euclidea tra le rispettive stime GPS è al massimo `communication_radius`. Il parametro CLI è espresso in celle della griglia fisica e viene convertito in unità mondo con la spaziatura media `dx, dy`. Con i valori CLI predefiniti (`250` celle), il raggio è circa `7.0` unità. Con `--fully-connected` ogni drone comunica con tutti gli altri, indipendentemente dalla distanza.

Il grafo può quindi cambiare con le posizioni stimate e con il rumore GPS. Il motore non modella perdita, ritardo o banda del collegamento: i messaggi sono consegnati sincronicamente a tutti i vicini del round.

### Consenso sulla mappa

Ogni messaggio include la griglia probabilistica, `signal_grid`, `information_grid` e metadati di contorno/pattuglia. In ciascun round il motore prima crea una fotografia dei messaggi di tutti i droni, poi consegna ai vicini i messaggi di quella fotografia. Il drone aggiorna `S` e `I` con la media elemento per elemento tra la propria fotografia e quelle ricevute, e ricava nuovamente `p = S/I` dove `I>0` (altrimenti `0.5`). Scambiare signal e informazione separatamente preserva il peso dell'incertezza: una cella ignota non equivale direttamente a una misura di acqua.

Sono eseguiti tre round per passo di simulazione (`consensus_rounds=3`). Ripetere le medie lungo i vicini permette alla mappa di propagarsi oltre un singolo collegamento; tuttavia l'algoritmo è un consenso locale iterativo e non una media globale esatta garantita in un numero fisso di round. Una rete disconnessa può mantenere mappe diverse. Il valore d'errore registrato è la media delle norme L2 tra la griglia di ogni drone e la media aggregata del sistema; misura il disaccordo, non l'errore rispetto alla verità dell'olio.

### Consenso sui metadati del contorno

Separatamente dalle celle di mappa, ciascun drone cerca una proposta di contorno chiuso. La routine individua le componenti di celle libere della propria maschera di bordo non raggiungibili dai bordi esterni della griglia (riempimento DFS a 4 vicini), richiede almeno `closure_min_enclosed_false_cells` celle racchiuse (default `250`), ed estrae l'isolinea più lunga dell'area racchiusa con `contourpy`. La proposta richiede almeno otto punti.

La proposta include qualità (informazione media lungo il contorno), epoca e ID leader. Nel consenso metadata, le proposte vengono confrontate con la tupla `(epoch, quality, leader_id)` e la migliore viene propagata tramite i messaggi dei vicini. La transizione è locale: un drone entra in `dynamic_lloyd` quando possiede un contorno valido per almeno due frame di consenso distinti. Non c'è un coordinatore globale che ordina la transizione nello stesso istante per tutta la squadra.

## Le tre fasi operative

### 1. Mapping

All'avvio tutti i droni sono nello stato `mapping`. Con la configurazione predefinita le misure vengono acquisite ogni tre frame; nei frame intermedi i droni continuano a scambiare mappe e a muoversi, ma non aggiornano la camera. In ogni frame di misura:

1. il drone acquisisce una stima GPS;
2. la camera campiona il proprio campo visivo dal campo corrente e aggiunge rumore;
3. il rilevatore produce punti di bordo locali;
4. il footprint completo della camera aggiorna `S`, `I`, `grid` e `boundary_grid`;
5. i droni svolgono i round di consenso.

Il comportamento di moto è decentralizzato e dipende dalla mappa disponibile:

- se non è noto alcun bordo utile, il drone percorre la direzione di esplorazione assegnata casualmente all'avvio;
- se la mappa ricevuta indica un bordo ma la camera del drone non ne vede uno localmente, si muove verso il punto di bordo noto più vicino (`consensus_seek`);
- se la propria camera rileva il bordo, il controllore stima la tangente locale tramite PCA/SVD sui punti vicini, ne stabilizza il verso rispetto al moto/tangente precedente, applica una correzione normale limitata per raggiungere il bordo e aggiunge un termine di lookahead tangenziale. Questo realizza un movimento di tracking lungo il margine (`boundary_tracking`).

Le azioni sono comandi di velocità 2D limitati a `0.12` unità per passo. `Drone.action` integra il comando con `dt` e riflette la traiettoria ai confini della mappa fisica.

La fase non si conclude semplicemente perché il campo di verità è chiuso. Deve apparire un'area libera racchiusa dalla maschera locale e superare la soglia configurata; la curva viene poi mantenuta/propagata con il consenso metadata. Rumore, copertura incompleta, connettività e soglia di celle racchiuse influenzano il rilevamento.

### 2. Copertura `dynamic_lloyd` (Lloyd coverage 1D)

Quando la proposta di contorno è stabile per due frame di consenso, il drone usa i punti ordinati del contorno come una curva chiusa unidimensionale parametrizzata dalla lunghezza d'arco `s`. Il drone proietta la propria stima di posizione sul segmento più vicino e ottiene il proprio arco `boundary_s`.

Ogni drone scambia iterativamente le posizioni e le coordinate d'arco conosciute. L'engine applica un gossip multi-hop fino a un numero di passaggi pari al numero di droni; tale diffusione permette ai controller di costruire localmente l'insieme di semi (droni noti) sul contorno. Per la partizione del bordo il controller esegue un **Multi-Source Dijkstra** sulla catena di punti: i semi partono dalla coordinata d'arco proiettata e la distanza di ogni arco è la distanza euclidea fra due punti consecutivi. Ogni punto del contorno viene assegnato al seme con il cammino più corto lungo la curva: è il Voronoi 1D del bordo.

Il target Lloyd per ciascun drone è il centro, in lunghezza d'arco, della sua cella di Voronoi: i confini della cella sono a metà dell'intervallo fra i semi vicini. Il controllore sposta il drone verso quel punto con un comando proporzionale all'errore e limitato dalla velocità massima. Le celle vengono quindi riequilibrate iterativamente sul contorno; non si calcola una partizione 2D del piano.

La copertura è chiamata `dynamic_lloyd` perché il bordo può essere aggiornato mentre la macchia si muove/espande e la proiezione/partizione viene ricalcolata. Nella simulazione il controller usa il contorno condiviso corrente; il codice non stima un modello predittivo del moto del bordo né anticipa la sua dinamica.

### 3. `boundary_patrol`

Durante Lloyd coverage, ogni drone controlla l'errore tra il proprio arco e il target assegnato. Se il massimo errore locale è al più `0.25` unità d'arco per otto controlli consecutivi, dichiara `patrol_ready` e passa a `boundary_patrol`. Questo stato viene diffuso tramite i messaggi metadata; gli altri droni passano a pattuglia quando ricevono la disponibilità e hanno un contorno chiuso.

In patrol il drone avanza lungo il contorno di un piccolo incremento di lunghezza d'arco, scegliendo la velocità in funzione della distanza dal vicino precedente e successivo nel ring. Se non ha altri archi di drone noti usa la velocità di fallback `0.05`; altrimenti la velocità viene limitata all'intervallo `[0.01, 0.10]`. Il target d'arco viene convertito in un punto cartesiano e il controller comanda il moto verso di esso. La velocità fisica resta limitata a `0.12`.

La fase di patrol applica inoltre il forgetting alle mappe: ogni frame le probabilità vengono spinte leggermente verso `0.5` con fattore `0.995`, e l'informazione viene moltiplicata per lo stesso fattore. In questo modo le evidenze vecchie perdono gradualmente peso mentre i droni pattugliano e misurano il bordo aggiornato. In uno stato non mapping le misure sono eseguite ogni frame anziché ogni tre.

## Sequenza di un passo di simulazione

`SimulationEngine.step()` esegue l'ordine seguente:

1. incrementa il frame e determina gli stati locali dei droni e la cadenza di misura;
2. aggiorna posizione/forma del modello di macchia per il `dt` corrente e ricampiona il campo;
3. rinnova le stime GPS e applica forgetting ai droni già in patrol;
4. nei frame previsti acquisisce le misure camera e aggiorna le mappe locali;
5. esegue i round di consenso mappe e metadati (se ci sono droni);
6. registra la media e la diagnostica di disaccordo;
7. aggiorna la vista aggregata dello stato;
8. scambia posizioni/arco multi-hop se almeno un drone non è più in mapping;
9. calcola le azioni dei controllori e integra il moto dei droni;
10. aggiorna nuovamente le proiezioni sul contorno e lo stato aggregato.

Quindi l'ambiente viene aggiornato prima della misura di quel frame. Nei frame non di misura i droni possono continuare a muoversi e a eseguire consenso sulle mappe già disponibili.

## Struttura dei moduli

### `main.py`

È l'entry point. Configura il seed di NumPy e Python, crea ambiente/engine/droni/visualizzatore, esegue il numero richiesto di frame e salva immagini, GIF, mappe e diagnostica. I default della funzione `run_simulation` e i default della CLI sono per lo più allineati: 500 frame, seed 42, cinque droni, poligono smussato, 3 round di consenso e misura ogni tre frame. La CLI abilita il salvataggio GIF per default; `--no-save-gif` lo disabilita.

### `simulation_engine.py`

Coordina i sottosistemi ma delega le responsabilità: il drone esegue sensing e fusione; `DroneController` produce il comando; il motore determina i vicini e consegna i messaggi sincroni. Fornisce anche griglia media pesata per informazione, maschera di bordo aggregata, cronologia del disaccordo e dati di visualizzazione. `engine.control_state` è una vista riassuntiva degli stati locali, non uno stato macchina globale che impone la stessa fase a tutti.

### `drone.py`

Contiene lo stato dell'agente, i dati sensoriali recenti, mappe locali e serializzazione dei messaggi di consenso. L'aggiornamento camera-to-grid proietta ogni pixel sulla griglia di occupancy, esclude pixel fuori dominio/sorgente e aggrega i campioni per cella. Il metodo di forgetting avvicina le probabilità all'incertezza e riduce l'informazione disponibile.

### `controller.py`

`Controller` contiene strumenti per contorni, proiezione e partizione Voronoi 1D; `DroneController` li estende con le azioni locali di mapping e patrol. Il controllo non riceve l'oggetto della macchia. In mapping sceglie tra esplorazione, ricerca di un bordo condiviso e tracking locale; nelle fasi successive usa la curva di bordo e le posizioni GPS note.

### `visualization.py`

Mostra il campo simulato, le posizioni e traiettorie/azioni dei droni, le misure e i contorni mappati. Pannelli di debug e overlay Voronoi sono opzionali. La visualizzazione è separata dalla logica di simulazione; il motore fornisce uno snapshot dei dati necessari.

## Esecuzione e parametri

Dipendenze principali: Python 3, NumPy e Matplotlib. SciPy abilita gaussian smoothing, interpolazione spline e KD-tree; `contourpy` viene usato per estrarre contorni chiusi. Alcune funzionalità hanno fallback quando SciPy o `contourpy` non sono presenti, ma il rilevamento di contorno della transizione dipende da `contourpy`. Per esportare animazioni GIF è inoltre necessario il supporto Pillow di Matplotlib.

Esempio headless:

```bash
python main.py --frames 500 --seed 42 --num-drones 5
```

Esempio con visualizzazione e comunicazione a raggio limitato:

```bash
python main.py --visualize --range-based --communication-radius-cells 250
```

Esempio con macchia circolare dinamica:

```bash
python main.py --oil-shape circle --dynamic-move 0.01 --dynamic-expand 0.002
```

Parametri CLI più rilevanti:

| Parametro | Default | Significato |
| --- | ---: | --- |
| `--frames` | `500` | Numero di passi temporali. |
| `--seed` | `42` | Seed per scenario, direzioni e posizioni casuali. |
| `--num-drones` | `5` | Numero di agenti. |
| `--oil-shape` | `smoothed_polygon` | `circle` o `smoothed_polygon`. |
| `--fully-connected` | disattivo | Rete completa; altrimenti range-based. `--range-based` forza range-based anche se passato `--fully-connected`. |
| `--communication-radius-cells` | `250` | Raggio comunicazione in celle della griglia fisica. |
| `--measure-every` | `3` | Intervallo tra misure camera durante il mapping. |
| `--dt` | `1.0` | Durata simulata di un frame. |
| `--closure-min-enclosed-false-cells` | `250` | Celle libere racchiuse minime richieste per proporre un contorno. |
| `--mapping-point-radius-cells` | `1` | Parametro esposto dalla CLI per la rasterizzazione dei punti di bordo legacy; l'aggiornamento principale basato sull'intera immagine camera non lo usa. |
| `--dynamic-move` | disattivo | Modulo della velocità di deriva per passo. |
| `--dynamic-expand` | disattivo | Velocità radiale di espansione per passo. |
| `--no-save-gif` | disattivo | Non registra/esporta GIF. |

Per l'elenco completo delle opzioni: `python main.py --help`.

## Output

I risultati vengono scritti in `tmp_output/` (o nel percorso specificato per `--oil-mapping-output`):

- `final_simulation_state.png`: stato grafico finale;
- `consensus_convergence.png`: andamento del disaccordo;
- `final_occupancy_grid.png`: mappa media finale;
- un'immagine di occupancy grid per drone;
- `simulation_animation.gif` e GIF di debug, se la registrazione è attiva;
- `oil_mapping_data.npy` e `oil_mapping_metadata.json`: punti del bordo derivati dalla mappa aggregata.

L'esportazione `.npy` non salva i vertici di ground truth. Parte dalla maschera di bordo della mappa media, converte le celle in coordinate mondo e ordina i punti per angolo rispetto al loro centroide. Il metadata documenta coordinate, soglia, dimensione della griglia e dinamica configurata. Per bordi non convessi o molto irregolari, l'ordinamento angolare può non coincidere con l'ordine lungo la curva; va considerato quando il file viene usato come geometria a valle.

## Assunzioni e limiti del modello

- È una simulazione 2D a campo scalare. Non simula quota, assetto, inerzia, campo visivo prospettico, occlusione, vento sul drone né consumo energetico.
- Il campo sorgente è noto al simulatore e campionato dal sensore; non si simulano immagini RGB o un processo fisico/chimico dell'olio.
- Le letture sono rumorose, ma i collegamenti sono ideali, sincroni e senza perdite o ritardi.
- Il consenso locale usa un numero fisso di medie e non garantisce convergenza globale se il grafo è disconnesso o cambia troppo rapidamente.
- Le probabilità derivano da classificazione binaria sogliata e media di signal/information. Non sono posteriori bayesiani calibrati.
- La chiusura è inferita topologicamente dalle celle non raggiungibili dall'esterno e dipende dalla risoluzione, dal rumore e dalla soglia delle celle racchiuse.
- Lloyd è una copertura su contorno 1D ordinato; non è coverage control di area nel piano. La fase di pattuglia usa coordinata d'arco e un semplice aggiustamento di velocità tra vicini.
- La visualizzazione può mostrare il campo vero per confrontare il risultato, ma questa informazione non è input del controllo.

Questi limiti distinguono il prototipo algoritmico da un sistema di percezione robotica fisico e sono utili da esplicitare nel report, insieme alla scelta di confrontare la mappa stimata con la ground truth disponibile in simulazione.

---

# Appendice A — Lettura dettagliata del codice

Questa appendice segue il programma dal punto di ingresso fino ai singoli agenti.
Lo scopo è permettere di collegare ogni affermazione del report a una responsabilità concreta nel codice.
I nomi in codice sono mantenuti in inglese quando identificarli aiuta a ritrovare una funzione o una variabile.

## A.1 Convenzioni di coordinate e array

Il sistema usa coordinate cartesiane del mondo nella forma `(x, y)`.
`x` è la prima coordinata, orizzontale nel grafico, e `y` la seconda.
Le posizioni dei droni sono vettori NumPy di due elementi.

La griglia ambientale è creata con `meshgrid(..., indexing="ij")`.
Di conseguenza `world_field[ix, iy]` corrisponde alla coordinata `x_coords[ix]`, `y_coords[iy]`.
Molte librerie grafiche mostrano invece la prima dimensione come riga verticale; per questo alcuni `imshow` usano la trasposizione `.T`.

La griglia di occupancy conserva lo stesso ordine d'indici dell'ambiente.
Una coppia di coordinate mondo viene convertita in indici prima in X e poi in Y.
Le coordinate geometriche per una cella sono spesso poste al centro, aggiungendo mezzo passo alla coordinata del suo bordo inferiore.

La risoluzione `0.1` significa che ogni cella della mappa copre approssimativamente `0.1 × 0.1` unità mondo.
Non significa che il passo della griglia ambientale o del sensore sia anche `0.1`.
Il campo sorgente ha una discretizzazione più fine, di circa `0.0281` unità per asse.

In discussioni e report è utile distinguere pixel, celle, punti di contorno e posizioni.
Un pixel è una misura nella finestra locale della camera.
Una cella è un elemento della mappa globale del drone.
Un punto di contorno è una coordinata `(x, y)` estratta da un'immagine o da un'isolinea.

Le unità sono unità arbitrarie della simulazione.
Non è definita una conversione automatica in metri.
Le velocità sono espresse in unità mondo per passo temporale, salvo moltiplicazione per `dt`.

## A.2 `main.py`: ingresso e configurazione

`main.py` importa il modello d'ambiente, il motore e il visualizzatore.
Le funzioni con prefisso `_` sono helper interni di costruzione o salvataggio.
Il blocco finale `if __name__ == "__main__"` esegue la funzione `main()` solo quando il file è lanciato direttamente.

Le costanti `OUTPUT_DIR`, `OIL_MAPPING_DATA_PATH` e `OIL_MAPPING_METADATA_FILENAME` centralizzano i percorsi di default.
`OUTPUT_DIR` è `./tmp_output`, relativo alla directory corrente da cui si avvia Python.
Il file dati predefinito è `tmp_output/oil_mapping_data.npy`.

Prima di importare pyplot, il programma configura `MPLCONFIGDIR` solo se non esiste già.
Il percorso predefinito della cache Matplotlib è `/tmp/matplotlib`.
La scelta del backend grafico viene rinviata a `_configure_matplotlib`.

### A.2.1 Seed e ripetibilità

`_set_random_seed(seed)` inizializza sia `numpy.random` sia il generatore casuale standard di Python.
Questo rende ripetibili estrazioni che usano questi due generatori globali.
La funzione stampa il valore del seed usato.

Il modello poligonale usa inoltre un `numpy.random.default_rng(seed)` locale.
La direzione di drift usa un altro generatore NumPy inizializzato con lo stesso seed.
Il seed determina quindi più estrazioni distinte ma ripetibili nei rispettivi generatori.

Il seed non elimina il rumore: fa sì che la sequenza di rumore sia replicabile.
Due esecuzioni con gli stessi parametri e la stessa versione delle dipendenze dovrebbero generare gli stessi campioni casuali.
Differenze di versione, libreria o ordine delle chiamate casuali possono cambiare i risultati.

Le posizioni iniziali vengono estratte in `_add_drones` usando il generatore NumPy globale.
Il rumore GPS e quello camera usano anch'essi NumPy globale.
Le estrazioni successive dipendono dall'ordine con cui il programma campiona ciascun sensore.

### A.2.2 Scelta del backend Matplotlib

`_configure_matplotlib(visualize)` seleziona `TkAgg` quando è richiesta la visualizzazione interattiva.
In modalità batch seleziona `Agg`, che genera grafici senza aprire una finestra.
La funzione importa `pyplot` solo dopo aver selezionato il backend.

Se Tk non è disponibile, il programma stampa un messaggio e passa ad `Agg`.
Se il backend non interattivo non può essere configurato, l'eccezione viene propagata.
Questo helper tratta il rendering come una configurazione di esecuzione, non come parte della logica dei droni.

### A.2.3 Costruzione dell'ambiente

`_build_environment` fissa il dominio fisico a `[-7, 7]` su entrambi gli assi.
Crea una `SimulationMap` di 500 campioni per asse.
Poi sceglie il modello circolare o quello poligonale in base a `oil_shape`.

Per il cerchio, i parametri passati includono centro `(0,0)` e raggio `2.0`.
Per il poligono, i parametri di vertici, raggio di riferimento, smoothness e centro arrivano dalla CLI.
La macchia riceve anche drift e velocità di espansione.

Il ramo `if oil_shape == "circle"` crea il cerchio.
Ogni altro valore raggiungerebbe il ramo del poligono se la funzione fosse chiamata direttamente senza validazione.
La CLI protegge il normale percorso d'esecuzione limitando le scelte a due valori.

### A.2.4 Drift e espansione

`_dynamic_drift_velocity` genera un angolo uniforme tra zero e `2π`.
Converte il modulo `speed` nel vettore cartesiano `[speed cos(angle), speed sin(angle)]`.
Un modulo negativo genera `ValueError`.

La direzione è estratta una sola volta durante l'inizializzazione.
La velocità non cambia direzione nei frame successivi.
Il modello integra il vettore moltiplicandolo per `dt`.

`_dynamic_expansion_speed` converte `None` in zero.
Un valore negativo viene rifiutato.
Il valore restituito è una velocità radiale non negativa.

### A.2.5 Raggio di comunicazione

`_communication_radius` moltiplica il conteggio di celle per la media dei passi ambientali `dx` e `dy`.
Con `grid_size=500` e ampiezza 14, i due passi coincidono e valgono circa `0.028056`.
Un valore CLI di 250 celle corrisponde a circa 7.014 unità.

Il motore ripete il calcolo usando `sim_map.dx` e `sim_map.dy`.
La quantità viene poi usata come soglia di distanza euclidea GPS.
Non si confronta il numero di celle direttamente con una distanza in unità mondo.

### A.2.6 Costruzione del motore

`_build_engine` stabilisce la griglia di occupancy indipendente dalla mappa ambientale.
Il limite va da -10 a 10 per entrambi gli assi.
Con passo `0.1` il risultato è `200 × 200` celle.

Il sensore ha dimensione 120 pixel.
La soglia di occupancy è fissata a 0.6.
Il programma passa `temporal_alpha=0.05`, ma il costruttore attuale del motore imposta `self.temporal_alpha = None` e il percorso principale non usa questo parametro.

Il numero dei round di consenso mappe è fissato a 3 in questo helper.
Il motore stampa diagnostica verbosa per impostazione predefinita.
Le soglie di chiusura e il raggio di rasterizzazione arrivano dai parametri della simulazione.

### A.2.7 Aggiunta dei droni

`_add_drones` converte il numero richiesto in intero e lo limita inferiormente a zero.
Per ciascun drone assegna un ID deterministico `D0`, `D1`, `D2` e così via.
La posizione iniziale è campionata uniformemente nel dominio fisico.

L'offset iniziale della direzione è casuale.
La distanza angolare tra gli heading è `2π/N` quando ci sono `N` droni.
Con zero droni non viene eseguito il ciclo e non si verifica una divisione per zero.

Ogni sensore riceve rumore standard `0.03`.
Il controller viene costruito dal motore per ogni drone.
La squadra non condivide un controller Python: ciascun agente chiama il proprio `DroneController` con il proprio stato.

### A.2.8 Visualizzatore e intestazione del run

`_build_visualizer` passa le opzioni relative a comunicazione, punti edge, overlay Voronoi e finestra di debug.
Con rete fully connected, il raggio non viene mostrato.
In range-based, il raggio viene passato al visualizzatore.

`_print_run_header` riassume forma, parametri poligonali, dinamica, frequenza sensoriale e rete.
La velocità di drift viene ristampata come vettore e come norma.
L'intestazione serve a conservare nel log i parametri essenziali dell'esperimento.

### A.2.9 Ciclo principale

`_run_frames` invoca `engine.step()` una volta per frame.
Renderizza se la modalità visuale è attiva oppure se si sta registrando un'animazione.
Ogni 50 frame stampa il disaccordo corrente.

`KeyboardInterrupt` interrompe il loop e viene gestito stampando un messaggio.
Il codice prosegue poi con finalizzazione e salvataggio dei risultati.
Non viene re-innescato automaticamente il run dopo l'interruzione.

`run_simulation` coordina le fasi di setup, esecuzione e output.
La funzione configura backend e seed prima di creare l'ambiente.
Poi costruisce motore, droni e visualizzatore nell'ordine necessario.

Se `save_gif` è vero, avvia la registrazione prima del loop.
Dopo il loop finalizza le cronologie e salva le immagini.
Infine esporta il mapping e stampa le statistiche di consenso.

### A.2.10 Parser CLI

`_build_parser` definisce le opzioni command-line e i rispettivi default.
Le forme dell'olio sono limitate a `circle` e `smoothed_polygon`.
Le opzioni dinamiche accettano underscore e trattino per compatibilità.

`--fully-connected` attiva il grafo completo.
`--range-based` può prevalere nella funzione `main`, che calcola `args.fully_connected and not args.range_based`.
Se entrambe sono presenti, il risultato è range-based.

`--debug-window` usa `BooleanOptionalAction` e per default è attivo.
Per disattivarlo si può usare la forma negativa generata da argparse, `--no-debug-window`.
La finestra di debug è visibile solo con `--visualize` e quando è creata.

`--no-save-gif` cambia il valore booleano `save_gif` da vero a falso.
Gli altri parametri della GIF controllano nome, frame rate e stride.
Lo stride maggiore di uno riduce quanti frame vengono catturati.

## A.3 `environment.py`: mondo, campo e macchia

### A.3.1 Componente connessa maggiore

`_largest_connected_component` riceve una maschera booleana.
Scorre ogni cella e avvia una DFS su ogni componente non visitata.
La connettività usata per questa operazione è 4-connessa: sopra, sotto, sinistra e destra.

La funzione salva i pixel del componente più grande incontrato.
Alla fine restituisce una maschera in cui solo quel componente è vero.
Le componenti minori vengono eliminate dal campo interno poligonale.

Questa routine non è il medesimo riempimento usato per verificare la chiusura della mappa del drone.
Nel campo dell'ambiente seleziona la macchia interna più estesa.
Nel controller, invece, il riempimento dall'esterno identifica regioni libere racchiuse.

### A.3.2 `SimulationMap`

Il costruttore salva limiti e dimensione come attributi.
Controlla che la griglia abbia almeno due punti.
Controlla anche che i limiti inferiori siano minori dei superiori.

`x_coords` e `y_coords` sono array regolari prodotti da `linspace`.
Gli estremi sono inclusi.
Il passo quindi è ampiezza divisa per `grid_size - 1`, non per `grid_size`.

`X` e `Y` sono matrici coordinate create con `indexing="ij"`.
La proprietà `shape` restituisce la forma della matrice ambientale.
Le proprietà `dx` e `dy` restituiscono il passo del reticolo.

`is_inside` controlla se una coppia cade nei limiti inclusivi.
`clip_position` applica `np.clip` separatamente per asse.
Nel movimento fisico del drone non viene usato questo clipping diretto: il drone riflette sulla parete.

### A.3.3 Interfaccia `Field` e `OilSpill`

`Field` è un'interfaccia minimale che espone `field(X, Y)`.
La sua implementazione di base solleva `NotImplementedError`.
È una classe astratta in senso informale, non usa il modulo `abc`.

`OilSpill` definisce un aggiornamento inerte, che ignora `dt`.
Le implementazioni statiche possono ereditare questo comportamento senza fare nulla.
I modelli dinamici sovrascrivono `update`.

`get_field(X, Y)` ricalcola il campo sulle coordinate richieste.
Senza coordinate restituisce una copia della cache `_field` se esiste.
Se manca una cache e non sono fornite coordinate, solleva un errore.

`get_mask(threshold)` converte il campo in una maschera con confronto `>= threshold`.
Il default per questo helper è 0.5.
La soglia operativa di occupancy del motore è invece 0.6.

### A.3.4 `CircleOilSpill`

Il modello controlla che il centro abbia coordinate valide e la velocità sia un vettore di due elementi.
Raggio e sigma devono essere strettamente positivi.
Una velocità di espansione negativa viene rifiutata.

Per ogni punto calcola la distanza euclidea dal centro.
All'interno del raggio assegna un valore esattamente uguale a 1.
Fuori dal raggio usa `exp(-((d-r)^2)/(2 sigma^2))`.

Il valore al confine è continuo: quando `d=r`, il termine esponenziale vale 1.
Lontano dal bordo, la concentrazione tende a zero.
La formula non è una distribuzione spaziale completa in senso statistico: è il profilo di intensità scelto per la simulazione.

`update(dt)` calcola traslazione e incremento radiale separatamente.
Se entrambi sono numericamente nulli, esce senza modificare lo stato.
Se la macchia si muove, aggiorna `x0` e `y0`.

Se si espande, somma `expansion_speed * dt` al raggio.
L'aggiornamento invalida la cache `_field`.
Il motore ricampiona la mappa dopo aver aggiornato il modello.

### A.3.5 `SmoothedPolygonOilSpill`: parametri e vertici

Il costruttore copia le coordinate del reticolo ambientale in array floating point.
Verifica che le due matrici coordinate abbiano la stessa forma.
Inizializza un generatore casuale dedicato con `seed`.

Il numero di vertici è almeno tre.
Il raggio `r0` è limitato inferiormente a `1e-6`.
La smoothness è limitata all'intervallo `[0,1]`.

Se il centro non viene specificato, usa il centro del bounding box della griglia.
I vertici si ottengono ordinando angoli casuali.
Ogni vertice è poi collocato a un raggio perturbato dal centro.

La parte casuale radiale combina rumore normale e due armoniche sinusoidali.
L'ampiezza di irregolarità è `0.45 * (1 - smoothness)`.
Una smoothness più alta riduce la perturbazione casuale radiale.

I raggi vengono limitati tra `0.35*r0` e `1.85*r0`.
Questo evita vertici troppo vicini al centro o eccessivamente lontani.
Il valore `r0` è una scala nominale, non il raggio esatto di ogni punto.

### A.3.6 Curva chiusa e campo poligonale

`_build_smooth_boundary` genera almeno 24 campioni.
Se SciPy fornisce `splprep` e `splev`, prova una spline periodica.
La smoothing parameter cresce con smoothness e numero di vertici.

Se la spline fallisce, o SciPy non è disponibile, usa `_catmull_rom_closed`.
La curva Catmull–Rom tratta i punti in modo ciclico aggiungendo predecessore e successori ai bordi.
Campiona ogni segmento in modo uniforme nel suo parametro locale.

Il path Matplotlib viene creato con `closed=True`.
La proprietà `radius` è la massima distanza dei campioni della curva dal centro.
Questa misura viene usata anche per riscalare il contorno durante l'espansione.

`_evaluate_field` appiattisce le coordinate in una lista di punti.
Usa `Path.contains_points` per stabilire quali campioni sono interni.
Poi conserva la componente interna connessa più grande.

Se `cKDTree` non è disponibile, restituisce una maschera binaria 0/1.
Con SciPy, calcola la distanza di ogni campione dal campione di bordo più vicino.
Segna distanza negativa dentro il poligono e positiva fuori.

La larghezza della transizione è proporzionale a `r0` e a smoothness.
`continuous=False` riduce la larghezza moltiplicandola per 0.8.
Il campo è una logistica `1 / (1 + exp(signed_distance / boundary_width))`.

All'interno la distanza firmata negativa produce valori superiori a 0.5.
All'esterno produce valori inferiori a 0.5.
La soglia 0.6 sposta il bordo classificato leggermente all'interno rispetto all'isolinea 0.5.

### A.3.7 Aggiornamento dinamico del poligono

La deriva aggiunge lo stesso vettore a centro, vertici e campioni della curva.
L'espansione calcola un nuovo raggio massimo e un fattore di scala.
I punti vengono scalati radialmente rispetto al centro corrente.

Il raggio nominale `r0` viene aggiornato con lo stesso fattore.
La classe ricrea il path dopo una modifica geometrica.
Poi ricampiona il campo sulla griglia ambientale memorizzata.

La geometria non viene rigenerata a ogni passo.
Si trasforma la curva corrente, preservando la forma relativa.
Drift ed espansione possono essere attivi simultaneamente.

## A.4 `sensors.py`: dati e trasformazioni sensoriali

### A.4.1 Classe base `Sensor`

Il costruttore converte `noise_std` in float.
`add_noise` converte il valore in array NumPy.
Un rumore nullo o negativo disabilita l'aggiunta casuale.

Con rumore positivo genera campioni gaussiani indipendenti della stessa forma dell'input.
La media è zero e la deviazione standard è `noise_std`.
La funzione non limita i dati: la camera li limita successivamente, il GPS no.

### A.4.2 `GPSSensor`

Il GPS richiede un vettore posizione di forma `(2,)`.
Una forma diversa causa `ValueError`.
La misura è la posizione vera più il rumore additivo.

Gli errori sulle coordinate x e y vengono campionati indipendentemente.
La misura può teoricamente ricadere oltre il limite fisico.
Il controllo può quindi ricevere una stima leggermente fuori dai confini anche quando lo stato fisico è al bordo.

### A.4.3 `CameraMeasurement`

La classe è un semplice contenitore di risultati.
`image` è convertita in array di float.
`edge_points` è convertito in array di float.

`oil_fraction` viene convertita in float Python.
La classe non verifica le dimensioni dei dati in modo autonomo.
La forma e la coerenza sono assicurate dal sensore che la costruisce.

### A.4.4 Ingressi e validazione della camera

`CameraSensor.__init__` memorizza dimensione, rumore, blur, soglie e soglia di occupancy.
La dimensione deve essere positiva.
Le soglie Canny sono memorizzate per compatibilità ma non sono consultate da `detect_edges`.

`sense` converte campo e vettori di coordinate in float.
`_validate_inputs` richiede un campo 2D.
La forma deve coincidere esattamente con `(len(x_coords), len(y_coords))`.

Sono richiesti almeno due campioni per ciascun asse.
La soglia usata per `oil_fraction` può essere fornita per chiamata.
La chiamata del drone nel percorso standard omette tale parametro e quindi usa quella memorizzata nel sensore.

### A.4.5 Estrazione della finestra locale

La posizione viene convertita in indici tramite `(coordinate - primo_campione) / passo` e arrotondamento.
La finestra è centrata intorno a quegli indici.
Il codice seleziona una porzione dell'array sorgente usando limiti interi.

Ai margini, la porzione può essere più piccola della dimensione richiesta.
La differenza viene riempita con zeri.
L'output viene tagliato o riempito ulteriormente per forzare la forma esatta `(size, size)`.

La camera non ruota la finestra in funzione dell'heading del drone.
Il footprint è sempre allineato agli assi mondo.
La geometria sensoriale è quindi una finestra quadrata a terra, non una proiezione prospettica.

Per default il campo di vista ha lato approssimativo pari a 120 campioni per 0.0281 unità.
Questo corrisponde a circa 3.37 unità per lato.
L'esatto supporto utile dipende dall'arrotondamento e dal fatto che il campione centrale sia incluso.

### A.4.6 Rumore, clipping e frazione olio

Il rumore viene aggiunto dopo aver estratto il campo locale.
I valori della matrice rumorosa sono limitati a `[0,1]`.
Questo impedisce concentrazioni negative o superiori al massimo definito dal modello.

Il sensore opzionalmente applica una sfocatura gaussiana.
Questa opzione è disattivata nella configurazione usata da `main.py`.
Se viene richiesta, `scipy.ndimage.gaussian_filter` è necessario.

`oil_fraction` è la media booleana di `noisy_matrix >= threshold`.
È una statistica descrittiva dell'intera immagine locale.
Il controller non la usa per stimare la traiettoria principale di tracking.

### A.4.7 Coordinate locali e coordinate mondo

`extract_edge_points` restituisce coppie `(row, column)`.
La prima coordinata della coppia è convertita lungo l'asse X.
La seconda è convertita lungo Y.

Il sensore individua il campione del reticolo sorgente più vicino alla posizione GPS.
Quella coordinata diventa il centro mondo della finestra.
La distanza dal centro immagine viene moltiplicata per `dx` o `dy`.

La funzione restituisce punti `(world_x, world_y)`.
La conversione usa la stima di posizione passata in `position_estimate`.
Nel drone, questa è la lettura GPS appena aggiornata.

L'uso di un centro immagine geometrico `(dimensione-1)/2` gestisce anche dimensioni pari.
Con 120 pixel il centro è tra gli indici 59 e 60.
Il calcolo assegna quindi coordinate simmetriche rispetto al centro frazionario.

## A.5 `edge_detection.py`: estrazione dell'isolinea

### A.5.1 Controlli iniziali

`detect_edges` converte l'immagine in float.
Se l'immagine non è 2D o è vuota, restituisce una maschera booleana vuota della stessa forma.
Questa guardia evita di eseguire filtri su input senza immagine valida.

Calcola minimo e massimo dell'immagine.
Se il contrasto `max-min` è inferiore a 0.15, non dichiara alcun bordo.
Questa euristica respinge immagini quasi uniformi di acqua o olio.

Se `threshold1` compare nei kwargs e cade strettamente tra 0 e 1, sostituisce la soglia primaria.
I parametri `edge_threshold1` e `edge_threshold2` del sensore non vengono passati come tali.
La soglia pratica resta quindi quella di occupancy, salvo invocazione manuale con kwargs.

### A.5.2 Smoothing e maschera

Quando SciPy è disponibile e `sigma > 0`, viene applicato un filtro gaussiano.
Altrimenti l'immagine originale viene usata direttamente.
La maschera occupata è `smoothed >= threshold`.

Il codice non calcola un gradiente Canny o una soglia di isteresi.
Estrae la frontiera morfologica della maschera.
Questo metodo richiede un passaggio di concentrazione dalla parte occupata a quella libera.

### A.5.3 Erosione e pixels di edge

Con SciPy, `binary_erosion(mask, border_value=True)` contrae la regione.
L'edge risultante è XOR tra regione originale ed erosa.
Il trattamento `border_value=True` evita di trattare automaticamente il bordo dell'array come acqua.

Senza SciPy, viene costruito un padding di bordo e una erosione manuale a croce.
Il fallback considera i quattro vicini ortogonali per l'erosione.
La maschera di contorno è la regione originale meno quella erosa.

Una regione che occupa tutta l'immagine e ha contrasto sufficiente può non generare un contorno interno apprezzabile.
Il rilevatore è pensato per finestre che contengono una transizione spaziale.
L'immagine uniforme è già respinta dal controllo sul contrasto.

### A.5.4 `extract_edge_points`

La funzione converte una maschera 2D in coordinate di indice.
`np.nonzero` restituisce vettori di righe e colonne.
La funzione li impila in una matrice `N × 2` di float.

Se non ci sono pixel, restituisce una matrice vuota con forma `(0,2)`.
La convenzione di coordinate resta riga-colonna fino alla conversione del sensore.
Il controller non riceve indici d'immagine: riceve coordinate mondo.

## A.6 `drone.py`: agente e stato locale

### A.6.1 Costruttore e stato persistente

Il costruttore salva ID e posizione fisica.
Salva i limiti e la forma della mappa locale condivisa.
Inizializza la probabilità a 0.5 in ogni cella.

Le griglie di signal, informazione e bordo iniziano a zero.
Una griglia iniziale piena di 0.5 significa incertezza, non acqua.
L'informazione iniziale zero distingue esplicitamente le celle mai osservate.

Il drone crea un sensore GPS e una camera.
Acquisisce immediatamente una lettura GPS iniziale.
Quella lettura iniziale è usata anche per inizializzare `known_positions`.

### A.6.2 Variabili sensoriali transitorie

`edge_detected` indica se l'ultima misura ha prodotto almeno un punto.
`last_edge_points` conserva tutti i punti di bordo dell'ultima misura valida.
`last_edge_point` conserva il punto più vicino alla stima GPS.

`last_camera_image` conserva l'intero footprint rumoroso.
`last_camera_spacing` memorizza il passo della sorgente per asse.
`last_camera_center_position` memorizza il campione sorgente scelto come centro.

`last_camera_source_bounds` memorizza il bounding box delle coordinate sorgente.
Il rasterizzatore usa questi limiti per escludere pixel che sono padding.
I valori di padding zero non devono essere contabilizzati come osservazioni d'acqua fuori dal dominio reale.

`last_oil_fraction` conserva la frazione locale occupata.
`last_mapped_boundary_points` conserva la conversione in mondo dei pixel della maschera di bordo derivata dalla griglia.
`last_nls_points` è mantenuta come informazione accessoria/compatibilità e riceve gli edge points correnti.

### A.6.3 Variabili di controllo e fase

`last_control_mode` registra l'azione selezionata, ad esempio `explore` o `consensus_seek`.
`last_control_vector` conserva il comando effettivamente applicato dopo il clipping.
`last_mapping_tangent` conserva la tangente per mantenere coerenza di orientamento tra frame.

`control_state` parte da `mapping`.
Può avanzare a `dynamic_lloyd` o `boundary_patrol` in base ai metadati ricevuti.
Lo stato è locale al singolo agente.

`known_positions` associa ID di droni a stime di posizione.
`known_boundary_arcs` associa ID a coordinate di lunghezza d'arco.
Entrambe le strutture alimentano il controllo distribuito lungo il contorno.

### A.6.4 Stato del contorno

`boundary_index` è l'indice discreto del punto di curva più vicino.
`boundary_s` è la coordinata continua approssimata di lunghezza d'arco.
`pending_boundary_s` e `pending_boundary_point` sono campi temporanei per eventuali proiezioni differite.

`contour_points` conserva il contorno condiviso accettato.
`contour_epoch`, `contour_quality` e `contour_leader_id` definiscono la priorità della proposta.
`closure_detected` indica che esiste un contorno plausibile con almeno tre punti.

`consecutive_closure_frames` conta i frame di consenso distinti con una proposta valida.
La transizione richiede due frame consecutivi.
Più round nello stesso frame non incrementano ripetutamente il contatore, perché l'ID frame viene registrato.

`patrol_ready` segnala che il drone è arrivato abbastanza vicino al proprio target Lloyd.
`_patrol_ready_count` memorizza la sequenza di controlli soddisfacenti.
Otto controlli consecutivi richiesti dal codice fanno scattare lo stato di patrol.

### A.6.5 Metodi di delega del drone

`compute_action` inoltra gli argomenti al controller locale.
Il drone non incorpora direttamente il calcolo della politica di controllo.
Questo separa dati dell'agente e algoritmo decisionale.

`set_known_boundary` copia punti e stato di ordinamento nel drone e nel controller.
`project_to_boundary` chiede al controller di proiettare fisicamente il drone.
`update_boundary_projection` aggiorna invece solo le coordinate d'arco e indice, senza spostarlo.

La distinzione tra proiezione geometrica e proiezione dello stato è rilevante.
La normale fase di controllo chiama `update_boundary_projection` per aggiornare la misura geometrica.
Lo snap fisico viene usato soltanto se il controller invoca esplicitamente `project_to_boundary`.

### A.6.6 `sense`

Il metodo aggiorna prima la posizione GPS.
Passa campo e coordinate ambientali alla camera.
Passa anche la stima GPS come centro del footprint.

Se la camera non restituisce una misura, cancella i dati di edge e le informazioni del footprint.
Se l'immagine esiste, conserva una copia e calcola il passo della sorgente.
Il centro camera viene ricostruito arrotondando la stima GPS alla coordinata della griglia ambientale più vicina.

Se non sono rilevati edge points, i dati transitori di bordo sono cancellati.
La frazione d'olio viene mantenuta come diagnostica dell'ultima immagine.
Quando ci sono punti, il drone salva punti, conteggio e frazione.

L'anchor locale è la media aritmetica dei punti dell'immagine.
Viene filtrato temporalmente come `0.8 * anchor_precedente + 0.2 * anchor_corrente`.
La media non viene usata come contorno completo: è una traccia sintetica dell'ancora osservata.

Il punto più vicino è scelto confrontando distanze euclidee tra edge points e GPS stimato.
Il campo `last_edge_point` è quindi utile per diagnostica o controlli semplici.
Il controllo di mapping più avanzato usa il point cloud e la mappa del bordo.

### A.6.7 Aggiornamento mappa da immagine

`update_grid` preferisce il footprint camera completo se `last_camera_image` esiste.
La modalità point-only è un fallback per misure legacy.
I parametri `alpha`, `point_radius_cells`, `x_coords` e `y_coords` vengono ignorati nel ramo camera.

`_update_grid_from_camera` costruisce indici riga/colonna per tutti i pixel.
Li converte in coordinate mondo rispetto al centro registrato.
Marca come validi solo pixel finiti entro i limiti sorgente e della griglia di occupancy.

`floor((world - min) / resolution)` assegna ogni pixel alla cella corrispondente.
Le coordinate che cadono esattamente sul limite superiore possono uscire dalla griglia.
La maschera valida previene accessi fuori dai limiti.

`np.bincount` conta i pixel per cella.
Un secondo conteggio pesato somma gli indicatori di olio.
La media dei pixel per cella dà la misura di occupancy tra 0 e 1.

Le celle viste hanno informazione impostata a 1.
La misura locale sostituisce il signal già presente per le celle osservate.
Le celle fuori dal footprint mantengono le proprie evidenze precedenti.

### A.6.8 Aggiornamento point-only

`_update_grid_from_points` converte ogni punto mondo in una cella.
Il punto diventa un'evidenza occupata con signal e information entrambi pari a 1.
I punti fuori dalla griglia vengono ignorati.

Se non ci sono punti validi, la mappa non cambia.
Il raggio di rasterizzazione non è implementato in questo helper.
La CLI conserva comunque il parametro per compatibilità con l'interfaccia esistente.

### A.6.9 Probabilità e bordo

`probability_from_signal` inizializza tutto a `unknown_probability`, per default 0.5.
Dove `information > 1e-12`, calcola signal diviso information.
Infine limita i valori nell'intervallo `[0,1]`.

`boundary_mask_from_probability` distingue celle occupate note e libere note.
Un bordo è una cella occupata nota adiacente a una cella libera nota.
La ricerca dei vicini usa tutte le otto direzioni, diagonali incluse.

L'array libero viene paddingato con `False` ai bordi della griglia.
Fuori dal campo mappato non viene assunto che esista acqua nota.
Questa scelta evita che il confine artificiale della mappa diventi automaticamente contorno d'olio.

`_boundary_points_from_mask` trasforma gli indici in centri di cella.
La risoluzione è calcolata dalla media delle risoluzioni per asse.
Con la griglia quadrata standard coincide con `0.1`.

`_refresh_probability_grid` ricalcola `grid`, bordo e punti di bordo derivati.
È richiamata dopo una misura, consensus e forgetting.
La griglia di bordo è una maschera derivata e non contiene probabilità indipendenti.

### A.6.10 Forgetting e consenso locale

`apply_forgetting(alpha)` combina la probabilità corrente con 0.5.
Con `alpha=0.995`, ogni cella si sposta di mezzo punto percentuale della distanza verso l'incertezza.
L'informazione è moltiplicata per lo stesso fattore.

Il signal viene ricostruito come probabilità per informazione.
Successivamente il metodo ricalcola probability e bordo.
Se alpha è almeno 1, il metodo non modifica la mappa.

`consensus_step` prende una fotografia della mappa del drone e i messaggi ricevuti.
Calcola un bordo locale provvisorio prima di fonderli.
Quel bordo e la relativa informazione vengono usati dal rilevatore metadata, così la candidatura si basa sull'evidenza locale precedente alla media del round.

Signal e information sono mediati separatamente tra agente e vicini.
La media aggiornata diventa la nuova memoria locale del drone.
Il controller aggiorna poi metadati di contorno e fase.

Al termine i riferimenti alle snapshot locali vengono cancellati.
Questo evita che il round successivo usi accidentalmente una snapshot obsoleta.
La mappa di probabilità viene restituita per comodità del chiamante.

## A.7 Intermezzo: qualità dell'evidenza e significato della media

Una cella con informazione zero rappresenta assenza di osservazioni.
Una cella con probabilità 0.5 ma informazione positiva potrebbe invece essere una misura che cade proprio nel punto neutro.
L'informazione è necessaria per distinguere questi due casi.

Il motore calcola la mappa aggregata mediando signal e information dei droni.
Poi divide i due campi mediati dove l'informazione media è positiva.
Questa costruzione equivale alla media delle probabilità dei contributori che hanno evidenza, se ciascun contributo è rappresentato in modo coerente.

Non è una media pesata per il numero effettivo di osservazioni indipendenti.
Ogni drone ha un peso simile nell'operazione di media del round.
I valori `information_grid` sono medie iterative e decadono durante patrol.

Una misura ripetuta dallo stesso drone non accumula automaticamente informazione locale oltre uno.
Il campo locale osservato viene rimesso a 1 per quella cella.
Il consenso riduce o diffonde i valori secondo la sua dinamica di medie.

Perciò la parola “probabilità” descrive una probabilità di occupancy rappresentata numericamente.
Non deve essere interpretata come posteriori bayesiana calibrata.
Non sono presenti prior e likelihood espliciti né un modello di indipendenza delle misure.

---
## A.8 `simulation_engine.py`: orchestrazione e protocollo temporale

### A.8.1 Costruttore del motore

Il motore riceve gli oggetti `SimulationMap` e `OilSpill` già creati.
Conserva i limiti della griglia di occupancy e la risoluzione.
Converte i parametri numerici in tipi Python espliciti.

La dimensione della griglia è calcolata arrotondando ampiezza diviso risoluzione.
Nel caso predefinito, 20 unità divise per 0.1 danno 200 celle per asse.
La forma interna è la tupla `(Nx, Ny)`.

La prima chiamata a `_get_world_field` valuta il campo sulla griglia ambientale.
Il campo è mantenuto come array NumPy `world_field`.
Ogni aggiornamento dinamico dell'ambiente sostituisce questo array.

Il motore conserva sia raggio in celle sia raggio convertito in coordinate mondo.
Il secondo è la distanza operativa usata per formare il grafo di comunicazione.
`fully_connected` permette al motore di saltare i controlli di distanza.

Le cronologie sono inizializzate vuote.
`error_history` contiene un valore per frame.
`mean_grid_history` contiene una copia della mappa aggregata per frame.

`measurement_consensus_history` raggruppa gli errori per drone durante una misura e i successivi round.
`_current_measurement_trace` resta attiva finché non inizia un nuovo ciclo di misurazione.
`finalize_histories` chiude e aggiunge l'ultima traccia alla fine del run.

### A.8.2 Creazione degli agenti

`add_drone` costruisce un `Drone` con forma e limiti condivisi.
Ogni drone riceve un controller istanziato con la stessa geometria e gli stessi parametri di comunicazione.
Le mappe restano oggetti distinti per agente.

La soglia di occupancy è passata sia alla camera sia al controller.
Questo mantiene coerente la classificazione tra acquisizione e azione.
Il drone riceve anche la soglia minima di area racchiusa per la sua rilevazione di chiusura.

Il motore registra l'oggetto nella lista `drones`.
Non crea un processo operativo o un thread separato per ogni drone.
La simulazione multiagente è eseguita sequenzialmente da un unico processo Python.

### A.8.3 Aggiornamento della macchia

`_update_environment` invoca il metodo `update(dt)` dell'oggetto macchia.
Il motore non ha bisogno di distinguere un cerchio da un poligono.
Dopo l'aggiornamento valuta nuovamente il campo sulla griglia fisica.

Quando il modello è statico, `update` non cambia lo stato.
Il ricampionamento continua comunque a essere richiesto a ogni passo.
Il modello poligonale mantiene una cache aggiornata internamente dopo la dinamica.

### A.8.4 Sensing in batch

`_perform_measurement` itera sui droni registrati.
Per ogni drone chiama `sense` con il campo e le coordinate correnti.
Passa i punti edge al metodo `update_grid`.

Il percorso di produzione mappa l'intera immagine camera.
I punti edge sono mantenuti per tracking e diagnostica.
Non sono l'unico dato usato per classificare l'occupancy.

La misura viene effettuata in modo sequenziale nel loop Python.
Tutti i droni campionano lo stesso `world_field` del frame.
Non viene simulato un offset temporale fra i sensori dei diversi agenti.

### A.8.5 Vicinato nel consenso

`_get_neighbors` restituisce gli altri oggetti drone.
Il drone stesso è escluso tramite identità dell'oggetto.
In fully connected, non viene effettuato un calcolo di distanza.

Nel grafo range-based, il motore confronta la distanza euclidea fra stime GPS.
La condizione usa `<= communication_radius`.
La comunicazione non dipende dalla forma dell'olio o dalla visibilità ottica.

Il raggio definisce un grafo geometricamente basato sulla posizione.
Non sono presenti ostacoli di comunicazione o schermature.
La prossimità GPS è assunta sufficiente per scambiare messaggi.

### A.8.6 Round sincrono di consenso

`_exchange_consensus_messages` crea prima un messaggio per ogni drone.
Ogni messaggio contiene copie delle griglie e dei metadati.
Le copie impediscono che l'aggiornamento del primo agente modifichi i dati letti dal secondo nello stesso round.

Il motore costruisce la lista di messaggi consegnati a ciascun destinatario.
Poi invoca `consensus_step` per ogni drone.
Questo realizza un aggiornamento sincrono, non un aggiornamento in-place asincrono.

Un round non è un broadcast garantito a tutti i droni.
Ogni destinatario riceve soltanto i messaggi dei vicini definiti in quel round.
La rete fully-connected è il caso speciale in cui tutti i vicini sono accessibili.

`_perform_consensus` ripete il round un numero configurato di volte.
Il metodo è disponibile come helper anche se `step` gestisce il proprio ciclo con diagnostica intermedia.
Nel normale passo ogni iterazione viene registrata per poter osservare la convergenza.

### A.8.7 Mappa media aggregata

`compute_mean_information_grid` calcola la media aritmetica delle griglie di informazione.
Senza droni restituisce zeri.
La media è per cella, non una media degli scalar score globali.

`compute_mean_grid` calcola la media dei signal.
Divide per la media di informazione tramite `probability_from_signal`.
Senza droni restituisce una mappa piena di 0.5.

Il nome “mean grid” non significa media diretta delle probabilità con ignoranza codificata come zero.
Le celle non informative restano neutre.
La ricostruzione usa signal e informazione per non trattare celle sconosciute come occupate o libere.

`compute_mean_boundary_grid` applica la stessa regola di bordo della mappa locale alla mappa aggregata.
Questa maschera è usata per il rendering e l'esportazione.
Non coincide necessariamente con il contorno metadata scelto per il controllo Lloyd.

### A.8.8 Disaccordo e cronologie

`compute_disagreement_error` calcola una norma L2 per ogni drone.
La distanza è tra `drone.grid` e la mappa media aggregata.
Il valore restituito è la media delle norme dei singoli agenti.

Non è la somma delle distanze quadrate.
Non è normalizzata per il numero di celle.
Una griglia più grande tende a produrre valori di norma più grandi a parità di errore per cella.

`_drone_error_snapshot` produce un dizionario ID → norma.
`_print_error_snapshot` stampa media, massimo e dettagli per drone solo se `verbose` è attivo.
La diagnostica aiuta a distinguere convergenza globale apparente e agente divergente.

`_start_new_measurement_trace` chiude la traccia precedente se esiste.
Una traccia inizia con un valore dopo sensing e aggiunge valori dopo ciascun round di consenso.
La forma tipica contiene quattro campioni: dopo sensing e dopo tre round.

La visualizzazione della convergenza dispone una traccia fra due numeri di misura.
Il suo asse x è indice del ciclo di misura, non tempo fisico in secondi.
Le curve illustrano il cambiamento del disaccordo nel processo discreto.

### A.8.9 Gossip multi-hop di posizioni

`_sensed_position` usa una lettura GPS valida se già memorizzata.
Se manca, prova a richiedere una nuova stima.
In ultima istanza delega alla funzione di fallback del controller.

`_exchange_positions_multihop` registra una snapshot delle posizioni GPS prima di propagare.
Inizializza ogni drone con la propria posizione nota.
Inizializza anche la coordinata d'arco locale, se disponibile.

Il numero di iterazioni è almeno uno e al massimo pari al numero di droni.
Ad ogni iterazione vengono raccolti aggiornamenti pendenti per tutti gli agenti.
Gli aggiornamenti vengono applicati dopo aver visitato il vicinato del round.

L'uso di strutture `pending_updates` evita l'ordine di iterazione dal contaminare la stessa iterazione.
Le posizioni propagano i dati che un drone vicino già conosce.
Su un grafo statico connesso, un numero di passaggi pari al numero di nodi è sufficiente a propagare lungo cammini semplici.

La distanza di collegamento per la propagazione usa la snapshot iniziale del frame.
Non viene ricalcolata dopo ciascun hop.
Questo rappresenta più round di comunicazione su una topologia congelata per il passaggio.

### A.8.10 Applicazione delle azioni

Prima delle azioni, `_apply_actions` aggiorna le stime GPS.
I droni già in uno stato di contorno aggiornano la loro proiezione.
Se almeno un agente è fuori da mapping, il motore esegue il gossip posizioni.

I comandi vengono calcolati per tutti i droni prima di applicarli.
Mapping riceve coordinate del reticolo sorgente e può usare logica di esplorazione.
Gli altri stati chiamano il controller senza passare il campo o le coordinate sensoriali.

Il motore applica ogni comando con `dt` e limiti fisici della mappa.
Il comando viene limitato dalla velocità massima del drone.
Per agenti in modalità contorno, aggiorna nuovamente GPS e proiezione dopo il moto.

Il moto dei droni non aggiorna la posizione in modo simultaneo come sistema fisico accoppiato.
I comandi sono calcolati da snapshot prima di muovere gli agenti.
Ogni posizione è poi integrata una volta.

### A.8.11 Osservazione dello stato aggregato

`_observe_local_states` raccoglie gli stati degli agenti.
L'engine mantiene una stringa riassuntiva scegliendo lo stato più frequente.
In caso di parità, l'ordinamento lessicografico partecipa alla scelta.

La variabile di engine è per diagnostica e compatibilità.
Non comanda gli agenti e non li sincronizza forzatamente.
La transizione effettiva viene compiuta individualmente nei controller.

Il primo frame in cui un drone entra in Lloyd viene registrato in `transition_frame`.
Il primo frame in cui un drone entra in patrol viene registrato in `patrol_start_frame`.
`closed_boundary_points` copia il contorno valido con più punti fra quelli disponibili.

### A.8.12 Passo di simulazione completo

`step` incrementa il contatore `frame` per primo.
Poi legge gli stati locali correnti.
Il periodo delle misure durante mapping è rispettato con `(frame - 1) % measure_every == 0`.

Se nessun agente è più in mapping e la squadra non è vuota, ogni frame è un frame di misura.
La variabile `active_mapping_phase` nel codice indica solo che esistono agenti.
Non va confusa con la condizione che tutti siano ancora in mapping.

In ogni passo l'ambiente viene aggiornato prima di misure e mappa.
Il GPS viene aggiornato prima di calcolare il grafo e i controlli.
Le mappe in patrol decadono prima di una possibile misura del frame.

Nei frame di misura inizia una nuova traccia, si sensa e si registra l'errore.
Seguono i round di consenso anche nei frame non di misura, se la squadra esiste.
I round continuano quindi a diffondere una mappa invariata fra misure camera.

Terminato il consenso, l'engine calcola diagnostica e media.
Aggiorna gli stati aggregati prima e dopo il controllo.
Restituisce l'errore del frame al chiamante.

## A.9 `controller.py`: algoritmi di movimento e copertura

### A.9.1 Separazione tra controller comune e controller drone

La classe `Controller` implementa geometria del contorno e algoritmo di partizione.
La sottoclasse `DroneController` aggiunge parametri e politiche specifiche dell'agente.
I metodi statici di supporto sono riusabili senza istanziare un motore.

Il controller riceve una posizione stimata GPS attraverso `_estimated_position`.
Non riceve `world_field` nella fase Lloyd o patrol.
Le decisioni post-mapping si basano sulla curva stimata e sulle posizioni/arco conosciuti.

### A.9.2 Candidate locale di contorno

`_local_contour_candidate` richiede `contourpy`.
Se la dipendenza non è disponibile, restituisce un contorno vuoto e area racchiusa zero.
Questo impedisce alla macchina di stato di passare a Lloyd tramite questa procedura.

La maschera usata è lo snapshot di bordo locale durante il round, se presente.
Fuori dal consensus-step usa `boundary_grid` corrente.
La maschera viene interpretata con soglia 0.5.

Il candidato è costruito trattando `False` come spazio potenzialmente libero.
La DFS parte dalle celle libere sui quattro lati della mappa.
Le celle `free` non raggiunte sono regioni racchiuse.

La dimensione dell'area racchiusa è confrontata con la soglia configurata.
La griglia usa connettività a quattro vicini per il flood fill.
Il contorno isolinea è poi costruito sull'area binaria racchiusa.

Il passo fisico per X e Y viene ricavato dai limiti e dalla forma della mappa.
L'isolinea più lunga fra quelle con almeno quattro vertici viene scelta.
Una ripetizione del primo punto alla fine viene eliminata.

Sono richiesti almeno otto campioni per accettare la proposta.
Il metodo restituisce anche il numero di celle racchiuse, utile in diagnostica.
Una macchia mappata parzialmente o una barriera interrotta può produrre candidati non validi.

### A.9.3 Selezione e propagazione del contorno

`consensus_metadata_step` calcola prima una proposta dai dati locali.
Confronta la proposta con contorni ricevuti per verificare se sono già equivalenti.
La comparazione richiede la stessa forma e coordinate entro tolleranza `atol=0.25`.

La qualità è la media di informazione nelle celle attraversate dai punti di contorno.
Le coordinate dei punti vengono convertite in indici con un arrotondamento e un offset di mezzo pixel.
Gli indici vengono limitati ai limiti della griglia.

Una proposta locale forma la chiave `(epoch, quality, leader_id)`.
Un epoch è il frame corrente quando è fornito dal motore.
La chiave più grande secondo confronto lessicografico è mantenuta.

L'ordine delle chiavi dà precedenza all'epoca prima della qualità.
Quindi una proposta più recente può prevalere anche se ha qualità inferiore.
La qualità e l'ID leader risolvono i confronti con epoca uguale.

I messaggi di contorno vengono esaminati uno per uno.
Un messaggio deve dichiarare `closure_detected` e includere `contour_points`.
Il contorno migliore viene copiato localmente insieme ai suoi metadati.

La prontezza di pattuglia ricevuta da un vicino viene acquisita.
Un drone con contorno valido incrementa una volta per frame il contatore di chiusura.
Due frame consecutivi portano il drone a salvare quel bordo nel proprio controller e a passare a Lloyd.

Se `patrol_ready` è vero e il contorno è valido, lo stato passa a `boundary_patrol`.
Questa decisione locale può avvenire in un frame diverso tra agenti.
La comunicazione diffonde lo stato di readiness, non un comando centrale di transizione.

### A.9.4 Ordinamento dei punti del contorno

`set_known_boundary` memorizza i punti e i flag di chiusura/ordinamento.
Il contorno derivato da `contourpy` è già ordinato lungo l'isolinea.
I punti esportati a fine run seguono un percorso di ordinamento diverso, descritto più avanti.

`_ensure_ordered_closed_boundary` controlla il flag `known_boundary_ordered`.
Se i punti non sono ordinati, calcola la distanza al vicino più prossimo per ogni punto.
Usa un ordinamento greedy che parte dall'indice zero.

Questa procedura è una riparazione euristica.
Non garantisce il ciclo geometrico ottimo per point cloud arbitrari.
I bordi provenienti dall'isolinea di contourpy evitano normalmente di dover usare il fallback.

### A.9.5 Proiezione sulla curva

`_boundary_arc_lengths` calcola lunghezze dei segmenti consecutivi.
L'array cumulativo assegna a ogni vertice un arco `s`.
Per un ciclo chiuso, aggiunge la distanza tra ultimo e primo al perimetro totale.

`_arc_length_at_position` esamina i segmenti uno alla volta.
Proietta la posizione sul segmento con parametro limitato a `[0,1]`.
Sceglie la proiezione con distanza quadratica minima.

Il risultato include arco, posizione proiettata e indice approssimato.
Per il segmento di chiusura tra ultimo e primo usa la lunghezza del perimetro.
L'arco viene ridotto modulo la lunghezza totale se il bordo è chiuso.

`update_boundary_projection` salva arco e indice senza cambiare posizione fisica.
Aggiorna anche la mappa `known_boundary_arcs` del drone per includere il suo arco corrente.
`project_drone_to_boundary` può invece impostare la posizione fisica al punto proiettato.

### A.9.6 Multi-source Dijkstra

`multi_source_shortest_path_voronoi` riceve una lista di punti ordinati e semi.
Ogni seme identifica un drone e un indice nel contorno.
Un heap di priorità avvia una propagazione dai semi con costo zero.

I nodi sono punti discreti del contorno.
Gli archi collegano punti consecutivi.
Su un ciclo, il primo e l'ultimo sono anche vicini.

Il costo dell'arco è distanza euclidea cartesiana tra i due punti.
La distanza accumulata approssima la lunghezza d'arco.
Ogni punto viene marcato con il drone il cui seme offre il cammino più breve.

Se due semi cadono sullo stesso indice, la condizione di inizializzazione determina quale entra per primo.
La coda include ordine del seme per stabilizzare i confronti di priorità.
L'algoritmo restituisce un proprietario per nodo e la distanza dal seed proprietario.

Il costo computazionale è quello di Dijkstra su un grafo ad anello molto sparso.
Il codice usa heap generico e non sfrutta una formula chiusa della catena.
La scelta rende la procedura estendibile a pesi di segmento non uniformi.

### A.9.7 Target di Lloyd in lunghezza d'arco

`_lloyd_targets_from_seed_arcs` ordina i droni per il loro arco.
Con un unico seme, l'intero perimetro appartiene a quel drone.
Il target coincide con il suo arco corrente.

Per contorni aperti, i confini tra celle sono i punti medi dei semi adiacenti.
Le estremità della curva delimitano la prima e l'ultima cella.
Il target è il punto medio in lunghezza della cella assegnata.

Per contorni chiusi, i gap sono calcolati modulo la lunghezza totale.
La cella del drone comprende metà del gap precedente e metà del gap successivo.
Il target è il centro della cella risultante.

Il target in arco viene poi interpolato tra vertici con `_point_at_arc_length`.
L'interpolazione è lineare sul segmento che contiene l'arco richiesto.
Il punto risultante è una coordinata mondo continua.

La partizione è a pesi geometrici uniformi del contorno.
Non è pesata per densità di rischio, qualità del sensore o velocità del bordo.
Ogni drone tende a possedere una lunghezza simile solo quando i semi sono distribuiti coerentemente.

### A.9.8 `compute_ring_ordering`

Il metodo richiede un bordo noto e almeno un punto.
Prima assicura un ordine sequenziale.
Poi calcola lunghezza d'arco, lunghezza totale e chiusura.

I seed provengono da `current_drone.known_positions`.
Se il relativo arco è noto, viene usato direttamente.
Altrimenti la posizione GPS viene proiettata sulla curva.

I seed sono ordinati per coordinata d'arco.
Dijkstra assegna tutti i punti a un drone.
Per ogni agente vengono creati indici assegnati, target, cell length e limiti della cella.

Il risultato include dati di ring per tutti i droni noti.
Include anche predecessore e successore in ordine d'arco.
Queste relazioni descrivono il vicinato lungo il contorno, distinto dal vicinato radio.

Per l'agente corrente vengono aggiornati `target_centroid`, `boundary_s` e `boundary_index`.
La struttura completa è conservata in `last_ring_info`.
La visualizzazione di debug usa questa struttura per mostrare semi e target.

### A.9.9 Azioni di mapping

`_exploration_action` normalizza la direzione d'esplorazione.
Se la direzione non è valida, ne estrae una casuale.
Il vettore viene moltiplicato per `exploration_speed`.

`_grid_target` cerca le celle di bordo se la griglia di bordo non è vuota.
Altrimenti usa come fallback le celle occupate della mappa probabilistica.
Converte i centri cella in coordinate mondo e sceglie il punto più vicino al GPS.

`_mapping_action` non usa automaticamente un bordo remoto come tangente locale.
Se il drone ha edge points della propria ultima camera, chiama boundary tracking.
Se conosce solo un bordo tramite consenso, prima si muove verso il target condiviso.

Quando non esiste una mappa utile, esplora lungo l'heading corrente.
Il comando viene limitato a `max_speed` prima di essere restituito.
Il moto può cambiare heading d'esplorazione quando riflette contro una parete.

### A.9.10 Tangente da point cloud e boundary tracking

`_point_cloud_tracking_action` filtra i punti non finiti.
Ordina i rimanenti per distanza dalla posizione stimata.
Usa fino a 16 punti vicini per stimare la tangente con PCA/SVD.

La prima componente singolare descrive la direzione di massima variazione locale.
Se la PCA non produce una tangente valida, usa una perpendicolare al vettore normale.
Se anche quel fallback fallisce, il metodo restituisce `None`.

Il verso della tangente è ambiguo matematicamente.
Il controller lo sceglie in modo da non invertire bruscamente rispetto alla tangente precedente.
Al primo campione confronta la tangente con l'ultimo comando o heading d'esplorazione.

L'errore del bordo è la differenza tra punto edge più vicino e posizione stimata.
La parte tangenziale viene separata dalla normale.
La correzione normale è limitata a 0.07 unità prima di essere pesata.

Il controller aggiunge un lookahead di 0.45 unità lungo la tangente.
Se trova punti in avanti entro una banda laterale e un intervallo longitudinale, usa il punto candidato più vicino al lookahead.
L'azione combina velocità tangenziale, correzione edge e lookahead.

I guadagni impostati nella sottoclasse sono `edge_tracking_gain=0.85` e `edge_lookahead_gain=0.30`.
La velocità tangenziale nominale è `exploration_speed=0.12`.
Il vettore finale viene limitato a `max_speed=0.12`.

`_boundary_tracking_action` preferisce il point cloud della mappa locale.
Se non esiste, prova i punti camera grezzi.
Se nessuna delle due sorgenti produce una direzione valida, usa il gradiente dell'immagine come fallback.

Il gradiente di fallback è calcolato su una patch centrale `3 × 3` con maschere Sobel.
La normale deriva dal gradiente di concentrazione.
La tangente è la normale ruotata di 90 gradi.

La correzione dipende dall'errore tra concentrazione centrale e soglia.
Il fallback non usa le coordinate dell'ambiente passate come argomenti, perché `world_field`, `x_coords` e `y_coords` vengono eliminati.
Le unità dei gradienti sono scalate con la spaziatura memorizzata della camera.

### A.9.11 Azione Lloyd e condizione di readiness

`_lloyd_action` ricalcola l'ordinamento del ring a ogni chiamata.
Imposta il nome del control mode in base allo stato.
In `dynamic_lloyd` misura l'errore d'arco verso il target per ogni entry.

La distanza è circolare: confronta i due versi lungo il perimetro e sceglie il minore.
Se l'errore massimo non supera 0.25, incrementa `_patrol_ready_count`.
Altrimenti azzera il contatore.

Dopo otto successi consecutivi il drone imposta `patrol_ready=True`.
Il suo stato locale diventa `boundary_patrol`.
La transizione può essere ricalcolata durante lo stesso passaggio di controllo.

La funzione `_equidistant_action` attira il drone verso `target_centroid`.
L'errore viene moltiplicato per `k_t`.
Il comando è limitato alla velocità massima configurata.

### A.9.12 Controllo di pattuglia

`_patrol_action` richiede almeno tre punti di contorno.
Se il contorno è insufficiente torna al controllo Lloyd.
Se la lunghezza totale è praticamente zero restituisce un vettore nullo.

Usa l'arco GPS proiettato come posizione corrente lungo il bordo.
Legge gli archi dei vicini noti e calcola una velocità di platooning.
Il target è l'arco corrente più velocità per `dt`, ridotto modulo perimetro.

`_point_at_arc_length` converte il target in coordinate cartesiane.
Il controller salva `patrol_target_s` e `target_centroid` per diagnostica.
Il comando è proporzionale alla differenza tra target e posizione GPS corrente.

`_platooning_speed` ordina gli archi locali e conosciuti.
Determina distanze circolari al drone avanti e al drone dietro.
Calcola `0.05 + 0.08 * (d_front - d_back)` e limita il risultato tra 0.01 e 0.10.

Questa formula accelera quando lo spazio davanti supera quello dietro.
Rallenta quando il drone è più vicino al predecessore che al successore.
Non implementa una legge di inseguimento con accelerazione o collision avoidance esplicita.

## A.10 `visualization.py`: traduzione dello stato in viste

### A.10.1 Responsabilità del visualizzatore

Il visualizzatore riceve `SimulationMap`, eventualmente il modello della macchia, e gli stati dei droni.
Non invia comandi al motore.
Le sue funzioni disegnano e salvano informazioni già calcolate dagli altri moduli.

Il costruttore crea la figura principale e, facoltativamente, una seconda figura di debug.
La figura principale contiene un asse del mondo e un asse laterale predisposto ma nascosto.
L'asse mondo mantiene la stessa scala su x e y per non deformare la forma geometrica della macchia.

La visualizzazione iniziale disegna il campo dell'olio come immagine in scala di grigi.
Sovrappone isolinee di concentrazione 0.1, 0.5 e 0.9.
Il drone può così essere confrontato visivamente con il campo di verità della simulazione.

Il titolo distingue modalità fully connected e raggio di comunicazione.
Il cerchio di comunicazione viene disegnato intorno ai droni quando l'opzione è attiva.
La presenza del cerchio è una visualizzazione del parametro, non una verifica fisica del canale.

### A.10.2 Rendering del campo

`_draw_environment` valuta il campo sugli stessi `X` e `Y` dell'ambiente.
Passa `field_data.T` a `imshow` per allineare gli indici alla convenzione grafica.
L'origine viene fissata in basso a sinistra.

`update_environment` aggiorna i pixel dell'immagine con il campo corrente.
Rimuove le vecchie curve di livello e le ricrea sul nuovo campo.
Questo è necessario quando la macchia si muove o cambia area.

La funzione di rimozione gestisce differenze tra versioni Matplotlib.
Prova a rimuovere l'oggetto contour e poi le singole collection.
Le eccezioni di cleanup sono ignorate per consentire il rendering del frame successivo.

### A.10.3 Overlay del bordo di occupancy

`update_occupancy_boundary_overlay` prende `mean_boundary_grid` dallo snapshot engine.
Converte gli indici in centri di celle mondo.
Disegna i punti come quadrati azzurri sopra l'immagine ambientale.

L'overlay è derivato dalla media delle mappe, signal e informazioni.
Non rappresenta direttamente i punti del contorno metadata che ha attivato Lloyd.
Due visualizzazioni del bordo possono dunque differire durante il consenso o con copertura incompleta.

### A.10.4 Overlay della partizione Voronoi

`_ring_data_from_drones` cerca il primo drone con dati di ring validi.
Il ring contiene punti di contorno e assegnazioni ai droni.
Il visualizzatore non ricalcola la partizione.

`update_voronoi_map_overlay` divide i segmenti del bordo per proprietario.
Se i due estremi di un segmento appartengono a droni diversi, divide il segmento al punto medio.
Assegna un colore stabile per drone a partire dalla palette `tab10`.

Il flag `show_2d_voronoi` abilita l'overlay sul piano.
Il nome si riferisce alla resa in coordinate 2D.
L'assegnazione sottostante resta calcolata su un bordo 1D.

La figura debug mostra per ogni drone la propria vista locale.
Durante mapping mostra la griglia occupancy del singolo agente.
Durante Lloyd mostra l'asse di lunghezza d'arco, i semi e i target delle celle.

### A.10.5 Rappresentazione degli agenti

`update_drone` gestisce gli artist grafici appartenenti al drone.
Prima elimina gli oggetti grafici precedenti, poi crea quelli del frame corrente.
Questo evita di accumulare un artista nuovo a ogni aggiornamento e degradare il rendering.

Il drone è disegnato nella posizione corrente.
Etichette e colore identificano l'agente.
Vettori di controllo e centroidi/target possono essere disegnati per spiegare l'azione.

I punti edge possono essere mostrati con l'opzione `--show-nls-points`.
Questa vista è utile per diagnosticare errori GPS, bordo rumoroso o tangent estimation.
Non cambia il controllo o la mappa.

La finestra debug crea una griglia di subplot con massimo tre colonne.
Per ogni drone in mapping visualizza `grid.T` in scala di grigi.
Per i droni in Lloyd visualizza la partizione dal punto di vista del contorno ordinato.

### A.10.6 Rendering e cattura frame

`render` estrae dal dizionario `simulation_data` campo, droni e overlay.
Aggiorna gli artist esistenti e poi richiede il disegno delle canvas.
Il parametro `pause` mantiene compatibilità con le chiamate preesistenti.

Se la registrazione è attiva, `_capture_animation_frame` copia la canvas RGB.
Il frame viene catturato a ogni `animation_frame_stride` chiamate render.
Se esiste una finestra debug, viene catturata anche quella.

Le immagini restano in memoria finché non viene salvata o cancellata l'animazione.
Un run lungo con stride 1 può richiedere molta memoria.
Lo stride è il parametro previsto per ridurre il numero di immagini memorizzate.

### A.10.7 Esportazione PNG e GIF

`_ensure_output_dir` crea la cartella di destinazione se manca.
`_output_path` costruisce il percorso assoluto del file.
Le directory di output vengono quindi create automaticamente.

`_save_grid_plot` soglia la mappa e disegna una vista binaria.
Salva una colorbar, titoli ed etichette di indici griglia.
La figura temporanea viene chiusa al termine del salvataggio.

`save_final_occupancy_grid_per_robot` salva un'immagine separata per ogni drone.
Ogni immagine mostra la mappa locale finale.
La differenza tra questi file può aiutare a vedere se il consenso ha propagato informazione.

`plot_consensus_convergence` preferisce la cronologia per ciclo di misura quando disponibile.
Traccia la norma di disaccordo per ogni drone nei round successivi alla misura.
Se non esiste tale cronologia, disegna il solo errore per frame.

`_save_frames_as_gif` richiede Pillow.
La durata di un frame è ricavata dall'inverso di fps.
Il default di `loop=0` richiede un'animazione in loop infinito.

### A.10.8 Interpretare correttamente le figure

Il campo grigio è la simulazione vera, non la mappa stimata dai droni.
Le celle blu rappresentano invece il bordo aggregato percepito.
La vista debug mostra la memoria locale di ogni agente.

Un bordo colorato con l'overlay Voronoi indica il drone assegnatario per quel tratto.
Non mostra la connettività radio.
Un drone può essere vicino lungo il contorno ma non rientrare nel raggio radio, o viceversa.

Una mappa locale più vuota non implica necessariamente che il drone non abbia visto l'olio.
Potrebbe non aver ancora ricevuto le mappe dei vicini.
Può anche avere informazione dissolta durante il patrol.

## A.11 Flussi di dati e contratti tra moduli

### A.11.1 Flusso dal modello alla mappa locale

`OilSpill.field` produce valori per un insieme di coordinate.
`SimulationEngine.world_field` conserva il campionamento sul reticolo fisico.
`Drone.sense` passa il campionamento alla camera usando gli assi del reticolo.

`CameraSensor.sense` estrae e degrada un footprint locale.
`detect_edges` produce una maschera booleana di contorno in indici.
`extract_edge_points` e la conversione del sensore restituiscono coordinate mondo.

Il drone conserva l'immagine e i punti.
`update_grid` rasterizza ogni pixel valido nella mappa meno fine.
La mappa locale e la maschera di bordo sono ricalcolate.

### A.11.2 Flusso da mappe locali alla mappa distribuita

Il drone serializza snapshot mediante `create_consensus_message`.
Il motore costruisce il vicinato in base al GPS e copia i messaggi prima di applicarli.
Ogni drone fonde il messaggio proprio e quelli dei vicini.

Dopo tre round, il motore ricostruisce una mappa media per diagnostica e grafica.
Le mappe individuali restano mantenute su ciascun drone.
La mappa media del motore non è inviata come nuovo messaggio centralizzato.

Questa distinzione separa calcolo distribuito di mappa e visualizzazione centralizzata.
Il motore aggrega una vista globale per calcolare errori e disegnare.
Il controllore di ogni drone riceve invece la propria memoria e i messaggi vicini.

### A.11.3 Flusso dei metadati di contorno

La mappa locale induce un bordo occupancy.
Il rilevatore verifica se c'è una regione libera chiusa.
La curva candidata e le sue qualità entrano nel messaggio metadata.

Il consenso metadata diffonde la proposta migliore attraverso i vicini.
Un agente con una proposta stabile per due frame entra in Lloyd.
La posizione proiettata sul contorno diventa un seme della partizione.

La readiness di Lloyd si diffonde a sua volta.
Quando un drone è pronto, i vicini ricevono l'indicazione nei successivi round.
Il gruppo può quindi avere per alcuni frame agenti in stati diversi.

### A.11.4 Cosa viene condiviso

`grid` contiene probabilità di occupazione.
`signal_grid` e `information_grid` permettono di ricostruire la probabilità senza perdere lo stato ignoto.
`closure_detected` segnala una proposta topologica valida.

`contour_quality` fornisce una statistica di qualità informativa.
`contour_leader_id` e `contour_epoch` permettono confronto deterministico delle proposte.
`contour_points` è la geometria ordinata che alimenta Lloyd.

`patrol_ready` comunica il completamento della convergenza locale.
Il messaggio include `sender_id` per associare payload e agente.
Il payload è costituito da copie NumPy, quindi lo scambio ha costo proporzionale alla dimensione delle griglie.

### A.11.5 Cosa non viene condiviso

Non viene trasmessa la ground truth `world_field` nei messaggi tra droni.
Non viene trasmessa l'intera immagine camera come payload di consenso.
Gli edge points locali non sono inclusi nella struttura standard del messaggio.

Viene trasmessa la mappa rasterizzata e l'eventuale contorno stimato.
Non sono simulate dimensione dei pacchetti o capacità del canale.
Il costo di rete è quindi concettuale, non valutato in byte o latenza.

## A.12 Matrice delle fasi e transizioni

| Stato locale | Input principale del controller | Azione caratteristica | Condizione di avanzamento |
| --- | --- | --- | --- |
| `mapping` senza bordo locale | direzione di esplorazione e mappa condivisa | esplorare oppure dirigersi verso il target noto | nessuna transizione basata solo sul tempo |
| `mapping` con bordo camera | point cloud locale di bordo | tracking tangente con correzione normale | proposta di contorno chiuso stabile |
| `dynamic_lloyd` | contorno ordinato, GPS e archi dei droni | movimento verso il centro della cella Voronoi 1D | errore massimo <= 0.25 per 8 chiamate |
| `boundary_patrol` | coordinata d'arco e archi dei vicini | avanzamento ciclico con velocità platooning | stato operativo finale del ciclo corrente |

La fase si riferisce allo stato individuale, non a un enum unico gestito centralmente.
Il motore stampa una sintesi dello stato più frequente.
Il comportamento per-drone può divergere se manca connettività o se i messaggi non arrivano.

Una proposta di contorno è specifica di una snapshot di mappa.
Quando la mappa cambia, il codice può produrre una nuova curva candidata.
Il confronto fra epoch e qualità decide quale proposta sopravvive al consenso.

Durante patrol, il forgetting agisce sulle mappe.
La geometria di contorno metadata non viene automaticamente ricostruita come ground truth.
Il drone continua a osservare ogni frame e le nuove misure possono aggiornare la propria mappa.

## A.13 File di test e proprietà esercitate

La cartella `tests/` documenta con esempi eseguibili alcune proprietà desiderate.
Il README descrive lo scopo visibile dei test, senza implicare che siano stati eseguiti in questa modifica documentale.
Le fixture comuni si trovano in `tests/conftest.py`.

`test_decentralized_consensus.py` controlla che il receiver usi il messaggio snapshot.
Verifica inoltre la fusione di osservazioni concordanti e conflittuali.
Il caso conflittuale osservato da entrambi produce una probabilità media di 0.5.

`test_state_machine_transition.py` verifica diffusione del contorno e transizione distribuita.
Include anche un caso di componenti radio disconnesse che si sincronizzano quando il grafo diventa connesso.
Altri casi esercitano il feedback di velocità nel ring e il drone isolato in pattuglia.

`test_voronoi_mssp_static.py` copre partizioni su curve aperte e chiuse.
Verifica target Lloyd, assegnazioni e comandi limitati.
I casi sintetici isolano la geometria dagli effetti del sensore.

`test_dynamic_oil_spill.py` esercita traslazione, espansione e selezione seeded del drift.
Verifica anche il rifiuto di velocità negative.
Questi test riguardano il modello ambientale e helper di inizializzazione.

`test_sensor_ground_truth_boundaries.py` verifica che la camera venga centrata sulla stima GPS.
Controlla che il controller non ricada sulla posizione vera quando manca una stima GPS valida.
Controlla che l'esportazione mapping non legga la geometria ground truth della macchia.

`test_controller_boundary_mode.py` verifica ricerca del bordo da mappa condivisa.
Controlla che il target di consenso reindirizzi la direzione di esplorazione.
I test sono unitari e usano spesso droni fittizi per controllare casi limitati.

## A.14 Guida alla lettura del log

All'inizio il programma stampa il seed e se la visualizzazione è attiva.
L'intestazione riepiloga geometria, dinamica, frequenza misure e modello di comunicazione.
Queste righe sono utili da copiare nel registro di esperimento.

`Frame N [measurement]` indica un frame con acquisizione sensoriale durante il mapping.
`Frame N [consensus]` indica un frame senza nuova camera ma con round di consenso.
Dopo il passaggio di fase il tipo frame può mostrare lo stato aggregato dominante.

`After sensing` mostra il disaccordo dopo aggiornamenti locali.
`Consensus iteration i/3` mostra l'errore dopo il round indicato.
La riduzione tra questi punti fornisce una misura immediata dell'effetto dell'averaging.

Le righe `edge_points` e `nearest_edge` mostrano osservazioni sensoriali correnti.
Zero edge points non indica necessariamente guasto del sensore.
Il drone può essere in acqua libera, in una regione di concentrazione quasi costante o in un'immagine rifiutata dal controllo contrasto.

La sintesi `modes` stampa la politica scelta da ogni drone.
`explore` corrisponde a esplorazione libera.
`consensus_seek` indica che il drone sta andando verso un bordo noto nella mappa condivisa.

`boundary_tracking` indica che il drone sta usando un bordo locale osservato.
`dynamic_lloyd` indica che il controller sta spostando il drone verso il target del ring.
`boundary_patrol` indica avanzamento lungo il bordo con velocità di platooning.

La diagnostica finale stampa errore iniziale, finale e minimo.
Il minimo può comparire prima della fine, ad esempio se la dinamica del grafo successivamente cambia.
Un errore di disaccordo basso non prova da solo una buona stima dell'olio.

## A.15 Piano di analisi quantitativa per il report

Il codice fornisce direttamente cronologia di disaccordo, mappe finali e punti di bordo.
Questi output permettono analisi di consenso e confronto qualitativo.
Il repository non implementa una suite completa di metriche di segmentazione rispetto alla ground truth.

Per una valutazione di accuratezza è possibile confrontare la mappa media finale con il campo del modello.
Questa comparazione deve essere eseguita come analisi offline e dichiarata come uso della ground truth di simulazione.
Non va descritta come parte della percezione decentralizzata a bordo.

Una metrica IoU può confrontare maschere thresholded:

```text
IoU = |MappaStimata ∩ MappaVera| / |MappaStimata ∪ MappaVera|
```

Si può misurare precisione e recall delle celle occupate.
Una precisione alta significa pochi falsi positivi.
Un recall alto significa pochi falsi negativi.

Per la geometria, si può calcolare la distanza di Hausdorff tra contorno stimato e vero.
Per robustezza, si può ripetere l'esperimento su seed distinti.
Per gli effetti di comunicazione, si possono confrontare grafo fully-connected e range-based mantenendo gli altri parametri fissi.

Un esperimento riproducibile dovrebbe registrare seed, numero di droni, frequenza di misura, raggio radio, rumore e dinamica olio.
Il default della CLI può essere usato come baseline.
Cambiare un parametro alla volta rende più leggibile l'attribuzione degli effetti.

Per valutare consenso, riportare l'errore per round e frame.
Per valutare mapping, riportare una metrica rispetto al ground truth.
Per valutare coverage, riportare distribuzione delle lunghezze d'arco assegnate e massimo errore verso i centroidi Lloyd.

Per valutare patrol, riportare gap tra droni lungo il bordo nel tempo.
Una formazione uniforme tende ad avere deviazione standard dei gap bassa.
La velocità del ring può essere analizzata insieme alle distanze tra predecessori e successori.

Queste metriche supplementari sono suggerimenti per l'analisi del corso.
Non sono automaticamente prodotte o salvate dalla CLI corrente.
Se vengono aggiunte al codice, devono essere separate dalle metriche già esistenti nel report.

## A.16 Checklist concettuale per spiegare il sistema

Descrivere prima la differenza tra mondo simulato e mondo percepito.
Il simulatore conosce il campo esatto.
Il drone ottiene soltanto una finestra locale rumorosa e una stima GPS rumorosa.

Descrivere poi la rappresentazione occupazionale e l'incertezza.
Signal e information sono variabili ausiliarie che mantengono la semantica delle celle sconosciute.
La maschera di bordo viene derivata da celle occupate note vicine ad acqua nota.

Spiegare che il consenso opera su un grafo di comunicazione.
Nel caso a raggio, il grafo deriva dalle posizioni GPS e può essere disconnesso.
La media viene aggiornata iterativamente, tre round per frame nella configurazione standard.

Separare consenso sulle mappe e consenso sui metadati della curva.
Il primo fonde signal e informazione.
Il secondo propaga contorni candidati e stato di readiness.

Descrivere la chiusura come una proprietà inferita della mappa discreta.
La procedura cerca celle libere non raggiungibili dall'esterno.
La soglia di area racchiusa filtra cicli spurii piccoli.

Descrivere Lloyd come problema 1D lungo una geometria ordinata.
Il Multi-Source Dijkstra assegna ogni campione del bordo al seme più vicino lungo la curva.
I target sono i centri delle celle di Voronoi misurati in lunghezza d'arco.

Descrivere patrol come un comportamento successivo alla convergenza approssimata.
La velocità cambia in funzione degli spazi longitudinali fra vicini.
Il forgetting fa decadere progressivamente la memoria occupazionale durante l'inseguimento.

Terminare con le ipotesi e i limiti.
La rete è ideale, la dinamica del drone è cinematica e la camera non è prospettica.
Le metriche di accuratezza rispetto alla verità devono essere calcolate separatamente.

---
