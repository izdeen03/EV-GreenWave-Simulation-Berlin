# EV-GreenWave-Simulation-Berlin
SUMO-based simulation study for emergency vehicle (EV) priority management on Sonnenallee, Berlin-Neukölln. This repository contains the implementation files, TraCI scripts, and analysis tools developed for a bachelor thesis focused on evaluating rule-based signal preemption and green-wave coordination strategies to improve emergency response

## Voraussetzungen

- Python 3.12.4 oder höher 
- SUMO 1.19.0 oder höher
- Git 

## Installation links
https://www.python.org/downloads/

https://sumo.dlr.de/docs/Installing/index.html#windows

https://git-scm.com/downloads



## Installation

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/izdeen03/EV-GreenWave-Simulation-Berlin/EV-GreenWave-Simulation-Berlin.git
    cd EV-GreenWave-Simulation-Berlin
    ```

2. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Install SUMO:**
    - Follow the instructions on the official [SUMO Webseite](https://sumo.dlr.de/docs/Installing/) to install SUMO.

## Usage

1. **Select file:**
    - Select a file from the `Project files` directory.

2. **Run SUMO configuration file:**
    
    - Run the .sumocfg files with SUMO:
        ```sh
        sumo-gui -c Project files/berlin_original.sumocfg
        ```

3. **Wait for the simulation:**
    - Wait until the simulation is fully completed.

4. **Save results:**
    - After the simulation is completed, you will receive a message asking whether you want to generate the output (statistic files).



## Data sources
OpenStreetMap (OSM) Data for the road network : https://www.openstreetmap.org/export#map=18/52.48639/13.42930.

Traffic statistics of our location : https://fbinter.stadt-berlin.de/fb/index.jsp?loginkey=zoomStart&mapId=k_vmengen2019@senstadt&bbox=392873,5816101,393662,5816630.

Bus plan : https://www.bvg.de/de/verbindungen/linienuebersicht/m41#liniensteckbrief-als-pdf

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests accordingly.

## License

This project is licensed under the MIT License. For more information, see the [LICENSE](LICENSE) File.

