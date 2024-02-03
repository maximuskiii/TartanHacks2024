# 

```mermaid
flowchart TB
    A[App] -->|Coordinates| B[Imaging/Contour Detection Module];
    subgraph Maxim
    B -->|Coordinates| C[Google Maps API];
    C -->|Imagery| B
    B -->|Path contours| D[Flight Path Planning Module]
    B -->|Path contours, statellite imagery| E[Orientation and Surface Pathfinding Module]
    end
    subgraph Julius
    D -->|Flight path| F[Drone]
    F -->|Video| G[Crowd Detection Module]
    end
    subgraph Ben
    G -->|Images, crowd map| E
    end
    subgraph Alon
    E -->|Optimal ground path, drone localization visualization| A
    end
```