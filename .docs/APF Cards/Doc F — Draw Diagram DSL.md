**Mini-DSL (optional input for Draw cards)**

```
#drawdsl
TITLE: Retrofit HTTP path
NODES: ViewModel, UseCase, Repository, Retrofit, OkHttp, Server
EDGES: ViewModel->UseCase->Repository->Retrofit->OkHttp->Server
NOTES: "Interceptors: logging, auth, cache"
```

**Rendering rules**

* Convert nodes/edges to a single inline `data:image/svg+xml` diagram.
* Monospace labels; ‰9 nodes; avoid overlapping text; provide concise `alt` text.
