# JetTagging med EAAN (Edge-based Attention Aggregation Network)

Dette prosjektet implementerer en komplett pipeline for jet-klassifisering i høyenergifysikk, basert på partikkelnivådata og struktur-informert læring. Vi bruker en egendefinert EAAN-modell (Edge-based Attention Aggregation Network), inspirert av moderne arkitekturer som ParticleNet.

## 🛠️ Oppsett og kjøring

### 1. Last ned datasettet
```bash
python download_files.py
```

### 2. Konverter ROOT → HDF5
```bash
python convert_file.py
```

### 3. Tren modellen
```bash
python trainer.py
```

### 4. Evaluer modellen
```bash
python testing.py
```

### 5. Feature importance
```bash
python permutation_importance.py
```

## 💡 Modellarkitektur

EAAN er bygget opp av:
- EdgeConv-blokker med dynamisk KNN
- Attention-pooling (Graph Attention)
- SE-gating (Squeeze-and-Excitation)
- Residual shortcuts
- Multiskala-fusjon (PointNet++)
- Hard top-k pooling (Graph U-Net)
- Global aggregering og fullt klassifiseringshode

Modellen støtter GPU-akselerasjon, `torch.compile`, mixed precision og streaming fra disk.

## 📚 Inspirasjon og referanser

EAAN-modellen er inspirert av følgende arbeider:

- [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772)
- [ParticleNet (jet-tagging via particle clouds)](https://doi.org/10.1103/physrevd.101.056019)
- [Dynamic Graph CNN for Point Clouds](https://arxiv.org/abs/1801.07829)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [PointNet++ (multiskalafusjon)](https://arxiv.org/abs/1706.02413)
- [Graph U-Net (hard top-k-pooling)](https://arxiv.org/abs/1905.05178)


---

## ⚙️ Konfigurasjonsmuligheter

Konfigurasjonen styres gjennom `config.py`. Her er noen av de viktigste valgene:

- `pad_len`: Maks antall partikler per event. Må samsvare med treningsdata.
- `data_format`: `'channel_last'` eller `'channel_first'`, avhengig av modellarkitektur.
- `stream`: Hvis `True`, lastes data én og én batch direkte fra disk. Sparsomt minnebruk.
- `device`: Bruk `"cuda"` for GPU-trening. Anbefales sterkt.
- `batch_size`, `epochs`, `lr`, `weight_decay`: Vanlige treningsparametere.
- `label_smooth`: Smoothing for robustere klassifisering.
- `run_name`: Navn på eksperimentet – styrer hvor resultater lagres.

Modellen støtter fleksibel arkitektur og kan utvides med flere EdgeConv-blokker, flere FC-lag eller forskjellige pooling-mekanismer.

---

## ⚠️ Viktig anbefaling

Det **anbefales sterkt å trene på GPU**. EAAN er en dyp og kompleks modell med flere EdgeConv-blokker og attention-lag. CPU-trening fungerer, men vil være svært treg, spesielt på store datasett.

Hvis du har en moderne NVIDIA-GPU, vil du også få:
- Automatisk støtte for mixed precision (AMP)
- Optimalisert ytelse med `torch.compile`
- Effektiv batch-prosessering med `pin_memory=True`

---