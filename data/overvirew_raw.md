
the raw file overview 
 Design overview

- **Real dataset:** CIFAR-10 via torchvision.
- **Exports to disk:** Images saved under data/raw/{train,val,test}/{class}/image_XXXX.png.
- **Reproducible splits:** Deterministic with seed, configurable ratios.
- **Manifests:** JSON files with paths and labels for easy debugging and audits.
- **CLI:** Run with a single command and get clear output.

---

 Directory structure created

- **Base:** data/raw
- **Splits:** train, val, test
- **Classes:** one directory per class
- **Manifests:** data/raw/manifests/{train,val,test}.json
- **Stats:** data/raw/stats.json
