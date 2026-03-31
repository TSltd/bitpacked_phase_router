## 1. Generate ORIGINAL dataset

```bash
CXXFLAGS="-DFORCE_ORIGINAL" pip install -e .
python evaluation/compare_original_vs_interval.py --tag original
```

---

## 2. Generate INTERVAL dataset

```bash
CXXFLAGS="-DFORCE_INTERVAL" pip install -e .
python evaluation/compare_original_vs_interval.py --tag interval
```

---

## 3. Fit weights

```bash
python evaluation/fit_dispatch_weights.py
```

### Copy weights into C++

e.g:

```
cfg.a = 0.681453; // prefers interval for large N
cfg.b = -2.065438; // favors higher density
cfg.c = -3.792435; // favors large k/N
cfg.d = -3.445288; // bias toward original
cfg.e = -0.597596; // interaction term
```

---

## 4. Restore normal build

```bash
pip install -e .
```

(no FORCE flags)

---
