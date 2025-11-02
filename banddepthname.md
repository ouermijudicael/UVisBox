# Band Depth Naming Convention Fix

## Issue
The functional boxplot pipeline has inconsistent naming. The correct abbreviation should be **fbd** (functional band depth), not **fdb**.

## Files with Issues

### 1. `/uvisbox/Modules/FunctionalBoxplot/functional_boxplot_mesh.py`

**Line 4 - Function parameter default:**
```python
# INCORRECT
def get_band(data, percentile, method='fdb'):

# SHOULD BE
def get_band(data, percentile, method='fbd'):
```

**Line 16 - Docstring:**
```python
# INCORRECT
- 'fdb': functional band depth (default)

# SHOULD BE
- 'fbd': functional band depth (default)
```

**Line 17 - Docstring:**
```python
# INCORRECT
- 'mfdb': modified functional band depth

# SHOULD BE
- 'mfbd': modified functional band depth
```

**Line 29 - Example in docstring:**
```python
# INCORRECT
>>> bottom, top = get_band(data, 50, method='fdb')  # 50th percentile band

# SHOULD BE
>>> bottom, top = get_band(data, 50, method='fbd')  # 50th percentile band
```

### 2. `/uvisbox/Modules/FunctionalBoxplot/functional_boxplot_vis.py`

**Line 47 - Example in docstring:**
```python
# INCORRECT
>>> bottom, top = get_band(data, 50, method='fdb')

# SHOULD BE
>>> bottom, top = get_band(data, 50, method='fbd')
```

### 3. `/examples/functional_boxplot_example.py`

**Line 39 - Docstring example:**
```python
# INCORRECT
ax3 = functional_boxplot(X, method='mfdb', ax=ax3)

# SHOULD BE
ax3 = functional_boxplot(X, method='mfbd', ax=ax3)
```

**Line 82 - Title comment:**
```python
# INCORRECT (though this is just a display title)
ax2.set_title("Functional Boxplot (FDB) with Outliers")

# SHOULD BE
ax2.set_title("Functional Boxplot (FBD) with Outliers")
```

## Files that are CORRECT

### `/uvisbox/Modules/FunctionalBoxplot/functional_boxplot_stats.py`
- Line 7: Comment uses correct `fbd`
- Line 8: Comment uses correct `mfbd`
- Line 11: Function parameter default is correct `method='fbd'`
- Line 21-22: Docstring uses correct `'fbd'` and `'mfbd'`
- Line 24, 26, 29: Code logic uses correct values

### `/uvisbox/Modules/FunctionalBoxplot/functional_boxplot.py`
- Line 9: Function parameter default is correct `method='fbd'`
- Line 23-24: Docstring uses correct `'fbd'` and `'mfbd'`
- Line 91, 94: Code logic uses correct values

### `/examples/custom_boxplot_style_example.py`
- All instances use correct `method='fbd'`

## Summary of Changes Needed

| File | Line(s) | Change |
|------|---------|--------|
| `functional_boxplot_mesh.py` | 4 | `method='fdb'` → `method='fbd'` |
| `functional_boxplot_mesh.py` | 16 | `'fdb':` → `'fbd':` |
| `functional_boxplot_mesh.py` | 17 | `'mfdb':` → `'mfbd':` |
| `functional_boxplot_mesh.py` | 29 | `method='fdb'` → `method='fbd'` |
| `functional_boxplot_vis.py` | 47 | `method='fdb'` → `method='fbd'` |
| `functional_boxplot_example.py` | 39 | `method='mfdb'` → `method='mfbd'` |
| `functional_boxplot_example.py` | 82 | `(FDB)` → `(FBD)` |

**Total: 7 changes across 3 files**
