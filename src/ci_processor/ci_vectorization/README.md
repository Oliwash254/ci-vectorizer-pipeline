# elpi-vectorizer

Vectorize analog electrodogram recordings

See the manual [online](https://git.web.rug.nl/dBSPL/elpi-vectorizer/src/branch/main/docs/build/markdown/index.md).

## Installation

Clone the repository from https://git.web.rug.nl/dBSPL/elpi-vectorizer and install with:

```
pip install -e ./elpi-vectorizer
```

## Usage

```python
import pickle
from vectorizer import ConstantPulseVectorizerAB

fname = "pa_10-30.pickle"
dat = pickle.load(open(fname, 'rb'))

X = dat['X']
fs = dat['__info__']['fs']

V = ConstantPulseVectorizerAB(X, fs)
pulse_times, pulse_amplitudes, pulse_prms = V1.vectorize()
```
