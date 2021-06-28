# ncsn
Score-based generative modeling using NoiseConditionalScoreNetworks

Install [convae](https://github.com/rromb/convae):
```bash
git clone git@github.com:rromb/convae.git
pip install -e convae
```

Run streamlit script for unconditional image generation:
```bash
streamlit run unconditional_sampling.py -- -r <log-dir>
```

Some log-directories:
* LSUN churches: ``/export/scratch/jtraub/logs/ncsn/2021-06-26T12-45-49_lsun_vesde``
* FacesHQ: ``/export/scratch/jtraub/logs/ncsn/2021-06-26T13-23-06_faceshq_vesde_medium``
