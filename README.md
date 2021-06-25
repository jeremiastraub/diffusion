# ncsn
Score-based generative modeling using NoiseConditionalScoreNetworks

Install [convae](https://github.com/rromb/convae):
```bash
git clone git@github.com:rromb/convae.git
pip install -e convae
```

Run streamlit script for unconditional image generation:
```bash
streamlit run unconditional_sampling.py -- --config configs/latent/ve/LSUNchurch_small.yaml
```

