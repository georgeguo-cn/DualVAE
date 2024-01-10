## DualVAE (SDM'24)

This is the Pytorch implementation for our SDM 2024 paper:
>Zhiqiang Guo, Guohui Li, Jianjun Li, Chaoyang Wang, Si Shi. DualVAE: Dual Disentangled Variational AutoEncoder for Recommendation. In SDM 2024. [Paper](#)

### Data  

The interaction data is shared at `data/`.

### Training logs and models

The logs and parameters are shared at `log/` and `models/`, respectively.

### Environment

    pip install -r requirements.txt

### Run

Run `train.sh` to train DualVAE: 

    bash train.sh

You may specify other parameters in `train.sh`.

### Citation

    @inproceedings{guo2024dualvae,
    author = {Zhiqiang Guo, Guohui Li, Jianjun Li, Chaoyang Wang, Si Shi},
    title = {DualVAE: Dual Disentangled Variational AutoEncoder for Recommendation},
    booktitle = {Proceedings of SDM},
    pages = {xxx},
    year = {2024}
    }

