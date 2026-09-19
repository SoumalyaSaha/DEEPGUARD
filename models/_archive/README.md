# Archived models

## fire/ (archived, not deleted)
FIRE (Chuchad/FIRE, CVPR 2025, imagenet+adm checkpoint kept alongside as
`fire_imagenet_adm.pth`) excluded due to constant near-0/1 output driven by
input resize scale, not content — zero discriminative signal. Service ran on
port 5006. NOTE: `main.py` still defaults `WEIGHTS_PATH` to
`../../weights/`; point it at the local copy if ever revived.
