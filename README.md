# ml_pix2pix
#### _Projekat za kurs Mašinskog učenja, školska 2023/2024 godina_

## Skup podataka
Korišćen je facades skup podataka sa [ove lokacijje](https://www.google.com/url?q=http%3A%2F%2Fefrosgans.eecs.berkeley.edu%2Fpix2pix%2Fdatasets%2F)

## Pokretanje projekta

Korišćeni paketi:
* pytorch
* matplotlib
* requests
* shutil

### Struktura projekta
 U `src` folderu se nalaze pomoćne skripte:
 * `discriminator.py` - opisuje arhitekturu diskriminatora
 * `generator.py` - opisuje arhitekturu generatora
 * `utils.py` - skripta za iscrtavanje slika i grafikona
 * `process_data.py` - učitavanje podataka
 * `losses.py` - ižračunavanje funkcije greške
Jupyter sveske numerisane od 1 do 4 demonstriraju svaku od pomoćnih skripti.

Jupyter sveske numerisane od 5 do 7 prikazuju proces učenja modela za vrednosti parametra lambda 50, 75 i 100.
Model je treniran u GoogleColab okruženju.

## Literatura
Za izradu projekta korišćeno je:
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004)
* [pix2pix model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master)

Dodatna literatura:
* [Skip connections](https://theaisummer.com/skip-connections/)
