# ml_pix2pix
#### _Projekat za kurs Mašinskog učenja, školska 2023/2024 godina_

## Skup podataka
Korišćen je [facades](https://cmp.felk.cvut.cz/~tylecr1/facade/) skup podataka.
```
@INPROCEEDINGS{Tylecek13,
  author = {Radim Tyle{\v c}ek, Radim {\v S}{\' a}ra},
  title = {Spatial Pattern Templates for Recognition of Objects with Regular Structure},
  booktitle = {Proc. GCPR},
  year = {2013},
  address = {Saarbrucken, Germany},
}
```

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

Modeli su sačuvani na: https://drive.google.com/drive/folders/1dW6UztwjdhsJ0O2_kP__kaRWcO9fWKb0?usp=sharing

## Literatura
Za izradu projekta korišćeno je:
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004)
* [pix2pix model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master)

Dodatna literatura:
* [Skip connections](https://theaisummer.com/skip-connections/)

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
