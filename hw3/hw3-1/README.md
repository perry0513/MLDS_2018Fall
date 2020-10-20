# MLDS_hw3-1
## Hw3-1
### Prerequisites
    $ pip3 install tensorflow
    $ pip3 install numpy
    $ pip3 install opencv-python
    $ pip3 install matplotlib
    $ pip3 install tqdm

### Format
    > hw3-1
        > AnimeDataset/
        > model/
        > dcgan.py
        > wgan.py
        > gan.py
        > data_processor.py
        > main.py

### Installing
#### Specify `-h` for help
    $ python3 main.py -h
#### E.g. Train dcgan for 50 epochs from model ./model/dcgan_30 and save model as dcgan_80
    $ python3 main.py -type dcgan -model_dir ./model/dcgan_30 -save_model dcgan_80 -epoch 50

### Links
* data_link: https://drive.google.com/file/d/1tpW7ZVNosXsIAWu8-f5EpwtF3ls3pb79/view
* model_link: https://drive.google.com/open?id=1XghiPSW7T0H0AM02393hjRlB2WyOh4rI
* ppt_link: https://drive.google.com/open?id=1wWDJEfLbXRzEOt-f3hLmJypOR6nqsZEjU55EbB5SmDY
