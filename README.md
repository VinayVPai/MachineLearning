* [Pre-reading Material](https://github.com/soumendra/deep-learning-bdap/blob/master/resources/resources.md)
* [Slides](https://slides.com/soumendrap/deck-3/)
* [Guide to setting up anaconda environments](https://github.com/soumendra/python-machinelearning-setup/blob/master/setup_environments.md)

# Setting Up

## Ubuntu

```bash
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libboost-all-dev
sudo apt-get install python-dev

sudo apt-get install ubuntu-restricted-extras
sudo apt-get install ubuntu-restricted-addons
sudo apt-get install vlc
sudo apt-get install mencoder mplayer
```

## macos

* xcode
* xcode clt
* homebrew

```bash
brew install boost --with-python
brew install boost-python
brew list | grep 'boost'
    > boost
    > boost-python
brew tap homebrew/science
brew install opencv
```

## Installing the conda environment

After cloning this repo:

```bash
conda env create -f cv.yml
```
