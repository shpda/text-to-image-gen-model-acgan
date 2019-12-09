#!/usr/bin/env bash
# downloads the data for imagenet-small

#mkdir datasets
cd datasets
wget https://cs236-data.s3-us-west-1.amazonaws.com/tier1.zip
unzip tier1.zip -d tier1
mv tier1 ImageNet32
rm tier1.zip
