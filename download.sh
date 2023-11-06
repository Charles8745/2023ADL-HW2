#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W-8NmQ3GYWYQrBQ1WEq-NrYjcHDwrQT5" -O ADLHW2_beam_50k.zip && rm -rf /tmp/cookies.txt

unzip ADLHW2_beam_50k

rm -rf ADLHW2_beam_50k.zip


