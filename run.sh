#!/bin/bash
./aps -f 0
mv L.exr L0.exr
mv T.exr T0.exr

./aps -f 1
mv L.exr L1.exr
mv T.exr T1.exr

./sphere -r 0.1
mv img.exr red1.exr
./sphere -r 0.6
mv img.exr red2.exr

./sphere -r 0.1 -a 1
mv img.exr white1.exr
./sphere -r 0.6 -a 1
mv img.exr white2.exr

