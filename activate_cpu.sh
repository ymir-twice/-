#! /bin/bash

for i in {13..43}; do
    echo 1 | sudo tee /sys/devices/system/cpu/cpu$i/online
done
