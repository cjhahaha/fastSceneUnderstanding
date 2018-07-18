#!/bin/bash

cp ../input/image/img.jpg ../output/image/img.jpg
rm *.txt

luajit main.lua

luajit process.lua
