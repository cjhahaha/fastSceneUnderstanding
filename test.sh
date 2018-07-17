#!/bin/bash

cp ../input/image/img.jpg ../output/image/img.jpg

luajit main.lua

luajit process.lua
