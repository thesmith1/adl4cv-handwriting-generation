#!/usr/bin/env bash
# Frontend
curl -sL https://deb.nodesource.com/setup_11.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @angular/cli
sudo npm install --save @angular/material @angular/cdk @angular/animations