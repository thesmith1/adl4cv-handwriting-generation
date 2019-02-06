#!/usr/bin/env bash
# Frontend
curl -sL https://deb.nodesource.com/setup_11.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install @angular/cli --save-dev
sudo npm install --save @angular/material @angular/cdk @angular/animations
sudo npm install --save-dev @angular-devkit/build-angular
