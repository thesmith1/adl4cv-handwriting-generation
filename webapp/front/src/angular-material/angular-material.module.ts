import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import {MatButtonModule, MatButtonToggleModule, MatInputModule} from '@angular/material';

@NgModule({
  declarations: [],
  imports: [
    CommonModule,
    MatInputModule,
    MatButtonToggleModule,
    MatButtonModule
  ],
  exports: [
    MatInputModule,
    MatButtonToggleModule,
    MatButtonModule
  ]
})
export class AngularMaterialModule { }
