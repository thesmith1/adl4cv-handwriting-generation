import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import {MatButtonModule, MatButtonToggleModule, MatCardModule, MatInputModule} from '@angular/material';

@NgModule({
  declarations: [],
  imports: [
    CommonModule,
    MatInputModule,
    MatButtonToggleModule,
    MatButtonModule,
    MatCardModule
  ],
  exports: [
    MatInputModule,
    MatButtonToggleModule,
    MatButtonModule,
    MatCardModule
  ]
})
export class AngularMaterialModule { }
