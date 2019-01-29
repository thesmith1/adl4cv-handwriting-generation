import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import {AngularMaterialModule} from '../angular-material/angular-material.module';
import {ToBackendService} from './services/to-backend.service';
import {HttpClientModule} from '@angular/common/http';
import {UtilsService} from './services/utils.service';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    HttpClientModule,
    AngularMaterialModule
  ],
  providers: [
    ToBackendService,
    UtilsService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
