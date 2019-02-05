import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {GlobalVars} from '../global-vars';
import {Observable} from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ToBackendService {
  private globalVars: GlobalVars;

  constructor(private _http: HttpClient) {
    this.globalVars = new GlobalVars();
  }

  private static getHeaders(): HttpHeaders {
    return new HttpHeaders().set('Content-Type', 'application/json');
  }

  private doRequest(body: any, api: string): Observable<Object> {
    return this._http.post(this.globalVars.backendUrl + api, body, {headers: ToBackendService.getHeaders()});
  }

  // API
  public generateImageFromString(text: string, style: number, index: number): Observable<Object> {
    const body = {
      text: text,
      style: style,
      index: index
    };
    return this.doRequest(body, '/insert');
  }
  public reset(): Observable<Object> {
    const body = {};
    return this.doRequest(body, '/reset');
  }
}
