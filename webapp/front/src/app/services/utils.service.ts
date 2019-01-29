import { Injectable } from '@angular/core';
import {DomSanitizer, SafeResourceUrl} from '@angular/platform-browser';

@Injectable({
  providedIn: 'root'
})
export class UtilsService {

  constructor(private _sanitizer: DomSanitizer) {}

  public wait(ms: number): Promise<number> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  public sanitizeImageBase64(resource: string): SafeResourceUrl {
    return this._sanitizer.bypassSecurityTrustResourceUrl('data:image/jpg;base64,' + resource);
  }
}
